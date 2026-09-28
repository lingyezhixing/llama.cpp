param(
    [Parameter(Mandatory=$true)][string]$Mode,   # calib|ab|overlap|b|b3|neg|heal|ref|soak|fork
    [string]$Model = '<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf',
    [int]$Port = 8933,
    [string]$OutDir = '<TEMP>\v100\t32-stage3',
    [int]$Minutes = 5
)

$ErrorActionPreference = 'Stop'
$repo = 'D:\LLM\Backend\src\llama.cpp-my'
$srv  = "$repo\build\bin\Release\llama-server.exe"
New-Item -ItemType Directory -Force $OutDir | Out-Null
$env:CUDA_VISIBLE_DEVICES = '1'

$fails = 0
$script:srv_proc = $null

function Assert($cond, $what) {
    if ($cond) { Write-Output "PASS  $what" } else { Write-Output "FAIL  $what"; $script:fails++ }
}

function Stop-Srv {
    if ($script:srv_proc) {
        if (-not $script:srv_proc.HasExited) {
            Stop-Process -Id $script:srv_proc.Id -Force -ErrorAction SilentlyContinue
            $script:srv_proc.WaitForExit(10000) | Out-Null
        }
        $script:srv_proc = $null
    }
}

# ---- prompt construction (target ~N tokens, measured via /tokenize) ----
function TokCount($text) {
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/tokenize" -Method Post -ContentType 'application/json' -TimeoutSec 60 -Body (@{content=$text} | ConvertTo-Json -Compress)
    if ($null -eq $r.tokens) { throw 'unexpected /tokenize response shape' }
    return $r.tokens.Count
}
function Filler($salt, $n_tok) {
    $unit = "$salt filler sentence for the kv tree acceptance test number "
    $ut = TokCount $unit
    $t = ''
    $cur = 0
    while ($cur -lt $n_tok) {
        $need = [int][math]::Ceiling(($n_tok - $cur) / $ut)
        $t += ($unit * $need)
        $cur = TokCount $t
    }
    return $t
}

function Start-Srv($tree, $ram, $diskdir, $idle = $true, $anchor_step = 4096, $np = 1, $slot_save = '', $disk_mib = 2048, $tree_debug = $true, $ctx = 24576, $log_tag = '', $fork_step = 8192, $sim = -1.0) {
    $tag = $Mode; if ($log_tag -ne '') { $tag = $log_tag }
    $sargs = @('-m',$Model,'-np',"$np",'-ngl','99','-fa','on','-ctv','q8_0','-c',"$ctx",
              '--port',"$Port",'--cache-ram','0',
              '--no-context-shift','--host','127.0.0.1')
    if ($sim -ge 0.0) { $sargs += @('--slot-prompt-similarity', "$sim") }
    if ($idle) { $sargs += '--cache-idle-slots' } else { $sargs += '--no-cache-idle-slots' }
    if ($slot_save -ne '') { $sargs += @('--slot-save-path',$slot_save) }
    if ($tree) {
        $sargs += @('--kv-tree','--tree-chunk','512','--tree-checkpoint-anchor-step',"$anchor_step",
                   '--tree-checkpoint-fork-step',"$fork_step",
                   '--tree-ram',"$ram",'--tree-disk',$diskdir,'--tree-disk-limit',"$disk_mib")
        if ($tree_debug) { $sargs += '--tree-debug' }
    }
    $p = Start-Process -FilePath $srv -ArgumentList $sargs -NoNewWindow -PassThru `
         -RedirectStandardOutput "$OutDir\srv-$tag-out.txt" -RedirectStandardError "$OutDir\srv-$tag-err.txt"
    $script:srv_proc = $p
    for ($i = 0; $i -lt 120; $i++) {
        Start-Sleep -Milliseconds 1000
        if ($p.HasExited) { Stop-Srv; throw "server exited early (code $($p.ExitCode))" }
        try { Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -Method Get -TimeoutSec 3 | Out-Null; return $p } catch {}
    }
    Stop-Srv
    throw "server did not come up"
}

function Req($prompt, $cache) {
    $body = @{prompt=$prompt; n_predict=48; temperature=0; top_k=1; seed=42; cache_prompt=$cache; stream=$false}
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/completion" -Method Post -ContentType 'application/json' -TimeoutSec 300 -Body ($body | ConvertTo-Json -Compress)
    return $r
}

function ReqDelim($prompt, $cache) {
    $body = @{prompt=$prompt; n_predict=32; temperature=0; top_k=1; seed=42; cache_prompt=$cache; stream=$false; n_probs=5; message_delimiters=@(@{role='user'; delimiter='User:'})}
    return Invoke-RestMethod -Uri "http://127.0.0.1:$Port/completion" -Method Post -ContentType 'application/json' -TimeoutSec 120 -Body ($body | ConvertTo-Json -Compress -Depth 4)
}

# ---- session types (built after the server is up: Filler needs /tokenize) ----
# long sessions: shared system prompt only (low overlap -> f_keep < 0.5 -> tree park/restore)
function Build-Sys {
    if (-not $script:sys) { $script:sys = Filler 'system' 512 }
}
function Build-Long {
    Build-Sys
    if (-not $script:baseA) { $script:baseA = Filler 'alpha' 16384 }
    if (-not $script:baseB) { $script:baseB = Filler 'beta'  16384 }
}
# overlap sessions: share 12K (stock VRAM reuse; tree must stay out of the way)
function Build-Ovl {
    if (-not $script:ovl) { $script:ovl = Filler 'shared' 12288 }
    if (-not $script:baseC) { $script:baseC = "$script:ovl" + (Filler 'gamma' 4096) }
    if (-not $script:baseD) { $script:baseD = "$script:ovl" + (Filler 'delta' 4096) }
}
# short sessions for mode b
function Build-Sess {
    Build-Sys
    if (-not $script:sess) {
        $script:sess = @()
        foreach ($s in @('one','two','three','four')) { $script:sess += ("$script:sys" + (Filler $s 1536)) }
    }
}

$results = @{}
$hashes  = @{}
function ContentHash($s) {
    $sha = [System.Security.Cryptography.SHA256]::Create()
    $h = $sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($s))
    return ([System.BitConverter]::ToString($h)).Replace('-','').Substring(0,16)
}
function RunTurns($names, $bases, $rounds, $cache, $tag) {
    $hist = @{}
    foreach ($n in $names) { $hist[$n] = '' }
    for ($r = 1; $r -le $rounds; $r++) {
        for ($i = 0; $i -lt $names.Count; $i++) {
            $n = $names[$i]
            $prompt = "$sys`n" + $bases[$i] + $hist[$n] + "`nUser turn ${r}: continue`nAssistant:"
            $resp = Req $prompt $cache
            $hist[$n] += $resp.content
            $h = ContentHash $resp.content
            $hashes["$tag/$n/$r"] = $h
            $results["$tag/$n/$r"] = $resp.content
            Write-Output ("[$Mode/$tag] $n round ${r}: hash=$h prompt_n=$($resp.timings.prompt_n) pred=$($resp.timings.predicted_n) cached=$($resp.tokens_cached)")
        }
    }
}
function ComparePasses($names, $rounds) {
    for ($r = 1; $r -le $rounds; $r++) {
        foreach ($n in $names) {
            Assert ($hashes["tree/$n/$r"] -eq $hashes["full/$n/$r"]) "equal: $n round $r (tree reuse vs full prefill)"
        }
    }
}

try {
    switch ($Mode) {
        'ref' {
            $p = Start-Srv $false 0 '' $true
            Build-Long; Build-Ovl; Build-Sess
            RunTurns @('A','B') @($baseA,$baseB) 6 $false 'full'
            RunTurns @('C','D') @($baseC,$baseD) 3 $false 'full'
            RunTurns @('one','two','three','four') $sess 3 $false 'full'
            Stop-Srv
        }
        'calib' {
            $dir = "$OutDir\tree-calib"; Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
            $p = Start-Srv $true 4096 $dir
            Build-Long
            RunTurns @('A','B') @($baseA,$baseB) 2 $true 'tree'
            Stop-Srv
            $line = (Select-String -Path "$OutDir\srv-$Mode-err.txt" -Pattern 'kv tree: parked' | Select-Object -Last 1).Line
            Write-Output "CALIB LAST PARK: $line"
            $total = 0
            if ($line -match 'ram = (\d+) B, disk = (\d+) B') { $total = [int64]$matches[1] + [int64]$matches[2] }
            Write-Output "CALIB TOTAL BYTES AFTER 2 ROUNDS: $total"
            Assert ($total -gt 0) 'calib: tree stores data'
            Assert ((Select-String -Path "$OutDir\srv-$Mode-err.txt" -Pattern 'kv tree: restore miss').Count -ge 1) 'calib: cold restore miss is visible'
        }
        'ab' {
            $dir = "$OutDir\tree-ab"; Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
            $ram = 0; if ($env:T32_RAM_MIB) { $ram = [int]$env:T32_RAM_MIB }
            if ($ram -le 0) { $ram = 96 }
            $p = Start-Srv $true $ram $dir
            Build-Long
            RunTurns @('A','B') @($baseA,$baseB) 6 $false 'full'
            RunTurns @('A','B') @($baseA,$baseB) 6 $true  'tree'
            Stop-Srv
            $log = Get-Content "$OutDir\srv-$Mode-err.txt" -Encoding UTF8
            $parked = ($log | Select-String 'kv tree: parked').Count
            $restored = ($log | Select-String 'kv tree: restored').Count
            $miss = ($log | Select-String 'kv tree: restore miss').Count
            $diskmax = 0; $rammax = 0
            foreach ($l in $log) {
                if ($l -match 'ram = (\d+) B, disk = (\d+) B') {
                    $r = [int64]$matches[1]; $d = [int64]$matches[2]
                    if ($d -gt $diskmax) { $diskmax = $d }
                    if ($r -gt $rammax) { $rammax = $r }
                }
            }
            Write-Output "AB METRICS ram_mib=$ram parked=$parked restored=$restored miss=$miss rammax=$rammax diskmax=$diskmax"
            Assert ($parked -ge 6) 'ab: park ran on every switch'
            Assert ($restored -ge 8) 'ab: later turns restored from the tree'
            Assert ($miss -eq 0)   'ab: every tree-pass turn restored from the tree'
            Assert ($diskmax -gt 0) 'ab: SSD tier actually used (disk > 0)'
            ComparePasses @('A','B') 6
        }
        'overlap' {
            $dir = "$OutDir\tree-ovl"; Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
            $p = Start-Srv $true 96 $dir $false
            Build-Sys; Build-Ovl
            RunTurns @('C','D') @($baseC,$baseD) 3 $false 'full'
            RunTurns @('C','D') @($baseC,$baseD) 3 $true  'tree'
            Stop-Srv
            $log = Get-Content "$OutDir\srv-$Mode-err.txt" -Encoding UTF8
            $parked = ($log | Select-String 'kv tree: parked').Count
            $restored = ($log | Select-String 'kv tree: restored').Count
            $miss = ($log | Select-String 'kv tree: restore miss').Count
            Write-Output "OVL METRICS parked=$parked restored=$restored miss=$miss"
            Assert ($parked -eq 0)    'overlap: no park for high-overlap switches (idle park off)'
            Assert ($restored -eq 0)  'overlap: no restore for high-overlap switches'
            ComparePasses @('C','D') 3
        }
        'b' {
            $dir = "$OutDir\tree-b"; Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
            $p = Start-Srv $true 128 $dir
            Build-Sess
            RunTurns @('one','two','three','four') $sess 3 $false 'full'
            RunTurns @('one','two','three','four') $sess 3 $true  'tree'
            Stop-Srv
            $log = Get-Content "$OutDir\srv-$Mode-err.txt" -Encoding UTF8
            $restored = ($log | Select-String 'kv tree: restored').Count
            $ref4 = (($log | Select-String 'ref=') | Select-String 'ref=4').Count
            Write-Output "B METRICS restored=$restored ref4_lines=$ref4"
            Assert ($restored -ge 4) 'b: short sessions restore from the tree'
            Assert ($ref4 -ge 1) 'b: shared prefix stored once (refcount = 4)'
            ComparePasses @('one','two','three','four') 3
        }
        'b3' {
            $Model = '<models>\Qwen2.5-Coder-3B-IQ4_XS.gguf'
            $dir = "$OutDir\tree-b3"; Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
            $p = Start-Srv $true 512 $dir
            $sys3 = Filler 'system' 256
            $sys = $sys3
            $b1 = Filler 'aa' 4096; $b2 = Filler 'bb' 4096
            RunTurns @('P','Q') @($b1,$b2) 3 $false 'full'
            RunTurns @('P','Q') @($b1,$b2) 3 $true  'tree'
            Stop-Srv
            $log = Get-Content "$OutDir\srv-$Mode-err.txt" -Encoding UTF8
            $restored = ($log | Select-String 'kv tree: restored').Count
            $parked = ($log | Select-String 'kv tree: parked').Count
            $miss = ($log | Select-String 'kv tree: restore miss').Count
            $failed = ($log | Select-String 'failed to capture heal anchor').Count
            Write-Output "B3 METRICS parked=$parked restored=$restored miss=$miss failed_captures=$failed"
            Assert (($log | Select-String 'kv tree: parked').Count -ge 1) 'b3: tree still parks pure-attention content (D11: no fork reuse)'
            Assert ($failed -eq 0) 'b3: no failed captures'
            ComparePasses @('P','Q') 3
        }
        'neg' {
            $dir = "$OutDir\tree-neg"; Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
            $p = Start-Srv $true 96 $dir
            Build-Long
            RunTurns @('A','B') @($baseA,$baseB) 2 $true 'tree'
            $files = Get-ChildItem $dir -Recurse -File
            Write-Output "NEG removing $($files.Count) block files"
            $files | Remove-Item -Force
            RunTurns @('A') @($baseA) 1 $true 'tree-after-loss'
            Stop-Srv
            $log = Get-Content "$OutDir\srv-$Mode-err.txt" -Encoding UTF8
            $strict = (($log | Select-String 'failed to read block|restore failed').Count)
            $wide   = (($log | Select-String 'failed to read block|failed to read anchor|restore failed').Count)
            Write-Output "NEG METRICS strict=$strict wide=$wide"
            Assert ($wide -ge 1) 'neg: SSD read failure is visible'
            Assert ($results.ContainsKey('tree-after-loss/A/1')) 'neg: request still succeeded after the failure'
        }
        'heal' {
            $dir = "$OutDir\tree-heal"; Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
            # anchor_step 512, default fork_step: the guess at 487 must not suppress the fork capture at 1024 (D31)
            $p = Start-Srv $true 512 $dir $true 512
            $h  = "User: " + ('filler ' * 480) + "`nAssistant: ok`nUser: " + ('filler ' * 600) + "`nAssistant: ok`nUser: "
            $tA = 'alpha-tail ' * 450
            $tF = 'gamma-fork ' * 450
            $reqs = @("$h$tA", "$h$tF", ("Zeta: " + ('omega ' * 1500)), "$h$tF")
            # D10: run the tree pass first on a fresh tree so the first fork restore happens with cache_prompt=true
            foreach ($pass in @(@($true,'tree'), @($false,'full'))) {
                $i = 0
                foreach ($q in $reqs) {
                    $i++
                    $r = ReqDelim $q $pass[0]
                    $hsh = ContentHash $r.content
                    $hashes["$($pass[1])/heal/$i"] = $hsh
                    Write-Output ("[heal/$($pass[1])] req ${i}: hash=$hsh prompt_n=$($r.timings.prompt_n)")
                }
            }
            Stop-Srv
            $log = Get-Content "$OutDir\srv-$Mode-err.txt" -Encoding UTF8
            $captured  = ($log | Select-String 'captured heal anchor at \d+').Count
            $notstored = ($log | Select-String 'heal anchor at \d+ not stored').Count
            $failed    = ($log | Select-String 'failed to capture heal anchor').Count
            $missed    = ($log | Select-String 'heal position').Count
            $restored  = ($log | Select-String 'kv tree: restored').Count
            $parked    = ($log | Select-String 'kv tree: parked').Count
            $stored    = $false
            $capLine   = $log | Select-String 'captured heal anchor at (\d+)' | Select-Object -First 1
            if ($capLine -and $capLine.Line -match 'captured heal anchor at (\d+)') {
                $hp = $matches[1]
                $stored = (($log | Select-String "anchor [0-9a-f]{16}@$hp kind=2").Count -ge 1)
                Write-Output "HEAL ANCHOR pos=$hp stored_in_dump=$stored"
            }
            Write-Output "HEAL METRICS captured=$captured stored=$stored notstored=$notstored failed=$failed missed=$missed parked=$parked restored=$restored"
            $rebuilt = ($log | Select-String 'rebuilt \d+ context checkpoints').Count
            Write-Output "HEAL REBUILT lines=$rebuilt"
            Assert ($rebuilt -ge 1) 'heal: context checkpoints rebuilt after the fork restore (D12)'
            Assert ($captured -eq 1) 'heal: exactly one heal capture'
            Assert ($stored) 'heal: the captured anchor is genuinely stored (dump shows @pos kind=2)'
            Assert ($failed -eq 0) 'heal: no failed captures'
            for ($i = 1; $i -le 4; $i++) { Assert ($hashes["tree/heal/$i"] -eq $hashes["full/heal/$i"]) "heal: request $i identical (tree vs full prefill)" }
        }
        'soak' {
            $dir  = "$OutDir\tree-soak"; Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
            $save = "$OutDir\slot-save"; Remove-Item -Recurse -Force $save -ErrorAction SilentlyContinue
            New-Item -ItemType Directory -Force $save | Out-Null

            $np = 2
            # anchor_step 512 (not 4096): soak prompts are ~2.6K tokens; dense anchors are needed
            # for a deep restore to rebuild an anchor below its restore point (D12)
            $p = Start-Srv $true 64 $dir $true 512 $np $save 512 $false 16384
            $soak_pid = $p.Id

            Build-Sys

            $sessions = @()
            foreach ($s in @('alpha','beta','gamma','delta')) {
                $sessions += @{ body = (Filler $s 2048); hist = '' }
            }
            $fork = Filler 'shared' 1536
            $sessions += @{ body = ($fork + (Filler 'forkA' 1024)); hist = '' }
            $sessions += @{ body = ($fork + (Filler 'forkB' 1024)); hist = '' }

            $deadline = (Get-Date).AddMinutes($Minutes)
            $round = 0
            $cmp_ok = 0; $cmp_bad = 0; $cmp_tie = 0
            $io0 = $null; $rss0 = 0; $h0 = 0
            $last_prompt = ''

            while ((Get-Date) -lt $deadline) {
                $round++
                $si = $round % $sessions.Count
                $s  = $sessions[$si]

                # the "User:" delimiter (message_delimiters in ReqDelim) makes the server create a
                # checkpoint anchor in the shared prefix; that is what D12 rebuilds from
                $prompt = "$sys`nUser: " + $s.body + $s.hist + "`nUser turn ${round}: continue`nAssistant:"
                $last_prompt = $prompt

                $r = ReqDelim $prompt $true
                $s.hist += $r.content
                if ($s.hist.Length -gt 4000) { $s.hist = $s.hist.Substring($s.hist.Length - 2000) }

                if ($round % 5 -eq 0) {
                    $rf = ReqDelim $prompt $false
                    $h1 = ContentHash $r.content
                    $h2 = ContentHash $rf.content
                    if ($h1 -eq $h2) {
                        $cmp_ok++
                    } else {
                        # greedy near-ties can flip between the two computation paths (path-dependent
                        # fp accumulation, see T24): accept a flip when both paths' logits for the
                        # diverging token are within 0.05 nats; anything larger is a real mismatch
                        $ct = $r.completion_probabilities
                        $cf = $rf.completion_probabilities
                        $d = -1
                        $ncmp = [Math]::Min($ct.Count, $cf.Count)
                        for ($i = 0; $i -lt $ncmp; $i++) {
                            if ($ct[$i].token -ne $cf[$i].token) { $d = $i; break }
                        }
                        $tie = $false
                        if ($d -ge 0) {
                            $lt = $ct[$d].top_logprobs | Where-Object { $_.id -eq $cf[$d].id } | Select-Object -First 1
                            $lf = $cf[$d].top_logprobs | Where-Object { $_.id -eq $ct[$d].id } | Select-Object -First 1
                            if ($lt -and $lf) {
                                $m1 = [Math]::Abs($ct[$d].logprob - $lf.logprob)
                                $m2 = [Math]::Abs($cf[$d].logprob - $lt.logprob)
                                if ($m1 -lt 0.05 -and $m2 -lt 0.05) { $tie = $true }
                            }
                        }
                        if ($tie) {
                            $cmp_tie++
                            Write-Output "SOAK NEAR-TIE round=$round sess=$si at=$d tree='$($ct[$d].token)' full='$($cf[$d].token)' (logit delta < 0.05)"
                        } else {
                            $cmp_bad++
                            Write-Output "SOAK MISMATCH round=$round sess=$si tree=$h1 full=$h2"
                            $u8 = New-Object System.Text.UTF8Encoding($false)
                            [IO.File]::WriteAllText("$OutDir\soak-mismatch-round$round-prompt.txt", $prompt, $u8)
                            [IO.File]::WriteAllText("$OutDir\soak-mismatch-round$round-tree.txt", $r.content, $u8)
                            [IO.File]::WriteAllText("$OutDir\soak-mismatch-round$round-full.txt", $rf.content, $u8)
                            [IO.File]::WriteAllText("$OutDir\soak-mismatch-round$round-tree.json", ($r | ConvertTo-Json -Depth 8), $u8)
                            [IO.File]::WriteAllText("$OutDir\soak-mismatch-round$round-full.json", ($rf | ConvertTo-Json -Depth 8), $u8)
                        }
                    }
                }

                if ($round % 11 -eq 0) {
                    $es = [int]([math]::Floor($round / 11)) % $np
                    Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$Port/slots/$es`?action=erase" -TimeoutSec 60 | Out-Null
                }

                $io = Get-CimInstance Win32_Process -Filter "ProcessId=$soak_pid"
                if ($null -eq $io0) { $io0 = $io; $rss0 = $io.WorkingSetSize; $h0 = $io.HandleCount }

                $fs = Get-ChildItem $dir -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum
                Write-Output ("[soak] round=$round sess=$si prompt_n=$($r.timings.prompt_n) pred=$($r.timings.predicted_n) cached=$($r.tokens_cached) rss_mb=$([int]($io.WorkingSetSize/1MB)) handles=$($io.HandleCount) wr_mb=$([int]($io.WriteTransferCount/1MB)) rd_mb=$([int]($io.ReadTransferCount/1MB)) files=$($fs.Count) tree_mb=$([int]($fs.Sum/1MB))")
            }

            $alive = -not $p.HasExited
            $ioEnd = Get-CimInstance Win32_Process -Filter "ProcessId=$soak_pid"
            Stop-Srv

            $log = Get-Content "$OutDir\srv-$Mode-err.txt" -Encoding UTF8
            $parked   = ($log | Select-String 'kv tree: parked').Count
            $restored = ($log | Select-String 'kv tree: restored').Count
            $rebuilt  = ($log | Select-String 'rebuilt \d+ context checkpoints').Count
            $failed   = ($log | Select-String 'failed to').Count
            $evref    = ($log | Select-String 'eviction could not free enough ram').Count
            $diskmax  = 0
            foreach ($l in $log) {
                if ($l -match 'ram = (\d+) B, disk = (\d+) B') {
                    $d = [int64]$matches[2]; if ($d -gt $diskmax) { $diskmax = $d }
                }
            }
            Write-Output "SOAK METRICS rounds=$round parked=$parked restored=$restored rebuilt=$rebuilt failed=$failed evict_refused=$evref diskmax=$diskmax cmp_ok=$cmp_ok cmp_tie=$cmp_tie cmp_bad=$cmp_bad rss_mb=$([int]($rss0/1MB))->$([int]($ioEnd.WorkingSetSize/1MB)) handles=$h0->$($ioEnd.HandleCount) wr_mb=$([int]($ioEnd.WriteTransferCount/1MB)) rd_mb=$([int]($ioEnd.ReadTransferCount/1MB))"
            Assert ($alive) 'soak: server alive at the end of the run'
            Assert ($cmp_bad -eq 0) 'soak: sampled tree vs full prefill outputs are identical'
            Assert ($cmp_ok -ge 3) 'soak: at least three comparisons ran'
            Assert ($parked -ge 5) 'soak: parks ran repeatedly'
            Assert ($restored -ge 5) 'soak: restores ran repeatedly'
            Assert ($rebuilt -ge 1) 'soak: context checkpoints were rebuilt after a tree restore (D12)'
            Assert ($failed -eq 0) 'soak: no failures in the server log'
            # disk-full park refusal is a designed rollback (park_rollback); the rate bound stays valid on
            # 30/60 min runs and still catches leak regressions (pre-fix binary: 33 refusals in a 112-round run)
            Assert ($evref * 10 -le $round) 'soak: budget refusals stay rare (<= 10% of rounds)'
            Assert ($diskmax -le 512*1024*1024) 'soak: the disk tier stayed within the limit'
            Assert ($ioEnd.HandleCount -le $h0 + 100) 'soak: handle count stable'
            Assert ($ioEnd.WorkingSetSize -le $rss0 + 400MB) 'soak: RSS growth bounded'

            # crash-restart: stale files must be cleared and the server must work again
            $files_before = (Get-ChildItem $dir -Recurse -File -ErrorAction SilentlyContinue).Count
            $p2 = Start-Srv $true 64 $dir $true 512 $np $save 512 $false 16384 'soak-restart'
            $log2 = Get-Content "$OutDir\srv-soak-restart-err.txt" -Encoding UTF8
            $cleared = ($log2 | Select-String 'cleared \d+ stale files').Count
            $files_after = (Get-ChildItem $dir -Recurse -File -ErrorAction SilentlyContinue).Count
            Write-Output "SOAK RESTART files_before=$files_before cleared_lines=$cleared files_after=$files_after"
            Assert ($cleared -ge 1) 'soak: stale tree files cleared on restart (D13)'
            Assert ($files_after -eq 0) 'soak: tree disk is empty after restart'
            $r2 = ReqDelim $last_prompt $true
            Assert ($null -ne $r2.content -and $r2.content.Length -gt 0) 'soak: request succeeds after restart'
            Stop-Srv
        }
        'fork' {
            $dir = "$OutDir\tree-fork"; Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
            # no message delimiters -> no checkpoint guesses -> the first fork is a restore miss
            # ctx 32768: b1 is ~20500 tokens (the brief's 16384 rejected it); anchor_step stays 32768 so no regular anchor
            # sim 0: the stock slot-similarity shortcut would reuse the slot in-place and skip the tree (f_keep >= 0.5)
            $p = Start-Srv $true 512 $dir $true 32768 1 '' 2048 $false 32768 -sim 0
            Build-Sys
            $base = Filler 'shared' 8192
            $a1 = Filler 'branchA1' 8192
            $a2 = Filler 'branchA2' 4096
            $b1 = $base + $a1 + $a2                  # fork 1 at ~len(base)
            $b2 = $base + (Filler 'branchB' 2048)    # diverges at fork 1
            $b3 = $base + $a1 + (Filler 'branchC' 2048)  # diverges at ~fork 1 + 8192 (>= fork_step)

            $hashes = @{}
            foreach ($pass in @(@($true,'tree'), @($false,'full'))) {
                $i = 0
                foreach ($q in @($b1, $b2, $b2, $b3, $b3)) {
                    $i++
                    $r = Req $q $pass[0]
                    $hashes["$($pass[1])/fork/$i"] = (ContentHash $r.content)
                    Write-Output ("[fork/$($pass[1])] req ${i}: prompt_n=$($r.timings.prompt_n) cached=$($r.tokens_cached)")
                }
            }
            Stop-Srv

            $log = Get-Content "$OutDir\srv-$Mode-err.txt" -Encoding UTF8
            $captured = @($log | Select-String 'captured heal anchor at (\d+)' | ForEach-Object { [int]$_.Matches[0].Groups[1].Value })
            $restored = @($log | Select-String 'kv tree: restored (\d+) tokens' | ForEach-Object { [int]$_.Matches[0].Groups[1].Value })
            Write-Output "FORK METRICS captured=[$($captured -join ',')] restored=[$($restored -join ',')]"
            Assert ($captured.Count -ge 2) 'fork: two fork anchors captured (miss heal + second fork)'
            Assert ($restored -contains $captured[0]) 'fork: the miss-heal anchor is reused'
            Assert ($restored -contains $captured[1]) 'fork: the second fork anchor is reused'
            for ($i = 1; $i -le 5; $i++) {
                Assert ($hashes["tree/fork/$i"] -eq $hashes["full/fork/$i"]) "fork: request $i identical (tree vs full prefill)"
            }
        }
    }
} finally {
    Stop-Srv
}

Write-Output "RESULT ${Mode}: $fails failure(s)"
exit $fails
