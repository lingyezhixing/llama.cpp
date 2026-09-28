param(
    [string]$BinDir = 'D:\LLM\Backend\llama.cpp-my',
    [string]$Model  = '<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf',
    [int]$Port = 8945,
    [int]$NMax = 3,
    [int]$Rounds = 3,
    [int]$TreeRam = 2048,
    [int]$TreeDiskLimit = 8192
)

$ErrorActionPreference = 'Stop'
$env:CUDA_VISIBLE_DEVICES = '1'
$env:GGML_CUDA_GDN_REPLAY = '1'

$out = '<TEMP>\v100\t32-27b'
Remove-Item -Recurse -Force $out -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $out | Out-Null
$dir  = "$out\tree"
$save = "$out\slot-save"
New-Item -ItemType Directory -Force $save | Out-Null

$sargs = @('-m',$Model,'-np','2','-ngl','99','-fa','on','-ctv','q8_0','-ub','512','-c','65536','--seed','42',
           '--cache-ram','0','--no-context-shift','--host','127.0.0.1','--port',"$Port",'--cache-idle-slots',
           '--slot-save-path',$save,
           '--spec-type','draft-mtp','--spec-draft-n-max',"$NMax",
           '--kv-tree','--tree-chunk','512','--tree-checkpoint-anchor-step','32768','--tree-checkpoint-fork-step','8192',
           '--tree-ram',"$TreeRam",'--tree-disk',$dir,'--tree-disk-limit',"$TreeDiskLimit",'--tree-debug')
Write-Output ("[27b] starting: np=2 ctx=65536 n_max={0} tree_ram={1}MiB tree_disk={2}MiB" -f $NMax, $TreeRam, $TreeDiskLimit)
$p = Start-Process -FilePath (Join-Path $BinDir 'llama-server.exe') -ArgumentList $sargs -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput "$out\srv-out.txt" -RedirectStandardError "$out\srv-err.txt"

$mon = Start-Job -ScriptBlock {
    param($log, $dir)
    while ($true) {
        Start-Sleep -Seconds 10
        try {
            $stats = (Get-Content $log -Tail 500 -ErrorAction SilentlyContinue | Select-String 'kv tree stats:' | Select-Object -Last 1).Line
            $parked   = (Select-String -Path $log -Pattern 'kv tree: parked' -ErrorAction SilentlyContinue).Count
            $restored = (Select-String -Path $log -Pattern 'kv tree: restored' -ErrorAction SilentlyContinue).Count
            $miss     = (Select-String -Path $log -Pattern 'restore miss' -ErrorAction SilentlyContinue).Count
            $captured = (Select-String -Path $log -Pattern 'captured heal anchor' -ErrorAction SilentlyContinue).Count
            $fs = Get-ChildItem $dir -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum
            $vram = (& nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits) -replace '\s',''
            $line = "[mon] vram={0}MB disk={1}MB files={2} parked={3} restored={4} miss={5} captured={6} | {7}" -f `
                    $vram, [int]($fs.Sum/1MB), $fs.Count, $parked, $restored, $miss, $captured, $stats
            Add-Content -Path "$($dir)\..\monitor.txt" -Value $line -Encoding UTF8
        } catch {}
    }
} -ArgumentList "$out\srv-err.txt", $dir

function TokCount($text) {
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/tokenize" -Method Post -ContentType 'application/json' -TimeoutSec 120 -Body (@{content=$text} | ConvertTo-Json -Compress)
    return $r.tokens.Count
}
function Filler($salt, $n_tok) {
    $unit = "$salt filler sentence for the kv tree 27b test number "
    $ut = TokCount $unit
    $t = ''; $cur = 0
    while ($cur -lt $n_tok) { $need = [int][math]::Ceiling(($n_tok - $cur) / $ut); $t += ($unit * $need); $cur = TokCount $t }
    return $t
}

$ok = $false
for ($i = 0; $i -lt 600; $i++) {
    Start-Sleep -Seconds 2
    try { $h = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 3; if ($h.status -eq 'ok') { $ok = $true; Start-Sleep -Seconds 3; break } } catch {}
    if ($p.HasExited) { break }
}
if (-not $ok) { Write-Output 'SERVER FAILED'; Get-Content "$out\srv-err.txt" -Tail 30; Stop-Job $mon; Remove-Job $mon -Force; Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue; exit 1 }
Write-Output '[27b] server ready'

$sys = Filler 'system' 1024
$targets = @(12288, 14336, 16384, 10240)
$bodies = @()
$i = 0
foreach ($s in @('alpha','beta','gamma','delta')) {
    $bodies += (Filler $s $targets[$i]); $i++
}
Write-Output ("[27b] sessions: sys={0} bodies={1}" -f (TokCount $sys), (($bodies | ForEach-Object { TokCount $_ }) -join ','))

$rows = @()
for ($r = 1; $r -le $Rounds; $r++) {
    for ($si = 0; $si -lt $bodies.Count; $si++) {
        $prompt = "$sys`nUser: " + $bodies[$si] + "`nUser turn ${r}: continue`nAssistant:"
        $body = @{prompt=$prompt; n_predict=32; temperature=0; top_k=1; seed=42; cache_prompt=$true; stream=$false;
                  message_delimiters=@(@{role='user'; delimiter='User:'})}
        $w0 = Get-Date
        $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/completion" -Method Post -ContentType 'application/json' -TimeoutSec 3600 -Body ($body | ConvertTo-Json -Compress -Depth 4)
        $wall = [math]::Round(((Get-Date) - $w0).TotalSeconds, 1)
        $dn = 0; $da = 0
        if ($resp.timings.PSObject.Properties.Name -contains 'draft_n') { $dn = $resp.timings.draft_n }
        if ($resp.timings.PSObject.Properties.Name -contains 'draft_n_accepted') { $da = $resp.timings.draft_n_accepted }
        $row = [pscustomobject]@{ round=$r; sess=$si; prompt_n=$resp.timings.prompt_n; prompt_ms=[int]$resp.timings.prompt_ms; pred=$resp.timings.predicted_n; wall=$wall; cached=$resp.n_prompt_tokens_cache; draft_n=$dn; draft_acc=$da; len=$resp.content.Length }
        $rows += $row
        Write-Output ("[27b] r{0} s{1} prompt_n={2} prompt_ms={3} pred={4} wall={5}s cached={6} draft={7}/{8}" -f $r,$si,$row.prompt_n,$row.prompt_ms,$row.pred,$row.wall,$row.cached,$da,$dn)
    }
    if (Test-Path "$out\monitor.txt") { Get-Content "$out\monitor.txt" -Tail 6 -Encoding UTF8 | ForEach-Object { Write-Output $_ } }
}

$alive = -not $p.HasExited
Stop-Job $mon; Remove-Job $mon -Force
$fs = Get-ChildItem $dir -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum
$vram = (& nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits) -replace '\s',''
Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue; $p.WaitForExit(10000) | Out-Null

$log = Get-Content "$out\srv-err.txt" -Encoding UTF8
$parked   = ($log | Select-String 'kv tree: parked').Count
$restored = ($log | Select-String 'kv tree: restored').Count
$miss     = ($log | Select-String 'restore miss').Count
$captured = ($log | Select-String 'captured heal anchor').Count
$rebuilt  = ($log | Select-String 'rebuilt \d+ context checkpoints').Count
$fails    = ($log | Select-String 'failed to|GGML_ABORT|exceed_context_size').Count
$statsLine = ($log | Select-String 'kv tree stats:' | Select-Object -Last 1).Line
$diskMax = 0
foreach ($l in $log) { if ($l -match 'ram = (\d+) B, disk = (\d+) B') { $d = [int64]$matches[2]; if ($d -gt $diskMax) { $diskMax = $d } } }

Write-Output ''
Write-Output "27B SUMMARY alive=$alive parked=$parked restored=$restored miss=$miss captured=$captured rebuilt=$rebuilt fails=$fails vram=${vram}MB disk_final=$([int]($fs.Sum/1MB))MB files=$($fs.Count) disk_max=$([int]($diskMax/1MB))MB"
Write-Output "27B STATS $statsLine"
$r1 = ($rows | Where-Object { $_.round -eq 1 } | Measure-Object prompt_ms -Average).Average
$r23 = ($rows | Where-Object { $_.round -ge 2 } | Measure-Object prompt_ms -Average).Average
$n1 = ($rows | Where-Object { $_.round -eq 1 } | Measure-Object prompt_n -Average).Average
$n23 = ($rows | Where-Object { $_.round -ge 2 } | Measure-Object prompt_n -Average).Average
Write-Output ("27B REUSE round1 avg prompt_n={0} prompt_ms={1}; round2+ avg prompt_n={2} prompt_ms={3}" -f [int]$n1, [int]$r1, [int]$n23, [int]$r23)
$rows | ForEach-Object { Write-Output ("  r{0} s{1} prompt_n={2} prompt_ms={3} pred={4} wall={5} draft={6}/{7}" -f $_.round,$_.sess,$_.prompt_n,$_.prompt_ms,$_.pred,$_.wall,$_.draft_acc,$_.draft_n) }
