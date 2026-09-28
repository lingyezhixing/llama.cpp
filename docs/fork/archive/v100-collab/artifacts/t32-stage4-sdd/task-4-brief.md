### Task 4: soak 模式 + smoke

**Files:**
- Modify: `D:\LLM\Backend\v100-collab\artifacts\t32-stage3-ab.ps1` (设备 1, `param`, `Start-Srv`, 新 helpers, `soak` 模式)
- 产物: `artifacts\t32-stage4-soak-smoke.txt` + `t32-stage4-logs\srv-soak-*.txt` (从 `$OutDir` 复制)

**Interfaces:**
- Consumes: 现有 `Assert/Stop-Srv/TokCount/Filler/Req/ContentHash` 与 Task 2 的 `cleared` 日志、Task 3 的 `rebuilt` 日志.
- Produces: `-Mode soak [-Minutes N]` 运行; 结束输出 `SOAK METRICS ...` 与 `RESULT soak: N failure(s)`.

- [ ] **Step 1: 脚本改动**

(1) 第 12 行: `$env:CUDA_VISIBLE_DEVICES = '0'` -> `'1'`.

(2) `param` 块加: `[int]$Minutes = 5`.

(3) `Start-Srv` 整体替换 (向后兼容旧调用, 新参数都有默认值):

```powershell
function Start-Srv($tree, $ram, $diskdir, $idle = $true, $anchor_step = 4096, $np = 1, $slot_save = '', $disk_mib = 2048, $tree_debug = $true, $ctx = 24576, $log_tag = '') {
    $tag = $Mode; if ($log_tag -ne '') { $tag = $log_tag }
    $sargs = @('-m',$Model,'-np',"$np",'-ngl','99','-fa','on','-ctv','q8_0','-c',"$ctx",
              '--port',"$Port",'--cache-ram','0',
              '--no-context-shift','--host','127.0.0.1')
    if ($idle) { $sargs += '--cache-idle-slots' } else { $sargs += '--no-cache-idle-slots' }
    if ($slot_save -ne '') { $sargs += @('--slot-save-path',$slot_save) }
    if ($tree) {
        $sargs += @('--kv-tree','--tree-chunk','512','--tree-anchor-step',"$anchor_step",
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
```

(4) `switch ($Mode)` 里加 `soak` 分支:

```powershell
        'soak' {
            $dir  = "$OutDir\tree-soak"; Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
            $save = "$OutDir\slot-save"; Remove-Item -Recurse -Force $save -ErrorAction SilentlyContinue
            New-Item -ItemType Directory -Force $save | Out-Null

            $np = 2
            $p = Start-Srv $true 64 $dir $true 4096 $np $save 512 $false 16384
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
            $cmp_ok = 0; $cmp_bad = 0
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
                    if ((ContentHash $r.content) -eq (ContentHash $rf.content)) { $cmp_ok++ } else { $cmp_bad++ }
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
            Write-Output "SOAK METRICS rounds=$round parked=$parked restored=$restored rebuilt=$rebuilt failed=$failed evict_refused=$evref diskmax=$diskmax cmp_ok=$cmp_ok cmp_bad=$cmp_bad rss_mb=$([int]($rss0/1MB))->$([int]($ioEnd.WorkingSetSize/1MB)) handles=$h0->$($ioEnd.HandleCount) wr_mb=$([int]($ioEnd.WriteTransferCount/1MB)) rd_mb=$([int]($ioEnd.ReadTransferCount/1MB))"
            Assert ($alive) 'soak: server alive at the end of the run'
            Assert ($cmp_bad -eq 0) 'soak: sampled tree vs full prefill outputs are identical'
            Assert ($cmp_ok -ge 3) 'soak: at least three comparisons ran'
            Assert ($parked -ge 5) 'soak: parks ran repeatedly'
            Assert ($restored -ge 5) 'soak: restores ran repeatedly'
            Assert ($rebuilt -ge 1) 'soak: context checkpoints were rebuilt after a tree restore (D12)'
            Assert ($failed -eq 0) 'soak: no failures in the server log'
            Assert ($evref -eq 0) 'soak: the budget always freed enough ram'
            Assert ($diskmax -le 512*1024*1024) 'soak: the disk tier stayed within the limit'
            Assert ($ioEnd.HandleCount -le $h0 + 100) 'soak: handle count stable'
            Assert ($ioEnd.WorkingSetSize -le $rss0 + 400MB) 'soak: RSS growth bounded'

            # crash-restart: stale files must be cleared and the server must work again
            $files_before = (Get-ChildItem $dir -Recurse -File -ErrorAction SilentlyContinue).Count
            $p2 = Start-Srv $true 64 $dir $true 4096 $np $save 512 $false 16384 'soak-restart'
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
```

- [ ] **Step 2: 2 min smoke**

```powershell
$env:T32_RAM_MIB=''
& powershell -ExecutionPolicy Bypass -File 'D:\LLM\Backend\v100-collab\artifacts\t32-stage3-ab.ps1' -Mode soak -Minutes 2 | Tee-Object 'D:\LLM\Backend\v100-collab\artifacts\t32-stage4-soak-smoke.txt'
```

Expected: `RESULT soak: 0 failure(s)`; 输出含 `SOAK METRICS` 与 `SOAK RESTART` 行.

- [ ] **Step 3: 修正脚本问题并复跑 5 min**

把 smoke 暴露的问题 (参数拼写/断言口径/日志路径) 修好, 复跑 `-Minutes 5`, 覆盖 `t32-stage4-soak-smoke.txt`.

- [ ] **Step 4: 归档日志**

```powershell
New-Item -ItemType Directory -Force 'D:\LLM\Backend\v100-collab\artifacts\t32-stage4-logs' | Out-Null
Copy-Item '<TEMP>\v100\t32-stage3\srv-soak-*.txt' 'D:\LLM\Backend\v100-collab\artifacts\t32-stage4-logs\'
```

(无 git 提交: 脚本与证据都在频道 artifacts.)

---

