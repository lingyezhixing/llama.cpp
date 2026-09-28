param(
    [string]$Tag = 'run'
)

$ErrorActionPreference = 'Stop'
# same layout as t32-mmproj-check.ps1: model on the V100 (cuda1), mmproj on the RTX 4060 (cuda0)
$env:CUDA_VISIBLE_DEVICES = '0,1'
$env:GGML_CUDA_GDN_REPLAY = '1'
$repo = 'D:\LLM\Backend\src\llama.cpp-my'
$srv  = "$repo\build\bin\Release\llama-server.exe"
$Model  = '<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf'
$Mmproj = '<models>\Qwen3.8-27B\mmproj-F16.gguf'
$Port = 8948
$out = "<TEMP>\v100\t32-mmproj-probe-$Tag"
Remove-Item -Recurse -Force $out -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $out | Out-Null
$tree = "$out\tree"
New-Item -ItemType Directory -Force $tree | Out-Null

# same args as the accepted text-only check, plus --jinja so the media probe can use /v1/chat/completions
$sargs = @('-m',$Model,'-dev','cuda1','--mmproj',$Mmproj,'-mmdev','cuda0','-np','1','-ngl','99','-fa','on','-ctv','q8_0','-c','32768',
          '--cache-ram','0','--no-context-shift','--host','127.0.0.1','--port',"$Port",'--jinja',
          '--spec-type','draft-mtp','--spec-draft-n-max','3',
          '--kv-tree','--tree-ram','2048','--tree-disk',$tree,'--tree-disk-limit','4096',
          '--tree-checkpoint-anchor-step','32768','--tree-checkpoint-fork-step','8192')

function Send-Json($uri, $bodyObj) {
    $json = $bodyObj | ConvertTo-Json -Depth 12 -Compress
    try {
        $r = Invoke-WebRequest -Uri $uri -Method Post -ContentType 'application/json' -TimeoutSec 600 `
                -Body ([System.Text.Encoding]::UTF8.GetBytes($json)) -UseBasicParsing
        return [pscustomobject]@{ status = [int]$r.StatusCode; body = $r.Content }
    } catch {
        $resp = $_.Exception.Response
        if ($resp) {
            $sr = New-Object System.IO.StreamReader($resp.GetResponseStream())
            return [pscustomobject]@{ status = [int]$resp.StatusCode; body = $sr.ReadToEnd() }
        }
        return [pscustomobject]@{ status = 0; body = "CONNECTION FAILED: $($_.Exception.Message)" }
    }
}

function Server-Alive($p) {
    $p.Refresh()
    if ($p.HasExited) { return $false }
    try { return (Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 5).status -eq 'ok' } catch { return $false }
}

$p = $null
$probe = @()
$rows  = @()
try {
    $p = Start-Process -FilePath $srv -ArgumentList $sargs -NoNewWindow -PassThru -RedirectStandardOutput "$out\srv-out.txt" -RedirectStandardError "$out\srv-err.txt"
    $up = $false
    for ($i = 0; $i -lt 300; $i++) {
        Start-Sleep -Milliseconds 1000
        if ($p.HasExited) { throw "server exited early" }
        try { Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 3 | Out-Null; $up = $true; break } catch {}
    }
    if (-not $up) { throw 'server did not become healthy' }

    function TokCount($text) {
        return (Invoke-RestMethod -Uri "http://127.0.0.1:$Port/tokenize" -Method Post -ContentType 'application/json' -TimeoutSec 60 -Body (@{content=$text} | ConvertTo-Json -Compress)).tokens.Count
    }
    function Filler($salt, $n_tok) {
        $unit = "$salt filler sentence for the kv tree acceptance test number "
        $ut = TokCount $unit
        $t = ''; $cur = 0
        while ($cur -lt $n_tok) { $need = [int][math]::Ceiling(($n_tok - $cur) / $ut); $t += ($unit * $need); $cur = TokCount $t }
        return $t
    }
    function Ask($prompt, $tag) {
        $body = @{prompt=$prompt; n_predict=16; temperature=0; seed=42; cache_prompt=$true; stream=$false; message_delimiters=@(@{role='user'; delimiter='User:'})}
        $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/completion" -Method Post -ContentType 'application/json' -TimeoutSec 900 -Body ($body | ConvertTo-Json -Compress -Depth 4)
        return [pscustomobject]@{ tag=$tag; prompt_n=$r.timings.prompt_n; predicted_n=$r.timings.predicted_n }
    }

    # phase 1: text-only A/B/A/B, identical to the accepted check
    $sys = Filler 'system' 512
    $pA = "$sys`nUser: " + (Filler 'alpha' 5300) + "`nUser turn 1: continue`nAssistant:"
    $pB = "$sys`nUser: " + (Filler 'beta' 3300) + "`nUser turn 2: continue`nAssistant:"
    Write-Output "prompt tokens: A=$(TokCount $pA) B=$(TokCount $pB)"

    $rows += Ask $pA 'round1-A'
    $rows += Ask $pB 'round1-B'
    $rows += Ask $pA 'round2-A'
    $rows += Ask $pB 'round2-B'
    Write-Output '=== A/B/A/B prompt_n table ==='
    ($rows | Format-Table -AutoSize | Out-String).Trim() | Write-Output

    # phase 2: probes on the same server, after at least one tree restore
    $png = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=='
    $mediaBase = @{
        messages = @(
            @{ role = 'user'; content = @(
                @{ type = 'text';      text = 'What do you see in this image? Answer in one short sentence.' },
                @{ type = 'image_url'; image_url = @{ url = "data:image/png;base64,$png" } }
            ) }
        )
        max_tokens = 16; temperature = 0; seed = 42; cache_prompt = $true; stream = $false
    }

    $r1 = Send-Json "http://127.0.0.1:$Port/v1/chat/completions" $mediaBase
    $probe += [pscustomobject]@{ tag = 'media'; status = $r1.status; alive = (Server-Alive $p); body = $r1.body.Substring(0, [Math]::Min(240, $r1.body.Length)) }

    $reuseBase = $mediaBase.Clone()
    $reuseBase['n_cache_reuse'] = 64
    $r2 = Send-Json "http://127.0.0.1:$Port/v1/chat/completions" $reuseBase
    $probe += [pscustomobject]@{ tag = 'media+n_cache_reuse'; status = $r2.status; alive = (Server-Alive $p); body = $r2.body.Substring(0, [Math]::Min(240, $r2.body.Length)) }

    $r3 = Send-Json "http://127.0.0.1:$Port/completion" @{ prompt = "User: Say hello in one word.`nAssistant:"; n_predict = 4; temperature = 0; seed = 42; cache_prompt = $true; stream = $false }
    $probe += [pscustomobject]@{ tag = 'text-only-final'; status = $r3.status; alive = (Server-Alive $p); body = $r3.body.Substring(0, [Math]::Min(240, $r3.body.Length)) }

    Write-Output '=== probe results ==='
    ($probe | Format-Table -AutoSize -Wrap | Out-String).Trim() | Write-Output

    $rows | ConvertTo-Json | Set-Content "$out\prompt_n.json" -Encoding UTF8
    $probe | ConvertTo-Json | Set-Content "$out\probe.json" -Encoding UTF8
} finally {
    if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force; $p.WaitForExit(10000) | Out-Null }
}

Start-Sleep -Seconds 1
$log = Get-Content "$out\srv-err.txt" -Encoding UTF8
$treeLines = @($log | Select-String 'kv tree:')
Write-Output '=== kv tree log lines ==='
$treeLines | ForEach-Object { $_.Line }
$parked   = @($treeLines | Where-Object { $_.Line -cmatch 'parked' }).Count
$restored = @($treeLines | Where-Object { $_.Line -cmatch 'kv tree: restored' }).Count
$miss     = @($treeLines | Where-Object { $_.Line -cmatch 'restore miss' }).Count
Write-Output "tree_log: parked=$parked restored=$restored miss=$miss"

$aborts = @($log | Select-String -Pattern 'GGML_ASSERT|GGML_ABORT|Aborted|terminate called' | ForEach-Object { $_.Line })
Write-Output "abort_scan: $($aborts.Count) match(es)"
$aborts | ForEach-Object { Write-Output "  $_" }

$r1a = $rows | Where-Object { $_.tag -eq 'round1-A' }
$r2a = $rows | Where-Object { $_.tag -eq 'round2-A' }
$r1b = $rows | Where-Object { $_.tag -eq 'round1-B' }
$r2b = $rows | Where-Object { $_.tag -eq 'round2-B' }
$fast_a = $r2a.prompt_n -lt $r1a.prompt_n
$fast_b = $r2b.prompt_n -lt $r1b.prompt_n
Write-Output "round2-A faster than round1-A: $fast_a ($($r1a.prompt_n) -> $($r2a.prompt_n))"
Write-Output "round2-B faster than round1-B: $fast_b ($($r1b.prompt_n) -> $($r2b.prompt_n))"

$ab_md_gate  = ($parked -ge 2) -and ($restored -ge 1) -and ($fast_a -or $fast_b)
$media_ok    = ($probe | Where-Object { $_.tag -eq 'media' }).status -eq 200
$reuse_ok    = ($probe | Where-Object { $_.tag -eq 'media+n_cache_reuse' }).status -eq 200
$text_ok     = ($probe | Where-Object { $_.tag -eq 'text-only-final' }).status -eq 200
$alive_ok    = @($probe | Where-Object { -not $_.alive }).Count -eq 0
$ok = ($ab_md_gate -and $media_ok -and $reuse_ok -and $text_ok -and $alive_ok -and ($aborts.Count -eq 0))
Write-Output "RESULT: $(if ($ok) { 'PASS' } else { 'FAIL' })"
