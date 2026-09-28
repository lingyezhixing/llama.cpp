$ErrorActionPreference='Continue'
cmd /c "$env:TEMP\v100\build_ggml_cuda.cmd" 2>&1 | Select-String -Pattern ': error' | Select-Object -First 5
Copy-Item 'D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\ggml-cuda.dll' 'D:\LLM\Backend\llama.cpp-t24\' -Force
& powershell -NoProfile -ExecutionPolicy Bypass -File "$env:TEMP\v100\t24_ab.ps1" -BinDir 'D:\LLM\Backend\llama.cpp-t24' -Replay 1 -OutFile "$env:TEMP\v100\t24_v_on.txt" | Select-String 'replay='
& powershell -NoProfile -ExecutionPolicy Bypass -File "$env:TEMP\v100\t24_ab.ps1" -BinDir 'D:\LLM\Backend\llama.cpp-t24' -Replay 0 -OutFile "$env:TEMP\v100\t24_v_off.txt" | Select-String 'replay='
function ParseRes($log) { $o=@(); foreach ($l in (Select-String -Path $log -Pattern 'T24RES' | ForEach-Object { $_.Line })) { if ($l -match 'ptr=([0-9a-f]+) nt=(\d+) p=(\d+) half=(\d+) base=(\S+) res=(\S+)') { $o += [pscustomobject]@{ptr=$Matches[1];nt=[int]$Matches[2];p=[int]$Matches[3];base=$Matches[5];res=$Matches[6]} } }; $o }
function ParseSnap($log) { $o=@(); foreach ($l in (Select-String -Path $log -Pattern 'T24SNAP' | ForEach-Object { $_.Line })) { if ($l -match 'ptr=([0-9a-f]+) nt=(\d+) slot=(\d+) v=(\S+)') { $o += [pscustomobject]@{ptr=$Matches[1];nt=[int]$Matches[2];slot=[int]$Matches[3];v=$Matches[4]} } }; $o }
$R = ParseRes "$env:TEMP\t24_srv_1.log"; $S = ParseSnap "$env:TEMP\t24_srv_0.log"
"RES lines: $($R.Count)  SNAP lines: $($S.Count)"
$tgtR = ($R | Where-Object {$_.nt -eq 4} | Group-Object ptr | Sort-Object Count -Descending | Select-Object -First 1).Name
$tgtS = ($S | Where-Object {$_.nt -eq 4} | Group-Object ptr | Sort-Object Count -Descending | Select-Object -First 1).Name
"targetR=$tgtR targetS=$tgtS"
$A = @($R | Where-Object {$_.nt -eq 4 -and $_.ptr -eq $tgtR})
$Bs = @($S | Where-Object {$_.nt -eq 4 -and $_.ptr -eq $tgtS})
$B = @(); for ($i=0; $i+3 -lt $Bs.Count; $i+=4) { $h=@{}; foreach ($e in $Bs[$i..($i+3)]) { $h[[string]$e.slot]=$e.v }; $B += ,$h }
"A batches: $($A.Count)  B batches: $($B.Count)"
$n=[Math]::Min($A.Count,$B.Count); $bad=0
for ($k=0; $k -lt $n; $k++) {
  $p=$A[$k].p; $exp=$null
  if ($k-1 -ge 0 -and (4-$p) -ge 0) { $exp=$B[$k-1][[string](4-$p)] }
  if (-not ($exp -ceq $A[$k].res)) {
    "MISMATCH k=$k p=$p base=$($A[$k].base) res=$($A[$k].res) exp(B[$($k-1)] slot $(4-$p))=$exp"; $bad++
    if ($bad -ge 8) { break }
  }
}
if ($bad -eq 0) { "ALL MATCH ($n batches)" }
