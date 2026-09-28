$ErrorActionPreference='Continue'
cmd /c "$env:TEMP\v100\build_ggml_cuda.cmd" 2>&1 | Select-String -Pattern ': error' | Select-Object -First 5
Copy-Item 'D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\ggml-cuda.dll' 'D:\LLM\Backend\llama.cpp-t24\' -Force
& powershell -NoProfile -ExecutionPolicy Bypass -File "$env:TEMP\v100\t24_ab.ps1" -BinDir 'D:\LLM\Backend\llama.cpp-t24' -Replay 1 -OutFile "$env:TEMP\v100\t24_l_on.txt" | Select-String 'replay='
& powershell -NoProfile -ExecutionPolicy Bypass -File "$env:TEMP\v100\t24_ab.ps1" -BinDir 'D:\LLM\Backend\llama.cpp-t24' -Replay 0 -OutFile "$env:TEMP\v100\t24_l_off.txt" | Select-String 'replay='
$R=@(); foreach ($l in (Select-String -Path "$env:TEMP\t24_srv_1.log" -Pattern 'T24RES' | ForEach-Object {$_.Line})) { if ($l -match 'lid=(\d+) nt=(\d+) p=(\d+) half=(\d+) base=(\S+) res=(\S+)') { $R += [pscustomobject]@{lid=[int64]$Matches[1];nt=[int]$Matches[2];p=[int]$Matches[3];base=$Matches[5];res=$Matches[6]} } }
$S=@(); foreach ($l in (Select-String -Path "$env:TEMP\t24_srv_0.log" -Pattern 'T24SNAP' | ForEach-Object {$_.Line})) { if ($l -match 'lid=(\d+) nt=(\d+) slot=(\d+) v=(\S+)') { $S += [pscustomobject]@{lid=[int64]$Matches[1];nt=[int]$Matches[2];slot=[int]$Matches[3];v=$Matches[4]} } }
"RES=$($R.Count) SNAP=$($S.Count)"
function Groups($arr) { $g=@(); $cur=@(); foreach ($e in ($arr | Sort-Object lid)) { if ($cur.Count -gt 0 -and $e.lid -ne ($cur[-1].lid+1)) { $g+=,$cur; $cur=@() }; $cur+=$e }; if ($cur.Count) { $g+=,$cur }; $g }
$RG = Groups $R; $SG = Groups $S
"ON groups=$($RG.Count) OFF groups=$($SG.Count) sizes on=$($RG[0].Count),$($RG[1].Count) off=$($SG[0].Count)"
$bad=0; $n=[Math]::Min($RG.Count,$SG.Count)
for ($i=1; $i -lt $n; $i++) {
  $on=$RG[$i][0]; $op=$SG[$i-1][0]
  $ov=@{}; foreach ($e in $SG[$i-1]) { if ($e.lid -eq $op.lid) { $ov[[string]$e.slot]=$e.v } }
  $slot=$op.nt-$on.p; $exp=$ov[[string]$slot]
  if (-not ($exp -ceq $on.res)) { "MISMATCH i=$i on(lid=$($on.lid) nt=$($on.nt) p=$($on.p)) res=$($on.res) exp(lid=$($op.lid) nt=$($op.nt) slot=$slot)=$exp"; $bad++; if ($bad -ge 8) { break } }
}
if ($bad -eq 0) { "ALL RES MATCH ($n groups)" }
