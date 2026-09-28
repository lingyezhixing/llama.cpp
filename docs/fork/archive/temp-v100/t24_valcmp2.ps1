function ParseRes($log) { $o=@(); foreach ($l in (Select-String -Path $log -Pattern 'T24RES' | ForEach-Object { $_.Line })) { if ($l -match 'nt=(\d+) p=(\d+) half=(\d+) base=(\S+) res=(\S+)') { $o += [pscustomobject]@{nt=[int]$Matches[1];p=[int]$Matches[2];base=$Matches[4];res=$Matches[5]} } }; $o }
function ParseSnap($log) { $o=@(); foreach ($l in (Select-String -Path $log -Pattern 'T24SNAP' | ForEach-Object { $_.Line })) { if ($l -match 'nt=(\d+) slot=(\d+) v=(\S+)') { $o += [pscustomobject]@{nt=[int]$Matches[1];slot=[int]$Matches[2];v=$Matches[3]} } }; $o }
$R = ParseRes "$env:TEMP\t24_srv_1.log"; $S = ParseSnap "$env:TEMP\t24_srv_0.log"
# layer 0 = first launch of each 48-launch group; keep only nt=4 lines
$A = @(); for ($i=0; $i -lt $R.Count; $i+=48) { if ($R[$i].nt -eq 4) { $A += $R[$i] } }
$Bs = @(); for ($i=0; $i -lt $S.Count; $i+=192) { $h=@{}; foreach ($e in $S[$i..($i+3)]) { $h[[string]$e.slot]=$e.v }; $Bs += ,$h }
"layer0 nt=4 batches A=$($A.Count)  B=$($Bs.Count)"
$n=[Math]::Min($A.Count,$Bs.Count); $bad=0
for ($k=0; $k -lt $n; $k++) {
  $p=$A[$k].p; $exp=$null
  if ($k-1 -ge 0) { $exp=$Bs[$k-1][[string](4-$p)] }
  if (-not ($exp -ceq $A[$k].res)) { "MISMATCH k=$k p=$p base=$($A[$k].base) res=$($A[$k].res) exp=$exp"; $bad++; if ($bad -ge 10) { break } }
}
if ($bad -eq 0) { "ALL RES MATCH ($n batches)" }
"--- first 12 batches: (p, base, res) vs B[k-1] slots ---"
for ($k=0; $k -lt [Math]::Min(12,$n); $k++) {
  $p=$A[$k].p
  $s0=if($k-1 -ge 0){$Bs[$k-1]['0']}else{'-'}
  $s1=if($k-1 -ge 0){$Bs[$k-1]['1']}else{'-'}
  $s2=if($k-1 -ge 0){$Bs[$k-1]['2']}else{'-'}
  $s3=if($k-1 -ge 0){$Bs[$k-1]['3']}else{'-'}
  "k=$k p=$p res=$($A[$k].res)"
  "      prev slots: 3=$s3 2=$s2 1=$s1 0=$s0"
}
