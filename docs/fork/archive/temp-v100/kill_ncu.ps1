Get-Process | Where-Object { $_.ProcessName -match "^(ncu|llama-bench)$" } | ForEach-Object { Write-Host ("killing {0} pid {1}" -f $_.ProcessName, $_.Id); Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 2
Get-Process | Where-Object { $_.ProcessName -match "ncu|llama" } | Select-Object Id, ProcessName | Format-Table -AutoSize
