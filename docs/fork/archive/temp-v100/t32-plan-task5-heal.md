    'heal' {
        $dir = "$OutDir\tree-heal"; Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
        $p = Start-Srv $true 512 $dir
        $h = "User: " + (Filler 'h1' 480) + "`nAssistant: ok`nUser: " + (Filler 'h2' 600) + "`nAssistant: ok`nUser: "
        $tA = Filler 'tail-a' 1000; $tF = Filler 'tail-f' 1000
        $reqs = @("$h$tA", "$h$tF", ("Zeta: " + (Filler 'omega' 1800)), "$h$tF")
        foreach ($pass in @(@($false,'full'), @($true,'tree'))) {
            $i = 0
            foreach ($q in $reqs) {
                $i++
                $r = ReqDelim $q $pass[0]
                $hsh = ContentHash $r.content
                $hashes["$($pass[1])/heal/$i"] = $hsh
                Write-Output ("[heal/$($pass[1])] req $i: hash=$hsh prompt_n=$($r.timings.prompt_n)")
            }
        }
        $p | Stop-Process -Force
        $log = Get-Content "$OutDir\srv-$Mode-err.txt" -Encoding UTF8
        Assert ((($log | Select-String 'captured heal anchor').Count) -eq 1) 'heal: exactly one fork anchor captured'
        Assert ((($log | Select-String 'failed to capture heal anchor').Count) -eq 0) 'heal: no failed captures'
        for ($i = 1; $i -le 4; $i++) { Assert ($hashes["tree/heal/$i"] -eq $hashes["full/heal/$i"]) "heal: request $i identical (tree vs full prefill)" }
    }
}

