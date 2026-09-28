function ReqDelim($prompt, $cache) {
    $body = @{prompt=$prompt; n_predict=32; temperature=0; top_k=1; seed=42; cache_prompt=$cache; stream=$false; message_delimiters=@(@{user='User:'})}
    return Invoke-RestMethod -Uri "http://127.0.0.1:$Port/completion" -Method Post -ContentType 'application/json' -TimeoutSec 120 -Body ($body | ConvertTo-Json -Compress -Depth 4)
}

