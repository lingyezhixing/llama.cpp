@echo off
set CUDA_VISIBLE_DEVICES=1
if "%T24_REPLAY%"=="1" set GGML_CUDA_GDN_REPLAY=1
"%~1\llama-server.exe" -m "<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf" -np 1 -ngl 99 -fa on -c 16384 -ctv q8_0 --spec-type draft-mtp --spec-draft-n-max 1 --port 8479 --host 127.0.0.1 > "%~2" 2>&1
