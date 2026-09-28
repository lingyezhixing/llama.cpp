@echo off
call "E:\Program files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
nvcc -arch=sm_70 -O3 -Wno-deprecated-gpu-targets -o "<TEMP>\v100\dequant_bench.exe" "<TEMP>\v100\dequant_bench.cu" 2>&1
