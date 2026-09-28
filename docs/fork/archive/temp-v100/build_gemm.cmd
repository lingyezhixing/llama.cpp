@echo off
call "E:\Program files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
nvcc -arch=sm_70 -O3 -o "<TEMP>\v100\gemm_bench.exe" "<TEMP>\v100\gemm_bench.cu" -lcublas 2>&1
