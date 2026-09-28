@echo off
call "E:\Program files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d %TEMP%\v100
nvcc -O3 -arch=sm_70 -o gemm_sweep.exe gemm_sweep.cu -lcublas 2>&1
