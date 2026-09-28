@echo off
call "E:\Program files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d %TEMP%\v100
nvcc -O3 -arch=sm_70 -Wno-deprecated-gpu-targets -o t24_fold.exe t24_fold_harness.cu 2>&1
