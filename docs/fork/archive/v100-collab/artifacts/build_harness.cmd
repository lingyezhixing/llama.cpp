@echo off
call "E:\Program files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
nvcc -O3 -arch=sm_70 -std=c++17 -Wno-deprecated-gpu-targets -I"D:\LLM\Backend\src\llama.cpp-my\ggml\src\ggml-cuda" -I"D:\LLM\Backend\src\llama.cpp-my\ggml\include" -I"D:\LLM\Backend\src\llama.cpp-my\ggml\src" -o "%1" "%2" -lcublas %3 %4 %5
