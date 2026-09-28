@echo off
call "E:\Program files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d D:\LLM\Backend\src\llama.cpp-my
cmake -S . -B build -DLLAMA_BUILD_EXAMPLES=ON 2>&1
if errorlevel 1 exit /b 1
cmake --build build --config Release -j %NUMBER_OF_PROCESSORS% --target llama-batch-invariance 2>&1
