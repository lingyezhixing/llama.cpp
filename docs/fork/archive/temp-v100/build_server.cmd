@echo off
call "E:\Program files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d D:\LLM\Backend\src\llama.cpp-my
cmake --build build --config Release -j %NUMBER_OF_PROCESSORS% --target llama-server 2>&1
