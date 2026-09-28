@echo off
call "E:\Program files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d D:\LLM\Backend\src\llama.cpp-my
cmake -S . -B build -DGGML_CUDA_FORCE_MMQ=OFF 2>&1 | findstr /C:"Configuring done"
cmake --build build --config Release -j %NUMBER_OF_PROCESSORS% --target ggml-cuda 2>&1 | findstr /C:"error" /C:"Linking"
