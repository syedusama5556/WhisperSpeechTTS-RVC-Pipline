@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128
set COMMANDLINE_ARGS=--xformers --no-half --medvram --enable-insecure-extension-access --opt-split-attention
runtime\python.exe app.py
pause
