@echo off
rem Single-GPU full finetuning of MOSS-TTS-Local v1.5 via DeepSpeed ZeRO-3 + CPU offload (Windows).
rem Run from cmd with your Python env already activated (torch + accelerate + deepspeed installed;
rem build deepspeed with build_deepspeed_windows.bat first). Set CUDA_HOME if CUDA_PATH isn't it.
rem
rem Usage: run_train.bat <train.jsonl> <output_dir> [grad_accum=8] [num_epochs=12]
setlocal
set "HERE=%~dp0"
for %%I in ("%HERE%..\..") do set "REPO_ROOT=%%~fI"
set "FT=%REPO_ROOT%\moss_tts_local_v1.5\finetuning"
set "CFG=%HERE%configs\accelerate_zero3_offload_1gpu.yaml"

set "TRAIN=%~1"
if "%TRAIN%"=="" set "TRAIN=data\train.jsonl"
set "OUT=%~2"
if "%OUT%"=="" set "OUT=output\sft"
set "GAS=%~3"
if "%GAS%"=="" set "GAS=8"
set "EPOCHS=%~4"
if "%EPOCHS%"=="" set "EPOCHS=12"

rem --- initialize the MSVC toolchain (auto-detect VS 2019/2022/2026 via vswhere) ---
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSPATH=%%i"
if defined VSPATH call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul

rem --- CUDA + DeepSpeed runtime env, plus the Windows shims (see pyfix\sitecustomize.py) ---
if not defined CUDA_HOME set "CUDA_HOME=%CUDA_PATH%"
set "DS_BUILD_OPS=0"
set "DS_SKIP_CUDA_CHECK=1"
set "USE_LIBUV=0"
set "PYTHONPATH=%HERE%pyfix;%PYTHONPATH%"

python "%HERE%make_ds_config.py" %GAS% "%CFG%"

echo [train] data=%TRAIN% output=%OUT% grad_accum=%GAS% epochs=%EPOCHS%
accelerate launch --config_file "%CFG%" "%FT%\sft.py" ^
  --model-path OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 ^
  --codec-path OpenMOSS-Team/MOSS-Audio-Tokenizer-v2 ^
  --codec-weight-dtype fp32 ^
  --codec-compute-dtype bf16 ^
  --train-jsonl "%TRAIN%" ^
  --output-dir "%OUT%" ^
  --per-device-batch-size 1 ^
  --gradient-accumulation-steps %GAS% ^
  --learning-rate 2.0e-5 ^
  --warmup-ratio 0.05 ^
  --lr-scheduler-type cosine ^
  --num-epochs %EPOCHS% ^
  --save-every-epochs 2 ^
  --mixed-precision bf16 ^
  --channelwise-loss-weight 1,32 ^
  --gradient-checkpointing
echo [train] exit code = %errorlevel%

python "%HERE%fix_checkpoint_keys.py" "%OUT%"
endlocal
