@echo off
rem Build + install DeepSpeed (cpu_adam only) on Windows for single-GPU ZeRO-3 CPU offload.
rem DeepSpeed has no official Windows wheels, so we build the one op offload needs from source.
rem
rem Prerequisites (activate your Python env first -- the SAME torch you train with):
rem   - Visual Studio 2019+ (or Build Tools) with the "Desktop development with C++" workload
rem   - An NVIDIA CUDA Toolkit (nvcc). Set CUDA_HOME if CUDA_PATH isn't the right one.
setlocal
set "HERE=%~dp0"

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSPATH=%%i"
if not defined VSPATH ( echo [error] Visual Studio with the C++ tools was not found. & exit /b 1 )
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul

if not defined CUDA_HOME set "CUDA_HOME=%CUDA_PATH%"
set "CUDA_PATH=%CUDA_HOME%"
set "DISTUTILS_USE_SDK=1"
rem Build only cpu_adam (the op ZeRO-offload needs); skip everything else.
set "DS_BUILD_OPS=0"
set "DS_BUILD_CPU_ADAM=1"
rem A minor toolkit-vs-torch CUDA mismatch (e.g. 13.2 vs cu130) is fine here.
set "DS_SKIP_CUDA_CHECK=1"
set "DS_BUILD_AIO=0"
set "DS_BUILD_CUTLASS_OPS=0"
set "DS_BUILD_EVOFORMER_ATTN=0"
set "DS_BUILD_FP_QUANTIZER=0"
set "DS_BUILD_GDS=0"
set "DS_BUILD_RAGGED_DEVICE_OPS=0"
set "DS_BUILD_SPARSE_ATTN=0"
set "DS_BUILD_FUSED_ADAM=0"
set "DS_BUILD_FUSED_LAMB=0"
set "DS_BUILD_TRANSFORMER=0"
set "DS_BUILD_TRANSFORMER_INFERENCE=0"
set "DS_BUILD_STOCHASTIC_TRANSFORMER=0"
set "DS_BUILD_QUANTIZER=0"
set "DS_BUILD_INFERENCE_CORE_OPS=0"

set "DS_VERSION=0.19.2"
set "BUILDDIR=%HERE%_ds_build"
if not exist "%BUILDDIR%" mkdir "%BUILDDIR%"
pushd "%BUILDDIR%"
rem Build from the GitHub source tag -- the PyPI sdist omits the Windows launcher scripts.
if not exist "DeepSpeed\setup.py" git clone --depth 1 --branch v%DS_VERSION% https://github.com/deepspeedai/DeepSpeed.git DeepSpeed
cd DeepSpeed
if not exist "bin\deepspeed.bat" ( > "bin\deepspeed.bat" echo @echo off& >> "bin\deepspeed.bat" echo python -m deepspeed.launcher.runner %%* )
if not exist "bin\ds_report.bat" ( > "bin\ds_report.bat" echo @echo off& >> "bin\ds_report.bat" echo python -m deepspeed.env_report %%* )

python setup.py bdist_wheel || ( echo [error] DeepSpeed build failed & popd & exit /b 1 )
for %%f in (dist\deepspeed-*.whl) do python -m pip install --force-reinstall --no-deps "%%f"
python -m pip install hjson py-cpuinfo nvidia-ml-py ninja
popd

echo ============ verify ============
python -c "from deepspeed.ops.op_builder import CPUAdamBuilder; CPUAdamBuilder().load(); print('cpu_adam OK')"
endlocal
