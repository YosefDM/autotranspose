@echo off
rem Build the Signalsmith Stretch wrapper DLL. Needs MSVC Build Tools.
rem The project works without this DLL: Python falls back to the numpy vocoder.
setlocal
set ROOT=%~dp0..
set VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat
if not exist "%VCVARS%" set VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat
if not exist "%VCVARS%" (
    echo Could not find vcvars64.bat. Install "Desktop development with C++".
    exit /b 1
)
call "%VCVARS%" >nul 2>&1

rem Dependencies are fetched, not vendored.
if not exist "%ROOT%	hird_party\signalsmith-stretch\signalsmith-stretch.h" call "%~dp0fetch_deps.bat"
if not exist "%ROOT%	hird_party\linear\include\signalsmith-linear\stft.h" call "%~dp0fetch_deps.bat"

cl /nologo /LD /MT /O2 /EHsc /std:c++17 /DNDEBUG ^
   /I "%ROOT%\third_party\signalsmith-stretch" ^
   /I "%ROOT%\third_party\linear\include" ^
   "%~dp0stretch_wrapper.cpp" ^
   /Fo:"%~dp0stretch_wrapper.obj" ^
   /Fe:"%ROOT%\autotranspose\signalsmith_stretch.dll" ^
   /link /IMPLIB:"%~dp0stretch_wrapper.lib"
if errorlevel 1 (echo BUILD FAILED & exit /b 1)
echo Built autotranspose\signalsmith_stretch.dll
