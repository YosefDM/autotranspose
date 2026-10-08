@echo off
rem Fetch the pitch-shifting dependencies at pinned revisions.
rem
rem They are fetched rather than vendored, which is what upstream's own
rem CMakeLists does (FetchContent), so this repository carries no copy of
rem someone else's source. Both are MIT licensed.
rem
rem   signalsmith-stretch  https://github.com/Signalsmith-Audio/signalsmith-stretch
rem   signalsmith-linear   https://github.com/Signalsmith-Audio/linear
setlocal
set ROOT=%~dp0..
set TP=%ROOT%\third_party

set STRETCH_REV=a670068d9aeb64913331d5cc29337b19a457a7df
set LINEAR_TAG=0.6.4

where git >nul 2>&1
if errorlevel 1 (
    echo git is not on PATH; install Git for Windows.
    exit /b 1
)

if not exist "%TP%" mkdir "%TP%"

if not exist "%TP%\signalsmith-stretch\signalsmith-stretch.h" (
    echo Fetching signalsmith-stretch...
    rmdir /s /q "%TP%\signalsmith-stretch" 2>nul
    git clone --quiet https://github.com/Signalsmith-Audio/signalsmith-stretch.git "%TP%\signalsmith-stretch"
    if errorlevel 1 (echo clone failed & exit /b 1)
    git -C "%TP%\signalsmith-stretch" checkout --quiet %STRETCH_REV%
    if errorlevel 1 (echo checkout failed & exit /b 1)
)

if not exist "%TP%\linear\include\signalsmith-linear\stft.h" (
    echo Fetching signalsmith-linear %LINEAR_TAG%...
    rmdir /s /q "%TP%\linear" 2>nul
    git clone --quiet --depth 1 --branch %LINEAR_TAG% https://github.com/Signalsmith-Audio/linear.git "%TP%\linear"
    if errorlevel 1 (echo clone failed & exit /b 1)
)

echo Dependencies present in third_party\.
