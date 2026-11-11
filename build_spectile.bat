@echo off
setlocal
echo ======================================================================
echo  SpecTile / FENICE C++20 Build ^& Test Launcher (Windows x64)
echo ======================================================================

set "CMAKE_EXE="
for /f "delims=" %%I in ('where cmake 2^>nul') do (
    if not defined CMAKE_EXE set "CMAKE_EXE=%%I"
)
if not defined CMAKE_EXE if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" (
    set "CMAKE_EXE=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
)
if not defined CMAKE_EXE if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" (
    set "CMAKE_EXE=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
)
if not defined CMAKE_EXE if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" (
    set "CMAKE_EXE=C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
)
if not defined CMAKE_EXE if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" (
    set "CMAKE_EXE=C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
)
if not defined CMAKE_EXE if exist "C:\Program Files\CMake\bin\cmake.exe" (
    set "CMAKE_EXE=C:\Program Files\CMake\bin\cmake.exe"
)

if not defined CMAKE_EXE (
    echo [ERROR] CMake non trovato. Installa CMake o Visual Studio 2022 con C++ Desktop Development.
    exit /b 1
)

set "FETCH_FLAG=OFF"
if not exist "external\JUCE\CMakeLists.txt" (
    where git >nul 2>nul
    if not errorlevel 1 (
        echo [0/3] Inizializzazione submodule external/JUCE...
        git submodule update --init --recursive --depth 1
    )
    if not exist "external\JUCE\CMakeLists.txt" (
        echo [0/3] Submodule non presente: verra' scaricato JUCE 8.0.4 via CMake FetchContent.
        set "FETCH_FLAG=ON"
    )
)

echo [1/3] Configurazione CMake (Visual Studio 17 2022 -A x64 + JUCE 8 VST3/Standalone)...
"%CMAKE_EXE%" -B build -G "Visual Studio 17 2022" -A x64 -DSPECTILE_BUILD_PLUGIN=ON -DSPECTILE_FETCH_EXTERNAL_DEPS=%FETCH_FLAG% -DCMAKE_POLICY_DEFAULT_CMP0141=NEW -DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded
if errorlevel 1 exit /b %errorlevel%

echo [2/3] Compilazione C++20 in modalita Release (VST3 + Standalone + CLI + Tests)...
"%CMAKE_EXE%" --build build --config Release
if errorlevel 1 exit /b %errorlevel%

echo [3/3] Esecuzione Test Suite C++20 (spectile_unit_tests.exe)...
.\build\Release\spectile_unit_tests.exe
if errorlevel 1 exit /b %errorlevel%

echo.
echo [OK] Build e Test completati! Binari pronti in:
echo   - VST3 Plugin    : build\SpecTile_artefacts\Release\VST3\SpecTile.vst3
echo   - Standalone App : build\SpecTile_artefacts\Release\Standalone\SpecTile.exe
echo   - CLI Synth      : build\Release\spectile_cli.exe
echo   - Unit Tests     : build\Release\spectile_unit_tests.exe
endlocal
