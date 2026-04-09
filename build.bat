@echo off
echo Creating build directory...
if not exist build mkdir build

echo Building project...
cmake -B build -DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake
cmake --build build

@echo off
if errorlevel 1 (
    echo Build failed!
    pause
    exit /b 1
)

echo Running executable...
REM Go back to project root before running
build\Debug\main.exe
pause

