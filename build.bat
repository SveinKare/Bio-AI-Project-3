@echo off
echo Creating build directory...
if not exist build mkdir build

echo Configuring CMake...
cd build
cmake ..
if errorlevel 1 (
    echo CMake configuration failed!
    cd ..
    pause
    exit /b 1
)

echo Building project...
cmake --build .
if errorlevel 1 (
    echo Build failed!
    cd ..
    pause
    exit /b 1
)

echo Running executable...
REM Go back to project root before running
cd ..
build\Debug\main.exe
pause
