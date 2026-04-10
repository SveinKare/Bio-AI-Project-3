cmake_minimum_required(VERSION 3.14)

# Auto-detect vcpkg on Windows
if(WIN32 AND DEFINED ENV{VCPKG_ROOT})
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "")
endif()


project(Bio-AI-Project-3 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

add_executable(main main.cpp)

# Link HDF5 depending on platform
if(WIN32)
    target_compile_options(main PRIVATE /EHsc)
    find_package(hdf5 CONFIG REQUIRED)
    target_link_libraries(main PRIVATE hdf5::hdf5-shared hdf5::hdf5_cpp-shared)
else()
    find_package(HDF5 REQUIRED COMPONENTS C CXX)
    target_include_directories(main PRIVATE ${HDF5_INCLUDE_DIRS})
    target_link_libraries(main PRIVATE ${HDF5_LIBRARIES})
endif()
