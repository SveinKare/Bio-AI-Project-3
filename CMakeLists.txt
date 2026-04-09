cmake_minimum_required(VERSION 3.14)
project(Bio-AI-Project-3 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to Release for optimised builds
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Add your executable
add_executable(main main.cpp landscapes.cpp)

find_package(HDF5 REQUIRED COMPONENTS C CXX)
target_include_directories(main PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(main PRIVATE ${HDF5_LIBRARIES})

