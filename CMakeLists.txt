cmake_minimum_required(VERSION 3.15)
project(hybrid_shoot LANGUAGES CXX)

find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP)

pybind11_add_module(_hybrid_shoot cpp/env.cpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(_hybrid_shoot PRIVATE OpenMP::OpenMP_CXX)
endif()

# Enable optimizations for the release wheel build.
if(MSVC)
    target_compile_options(_hybrid_shoot PRIVATE /O2)
else()
    target_compile_options(_hybrid_shoot PRIVATE -O3)
endif()

install(TARGETS _hybrid_shoot DESTINATION hybrid_shoot)
