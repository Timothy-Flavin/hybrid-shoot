cmake_minimum_required(Version 3.15)
project(hybrid_shoot LANGUAGES CXX)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(cpp_env cpp/env.cpp)