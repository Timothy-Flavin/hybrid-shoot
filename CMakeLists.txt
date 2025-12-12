cmake_minimum_required(VERSION 3.15)
project(hybrid_shoot LANGUAGES CXX)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_hybrid_shoot cpp/env.cpp)
install(TARGETS _hybrid_shoot DESTINATION hybrid_shoot)