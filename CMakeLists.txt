cmake_minimum_required(VERSION 3.20)

project(cuebee_speaker_serving LANGUAGES CXX)

option(CUEBEE_ENABLE_ONNX "Build the worker with ONNX Runtime support" OFF)
set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to an extracted ONNX Runtime release")

enable_testing()
add_subdirectory(cpp)
