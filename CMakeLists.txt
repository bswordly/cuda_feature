
cmake_minimum_required(VERSION 3.20)
project(jetson_sobel LANGUAGES CXX)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
execute_process(
	COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
	OUTPUT_STRIP_TRAILING_WHITESPACE
	OUTPUT_VARIABLE pybind11_DIR
)
find_package(pybind11 CONFIG REQUIRED)
include (CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
   enable_language(CUDA)
   message(STATUS "CUDA found: Building GPU backend (targeting Orin sm_87
	 + PTX fallback)")
	 set(CMAKE_CUDA_STANDARD 17)
	 # Compile native SASS for Jetson Orin (87) plus virtual PIX for forward
	 # compatibility
	 set(CMAKE_CUDA_ARCHITECTURES "87-real;87-virtual")
   pybind11_add_module(sobel_ops src/bindings.cpp src/sobel_cuda.cu)
	 target_compile_definitions(sobel_ops PRIVATE USE_CUDA)
else()
	 message(STATUS "CUDA not found: Building portable CPU fallback backend")
	 pybind11_add_module(sobel_ops src/bindings.cpp src/sobel_cpu.cpp)
endif()

target_include_directories(sobel_ops PRIVATE src)



