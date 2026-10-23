cmake_minimum_required(VERSION 3.18)
project(chirp_interf LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(CHIRP_USE_OPENMP "Parallelise the core with OpenMP" ON)

# ---- Core library: pure C++, no Python dependency ------------------------------
add_library(chirp_core STATIC src/stft.cpp src/recovery.cpp)
target_include_directories(chirp_core PUBLIC include PRIVATE src)
if(MSVC)
  target_compile_options(chirp_core PRIVATE /W4)
else()
  target_compile_options(chirp_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(CHIRP_USE_OPENMP)
  # Homebrew's libomp is not on the default search path on macOS.
  if(APPLE AND NOT DEFINED OpenMP_ROOT)
    execute_process(COMMAND brew --prefix libomp
                    OUTPUT_VARIABLE _libomp OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(_libomp)
      set(OpenMP_ROOT ${_libomp})
    endif()
  endif()
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(chirp_core PUBLIC OpenMP::OpenMP_CXX)
  else()
    message(WARNING "OpenMP not found: building the serial version")
  endif()
endif()

# ---- Python extension (built when invoked by scikit-build-core or with Python available)
find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
                OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core bindings/core.cpp)
target_link_libraries(_core PRIVATE chirp_core)

install(TARGETS _core LIBRARY DESTINATION chirp_interf)
