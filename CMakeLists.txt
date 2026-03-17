cmake_minimum_required(VERSION 3.10)

project(nn VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(nn main.cpp 
    src/vector.cpp
    src/matrix.cpp 
    src/utils.cpp
    src/network.cpp 
)     
target_include_directories(nn PRIVATE include)


include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.8.1
)

FetchContent_MakeAvailable(Catch2)

add_executable(tests tests.cpp 
    src/vector.cpp
    src/matrix.cpp 
    src/utils.cpp
    src/network.cpp 
)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)
target_include_directories(tests PRIVATE include)