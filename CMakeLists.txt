cmake_minimum_required(VERSION 3.13)
project(OpencvApp)

set(CMAKE_CXX_STANDARD 14)

#add libs you need
set(
        SOURCE_FILES
        lib/OpenCVOp/Base.h
        lib/OpenCVOp/Base.cpp
        lib/OpenCVOp/OpencvTest.h
        lib/OpenCVOp/OpencvTest.cpp
)

add_executable(OpencvApp main.cpp ${SOURCE_FILES})

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(OpencvApp ${OpenCV_LIBS})




