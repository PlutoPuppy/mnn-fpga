cmake_minimum_required(VERSION 3.16)
project(mnn_fpga_smoke CXX)
set(CMAKE_CXX_STANDARD 17)
if(NOT MNN_SOURCE OR NOT MNN_BUILD)
    message(FATAL_ERROR "Set MNN_SOURCE and MNN_BUILD to the pinned source/build directories")
endif()
add_executable(mnn_tile_test mnn_tile_test.cpp)
target_include_directories(mnn_tile_test PRIVATE
    "${MNN_SOURCE}/include" "${MNN_SOURCE}/schema/current"
    "${MNN_SOURCE}/3rd_party/flatbuffers/include" "${CMAKE_CURRENT_LIST_DIR}/..")
find_library(MNN_LIB MNN PATHS "${MNN_BUILD}" NO_DEFAULT_PATH REQUIRED)
target_link_libraries(mnn_tile_test PRIVATE "${MNN_LIB}")
