if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "The FPGA UIO integration targets Linux")
endif()
target_sources(MNNCPU PRIVATE
    "${MNN_FPGA_ROOT}/mnn/fpga_bridge.cpp"
    "${MNN_FPGA_ROOT}/sw/src/fpga_gemm.cpp"
    "${MNN_FPGA_ROOT}/sw/src/uio.cpp")
target_include_directories(MNNCPU PRIVATE "${MNN_FPGA_ROOT}/mnn" "${MNN_FPGA_ROOT}/sw/include")
target_compile_definitions(MNNCPU PRIVATE MNN_FPGA_ENABLED=1)
# MNN globally disables exceptions; our access layer catches them locally.
set_source_files_properties(
    "${MNN_FPGA_ROOT}/mnn/fpga_bridge.cpp"
    "${MNN_FPGA_ROOT}/sw/src/fpga_gemm.cpp"
    "${MNN_FPGA_ROOT}/sw/src/uio.cpp"
    PROPERTIES COMPILE_OPTIONS "-fexceptions")
