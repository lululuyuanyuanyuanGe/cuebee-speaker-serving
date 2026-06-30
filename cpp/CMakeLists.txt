add_library(cuebee_worker_core
    src/deterministic_engine.cc
    src/protocol.cc
)
target_include_directories(cuebee_worker_core PUBLIC include)
target_compile_features(cuebee_worker_core PUBLIC cxx_std_17)
target_compile_options(cuebee_worker_core PRIVATE
    $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-Wall;-Wextra;-Wpedantic>
)

if(CUEBEE_ENABLE_ONNX)
    find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
        HINTS "${ONNXRUNTIME_ROOT}/include"
        REQUIRED
    )
    find_library(ONNXRUNTIME_LIBRARY onnxruntime
        HINTS "${ONNXRUNTIME_ROOT}/lib"
        REQUIRED
    )
    target_sources(cuebee_worker_core PRIVATE src/onnx_engine.cc)
    target_include_directories(cuebee_worker_core PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
    target_link_libraries(cuebee_worker_core PUBLIC "${ONNXRUNTIME_LIBRARY}")
    target_compile_definitions(cuebee_worker_core PUBLIC CUEBEE_ENABLE_ONNX=1)
    set_target_properties(cuebee_worker_core PROPERTIES
        BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
    )
else()
    target_compile_definitions(cuebee_worker_core PUBLIC CUEBEE_ENABLE_ONNX=0)
endif()

add_executable(cuebee-speaker-worker src/worker_main.cc)
target_link_libraries(cuebee-speaker-worker PRIVATE cuebee_worker_core)
set_target_properties(cuebee-speaker-worker PROPERTIES
    INSTALL_RPATH "$ORIGIN;@loader_path"
)

add_executable(cuebee-protocol-test tests/protocol_test.cc)
target_link_libraries(cuebee-protocol-test PRIVATE cuebee_worker_core)
add_test(NAME cuebee-protocol-test COMMAND cuebee-protocol-test)

install(TARGETS cuebee-speaker-worker RUNTIME DESTINATION bin)
