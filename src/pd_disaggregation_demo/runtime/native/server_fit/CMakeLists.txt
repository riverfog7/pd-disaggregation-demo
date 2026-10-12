# Add one probe executable to the already-configured pinned upstream build.
execute_process(COMMAND git rev-parse HEAD WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE PD_LINKED_SOURCE_REVISION OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
set(PD_PINNED_ENGINE_REVISION "23b0202a189c44a54625aadcb37a946dd1d6278d")
if(NOT PD_LINKED_SOURCE_REVISION STREQUAL PD_PINNED_ENGINE_REVISION)
    message(FATAL_ERROR "Server memory probe requires the pinned upstream source")
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_LIST_DIR}/server-fit-params.cpp")
file(SHA256 "${CMAKE_CURRENT_LIST_DIR}/server-fit-params.cpp" PD_PROBE_SOURCE_SHA256)
add_executable(pd-server-fit-params "${CMAKE_CURRENT_LIST_DIR}/server-fit-params.cpp")
target_link_libraries(pd-server-fit-params PRIVATE llama-common llama)
target_compile_features(pd-server-fit-params PRIVATE cxx_std_17)
target_compile_definitions(pd-server-fit-params PRIVATE
    PD_ENGINE_REVISION="${PD_PINNED_ENGINE_REVISION}"
    PD_PROBE_SOURCE_SHA256="${PD_PROBE_SOURCE_SHA256}")
set_target_properties(pd-server-fit-params PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
