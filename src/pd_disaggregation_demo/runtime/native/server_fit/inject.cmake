# Register the external target; upstream library targets resolve at generation.
# The pinned upstream source and all inference/library targets remain untouched.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR AND PROJECT_NAME STREQUAL "llama.cpp")
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.txt")
endif()
