cmake_minimum_required(VERSION 3.20)
project(IirFilters)

set(CMAKE_CXX_STANDARD 17)

option(BUILD_DEMO "Build Dear ImGui demo application" OFF)

add_subdirectory(source)
add_subdirectory(test)

if(BUILD_DEMO)
    add_subdirectory(examples/imgui_demo)
endif()
