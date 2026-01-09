Include(FetchContent)
FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.11.0
)
FetchContent_MakeAvailable(Catch2)

add_executable(tests
        tests.cpp
)

target_include_directories(tests
        PRIVATE
        "${CMAKE_SOURCE_DIR}/source"
)
if(MSVC)
    target_compile_definitions(tests PRIVATE _USE_MATH_DEFINES)
endif()
target_link_libraries(tests PRIVATE IirFilters)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(tests)
