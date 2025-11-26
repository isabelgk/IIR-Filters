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
target_link_libraries(tests PRIVATE IirFilters)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)
