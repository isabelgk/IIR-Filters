add_library(
        IirFilters
        STATIC
        "${CMAKE_CURRENT_LIST_DIR}/filters.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/math_supplement.cpp"
)

target_include_directories(
        IirFilters
        PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}"
)
