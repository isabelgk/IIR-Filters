cmake_minimum_required(VERSION 3.20)
project(IirFiltersDemo)

include(FetchContent)

# Fetch GLFW
FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG 3.4
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# Fetch Dear ImGui
FetchContent_Declare(
        imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG v1.91.6
)
FetchContent_MakeAvailable(imgui)

# Fetch ImPlot
FetchContent_Declare(
        implot
        GIT_REPOSITORY https://github.com/epezent/implot.git
        GIT_TAG v0.16
)
FetchContent_MakeAvailable(implot)

# Create ImGui library (ImGui doesn't have CMakeLists.txt)
add_library(imgui_lib STATIC
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_demo.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui_lib PUBLIC
        ${imgui_SOURCE_DIR}
        ${imgui_SOURCE_DIR}/backends
)
target_link_libraries(imgui_lib PUBLIC glfw)

# Create ImPlot library
add_library(implot_lib STATIC
        ${implot_SOURCE_DIR}/implot.cpp
        ${implot_SOURCE_DIR}/implot_items.cpp
)
target_include_directories(implot_lib PUBLIC ${implot_SOURCE_DIR})
target_link_libraries(implot_lib PUBLIC imgui_lib)

# Find OpenGL
find_package(OpenGL REQUIRED)

# Demo executable
add_executable(imgui_demo
        main.cpp
        app.cpp
        filter_designer.cpp
        frequency_response.cpp
)

target_include_directories(imgui_demo PRIVATE
        ${CMAKE_SOURCE_DIR}/source
)

# Silence OpenGL deprecation warnings on macOS
if (APPLE)
    target_compile_definitions(imgui_demo PRIVATE GL_SILENCE_DEPRECATION)
endif ()

target_link_libraries(imgui_demo PRIVATE
        IirFilters
        imgui_lib
        implot_lib
        OpenGL::GL
)
