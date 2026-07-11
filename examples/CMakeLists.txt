add_executable(image_fitting image_fitting/main.cpp)
target_link_libraries(image_fitting PRIVATE dressi)

add_executable(texture_optimization
    texture_optimization/main.cpp
    common/asset_utils.cpp
    common/vk_viewer.cpp)
target_link_libraries(texture_optimization PRIVATE dressi dressi_assets
                                                   dressi_glad glfw
                                                   spdlog::spdlog)

add_executable(silhouette_optimization
    silhouette_optimization/main.cpp
    common/asset_utils.cpp
    common/geometry_utils.cpp
    common/vk_viewer.cpp)
target_link_libraries(silhouette_optimization PRIVATE dressi dressi_assets
                                                      dressi_glad glfw
                                                      spdlog::spdlog)

add_executable(pbr_shading
    pbr_shading/main.cpp
    common/asset_utils.cpp
    common/pbr_graph.cpp
    common/vk_viewer.cpp)
target_link_libraries(pbr_shading PRIVATE dressi dressi_assets dressi_glad
                                          glfw spdlog::spdlog)

foreach(_tgt image_fitting texture_optimization silhouette_optimization
        pbr_shading)
    if(MSVC)
        target_compile_options(${_tgt} PRIVATE /W4 /permissive- /utf-8)
    endif()
endforeach()
