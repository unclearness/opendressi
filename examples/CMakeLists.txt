add_executable(image_fitting image_fitting/main.cpp)
target_link_libraries(image_fitting PRIVATE dressi)

add_executable(texture_optimization
    texture_optimization/main.cpp
    common/asset_utils.cpp)
target_link_libraries(texture_optimization PRIVATE dressi dressi_assets)

foreach(_tgt image_fitting texture_optimization)
    if(MSVC)
        target_compile_options(${_tgt} PRIVATE /W4 /permissive- /utf-8)
    endif()
endforeach()
