# FetchAssets.cmake — provision the example / test datasets into ${data}.
#
# data/ is .gitignore'd: the assets are large (a 97 MB EXR, an 8 MB glTF, a
# multi-MB OBJ) and some carry their own licenses, so they are downloaded at
# configure time rather than vendored. A fresh checkout becomes runnable with
# no manual steps. Each asset is guarded by a sentinel file, so this is a
# no-op once data/ is populated (in particular it does NOT re-clone the large
# glTF-Sample-Models repo on every fresh build tree).
#
# Set -DDRESSI_FETCH_DATA=OFF to skip entirely (offline / air-gapped CI that
# provisions data/ by other means). Downloads land in a build-tree staging
# dir first and are hash-verified before being moved into data/, so an
# interrupted or corrupted download never leaves a bad file behind the
# sentinel guard.

option(DRESSI_FETCH_DATA "Download example/test datasets at configure time" ON)
if(NOT DRESSI_FETCH_DATA)
    return()
endif()

include(FetchContent)

set(_data_dir "${CMAKE_SOURCE_DIR}/data")
set(_stage_dir "${CMAKE_BINARY_DIR}/_assets")
file(MAKE_DIRECTORY "${_data_dir}" "${_stage_dir}")

# Download <url> to a staging file, hash-check it, then place it at <dst>.
# Fails the configure with a clear message on any error and never leaves a
# partial file at <dst>.
function(_dressi_fetch_file url sha256 dst)
    get_filename_component(_name "${dst}" NAME)
    set(_tmp "${_stage_dir}/${_name}")
    message(STATUS "dressi: fetching ${_name}")
    file(DOWNLOAD "${url}" "${_tmp}"
        EXPECTED_HASH SHA256=${sha256}
        SHOW_PROGRESS
        STATUS _status)
    list(GET _status 0 _code)
    if(NOT _code EQUAL 0)
        list(GET _status 1 _msg)
        file(REMOVE "${_tmp}")
        message(FATAL_ERROR "dressi: download of ${_name} failed: ${_msg}")
    endif()
    get_filename_component(_dst_dir "${dst}" DIRECTORY)
    file(MAKE_DIRECTORY "${_dst_dir}")
    file(COPY_FILE "${_tmp}" "${dst}")
endfunction()

# --- bunny (kunzhou.net tex-models): OBJ + MTL + atlas, a 1.1 MB zip -------
# The archive holds the three files at its root; extract straight into
# data/bunny/ (the example default dir).
if(NOT EXISTS "${_data_dir}/bunny/bunny.obj")
    _dressi_fetch_file(
        "http://www.kunzhou.net/tex-models/bunny.zip"
        "a1fb80b85aedfd10f7e677f24c5984c42611d0717f1b830e3187908826bc756f"
        "${_stage_dir}/bunny.zip")
    file(ARCHIVE_EXTRACT INPUT "${_stage_dir}/bunny.zip"
        DESTINATION "${_data_dir}/bunny")
endif()

# --- glTF-Sample-Models 2.0 meshes: Avocado (Table 4) + DamagedHelmet (PBS)
# Pulled via a full git clone of the sample-models repo (~1.2 GB), then only
# the needed glTF folders are copied into data/ so runtime paths stay uniform
# with the rest of data/. Guarded on the copied files so the huge clone
# happens at most once per machine (skipped whenever every model is present).
if(NOT EXISTS "${_data_dir}/Avocado/glTF/Avocado.gltf" OR
   NOT EXISTS "${_data_dir}/DamagedHelmet/glTF/DamagedHelmet.gltf")
    message(STATUS "dressi: cloning glTF-Sample-Models (large, one-time)")
    FetchContent_Declare(gltf_sample_models
        GIT_REPOSITORY https://github.com/KhronosGroup/glTF-Sample-Models.git
        GIT_TAG main
        GIT_SHALLOW TRUE
        SOURCE_SUBDIR nonexistent_disable_add_subdirectory)
    FetchContent_MakeAvailable(gltf_sample_models)
    if(NOT EXISTS "${_data_dir}/Avocado/glTF/Avocado.gltf")
        file(COPY "${gltf_sample_models_SOURCE_DIR}/2.0/Avocado/glTF"
            DESTINATION "${_data_dir}/Avocado")
    endif()
    if(NOT EXISTS "${_data_dir}/DamagedHelmet/glTF/DamagedHelmet.gltf")
        file(COPY "${gltf_sample_models_SOURCE_DIR}/2.0/DamagedHelmet/glTF"
            DESTINATION "${_data_dir}/DamagedHelmet")
    endif()
endif()

# --- Environment map for the upcoming PBS shading (polyhaven 4k EXR) -------
if(NOT EXISTS "${_data_dir}/suburban_garden_4k.exr")
    _dressi_fetch_file(
        "https://dl.polyhaven.org/file/ph-assets/HDRIs/exr/4k/suburban_garden_4k.exr"
        "fe13dafff24598869e908a9e4fc73b5b8f2cf973767774be76a108e5a8f2171c"
        "${_data_dir}/suburban_garden_4k.exr")
endif()
