# ==============================================================================
# SpecTile — Rule-Based Multi-Sample Spectral Mosaicing & Chimeric Resynthesis
# Master CMakeLists.txt (C++20 / SIMD SoA / JUCE 8 / CLAP / Catch2 v3)
# ==============================================================================
cmake_minimum_required(VERSION 3.24)

project(SpecTile
    VERSION 1.0.0
    DESCRIPTION "Rule-Based Multi-Sample Spectral Mosaicing & Chimeric Resynthesis Synthesizer"
    LANGUAGES C CXX
)

# ------------------------------------------------------------------------------
# Build Options
# ------------------------------------------------------------------------------
option(SPECTILE_BUILD_PLUGIN     "Build JUCE 8 VST3/AU/CLAP/Standalone Plugin" ON)
option(SPECTILE_BUILD_TESTS      "Build Catch2 v3 Unit & Golden Tensor Tests"  ON)
option(SPECTILE_COPY_PLUGIN_STEP "Copy built plugin binaries to system folder" OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(Dependencies)

# ------------------------------------------------------------------------------
# 1. Core DSP & Generative Rule Library (spectile_dsp_core — JUCE-Independent)
# ------------------------------------------------------------------------------
file(GLOB_RECURSE SPECTILE_CORE_HEADERS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/rules/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/engine/StaggeredHopScheduler.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/engine/SpecTileVoice.hpp"
)

file(GLOB_RECURSE SPECTILE_CORE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/rules/*.cpp"
)

if(SPECTILE_CORE_SOURCES)
    add_library(spectile_dsp_core STATIC
        ${SPECTILE_CORE_SOURCES}
        ${SPECTILE_CORE_HEADERS}
    )
    target_include_directories(spectile_dsp_core PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
    )
    if(TARGET xsimd)
        target_link_libraries(spectile_dsp_core PUBLIC xsimd)
    endif()
    spectile_configure_target(spectile_dsp_core)
else()
    add_library(spectile_dsp_core INTERFACE
        ${SPECTILE_CORE_HEADERS}
    )
    target_include_directories(spectile_dsp_core INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
    )
    if(TARGET xsimd)
        target_link_libraries(spectile_dsp_core INTERFACE xsimd)
    endif()
    target_compile_features(spectile_dsp_core INTERFACE cxx_std_20)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64|amd64)")
        if(MSVC)
            target_compile_options(spectile_dsp_core INTERFACE /arch:AVX2 /fp:fast /Zc:__cplusplus /permissive-)
            target_compile_definitions(spectile_dsp_core INTERFACE SPECTILE_SIMD_AVX2=1 SPECTILE_SIMD_FMA=1 SPECTILE_SIMD_F16C=1 NOMINMAX _USE_MATH_DEFINES)
        else()
            target_compile_options(spectile_dsp_core INTERFACE -mavx2 -mfma -mf16c -ffast-math -fno-finite-math-only)
            target_compile_definitions(spectile_dsp_core INTERFACE SPECTILE_SIMD_AVX2=1 SPECTILE_SIMD_FMA=1 SPECTILE_SIMD_F16C=1)
        endif()
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(arm64|aarch64|ARM64)")
        target_compile_definitions(spectile_dsp_core INTERFACE SPECTILE_SIMD_NEON=1)
    endif()
endif()

# ------------------------------------------------------------------------------
# 2. SpecTile VST3 / AU / CLAP / Standalone Plugin Target (JUCE 8)
# ------------------------------------------------------------------------------
if(SPECTILE_BUILD_PLUGIN AND COMMAND juce_add_plugin)
    set(SPECTILE_PLUGIN_FORMATS VST3 Standalone)
    if(APPLE)
        list(APPEND SPECTILE_PLUGIN_FORMATS AU)
    endif()

    juce_add_plugin(SpecTile
        COMPANY_NAME                "SpecTile Audio Research"
        COMPANY_WEBSITE             "https://spectile.audio"
        COMPANY_EMAIL               "research@spectile.audio"
        BUNDLE_ID                   "org.spectile.audio.SpecTile"
        IS_SYNTH                    TRUE
        NEEDS_MIDI_INPUT            TRUE
        NEEDS_MIDI_OUTPUT           FALSE
        IS_MIDI_EFFECT              FALSE
        EDITOR_WANTS_KEYBOARD_FOCUS TRUE
        COPY_PLUGIN_AFTER_BUILD     ${SPECTILE_COPY_PLUGIN_STEP}
        PLUGIN_MANUFACTURER_CODE    SpTl
        PLUGIN_CODE                 SpT1
        FORMATS                     ${SPECTILE_PLUGIN_FORMATS}
        PRODUCT_NAME                "SpecTile"
    )

    # Enable CLAP format with Polyphonic Note Expression / MPE support if available
    if(COMMAND clap_juce_extensions_plugin)
        clap_juce_extensions_plugin(
            TARGET SpecTile
            CLAP_ID "org.spectile.audio.spectile"
            CLAP_FEATURES
                "instrument"
                "synthesizer"
                "sampler"
                "spectral"
                "stereo"
                "mpe"
        )
    endif()

    file(GLOB_RECURSE SPECTILE_ENGINE_UI_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/engine/*.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/engine/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/ui/*.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/ui/*.cpp"
    )

    target_sources(SpecTile PRIVATE
        ${SPECTILE_ENGINE_UI_SOURCES}
    )

    target_include_directories(SpecTile PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
    )

    # Optional Embedded GLSL Shaders / Assets
    file(GLOB_RECURSE SPECTILE_SHADER_FILES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/ui/shaders/*.glsl"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/ui/shaders/*.vert"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/ui/shaders/*.frag"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/ui/shaders/*.metal"
    )
    if(SPECTILE_SHADER_FILES)
        juce_add_binary_data(SpecTileShaders
            HEADER_NAME "SpecTileShaderBinaryData.h"
            NAMESPACE   spectile_shaders
            SOURCES     ${SPECTILE_SHADER_FILES}
        )
        target_link_libraries(SpecTile PRIVATE SpecTileShaders)
        target_compile_definitions(SpecTile PUBLIC SPECTILE_HAS_SHADER_BINARY_DATA=1)
    endif()

    target_compile_definitions(SpecTile PUBLIC
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0
        JUCE_DISPLAY_SPLASH_SCREEN=0
        JUCE_USE_FLAC=1
        JUCE_USE_OGGVORBIS=1
        JUCE_DSP_USE_INTEL_MKL=0
    )

    target_link_libraries(SpecTile
        PRIVATE
            spectile_dsp_core
            juce::juce_audio_basics
            juce::juce_audio_devices
            juce::juce_audio_formats
            juce::juce_audio_plugin_client
            juce::juce_audio_processors
            juce::juce_audio_utils
            juce::juce_core
            juce::juce_cryptography
            juce::juce_data_structures
            juce::juce_dsp
            juce::juce_events
            juce::juce_graphics
            juce::juce_gui_basics
            juce::juce_gui_extra
            juce::juce_opengl
        PUBLIC
            juce::juce_recommended_config_flags
            juce::juce_recommended_lto_flags
    )

    spectile_configure_target(SpecTile)
endif()

# ------------------------------------------------------------------------------
# 3. Catch2 v3 Unit & Golden Reference Test Suite (spectile_unit_tests)
# ------------------------------------------------------------------------------
if(SPECTILE_BUILD_TESTS)
    enable_testing()

    file(GLOB_RECURSE SPECTILE_TEST_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp"
    )

    if(SPECTILE_TEST_SOURCES)
        add_executable(spectile_unit_tests
            ${SPECTILE_TEST_SOURCES}
        )

        target_include_directories(spectile_unit_tests PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests"
        )

        target_compile_definitions(spectile_unit_tests PRIVATE
            SPECTILE_GOLDEN_TENSORS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/spectile_proto/golden_tensors"
            SPECTILE_GOLDEN_JSON_PATH="${CMAKE_CURRENT_SOURCE_DIR}/tests/golden_data/golden_vectors.json"
        )

        target_link_libraries(spectile_unit_tests PRIVATE
            spectile_dsp_core
        )
        if(TARGET Catch2::Catch2WithMain)
            target_link_libraries(spectile_unit_tests PRIVATE Catch2::Catch2WithMain)
        endif()

        spectile_configure_target(spectile_unit_tests)

        if(TARGET Catch2::Catch2WithMain)
            include(Catch)
            catch_discover_tests(spectile_unit_tests)
        else()
            add_test(NAME spectile_unit_tests COMMAND spectile_unit_tests)
        endif()
    endif()
endif()

# ------------------------------------------------------------------------------
# 4. Native C++20 Standalone CLI Synthesizer & Real-Time Profiler (spectile_cli)
# ------------------------------------------------------------------------------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/cli/SpecTileCLI.cpp")
    add_executable(spectile_cli
        "${CMAKE_CURRENT_SOURCE_DIR}/src/cli/SpecTileCLI.cpp"
    )

    target_include_directories(spectile_cli PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
    )

    target_link_libraries(spectile_cli PRIVATE
        spectile_dsp_core
    )

    spectile_configure_target(spectile_cli)
endif()
