# ==============================================================================
# SpecTile — cmake/Dependencies.cmake
# Settore 3: Real-Time C++20, SIMD (xsimd), JUCE 8, CLAP & Catch2 Dependencies
# ==============================================================================
include_guard(GLOBAL)
include(FetchContent)
include(CheckCXXCompilerFlag)

option(SPECTILE_USE_SYSTEM_PACKAGES "Prefer find_package() before FetchContent" OFF)
option(SPECTILE_FETCH_EXTERNAL_DEPS "Fetch external dependencies (xsimd, Catch2, JUCE) from GitHub" OFF)
option(SPECTILE_ENABLE_IPP          "Enable Intel IPP FFT backend if available"  OFF)
option(SPECTILE_ENABLE_ASAN         "Enable AddressSanitizer + UBSan"            OFF)
option(SPECTILE_ENABLE_TSAN         "Enable ThreadSanitizer"                     OFF)
option(SPECTILE_ENABLE_RTSAN        "Enable Clang RealtimeSanitizer (-fsanitize=realtime)" OFF)

# ------------------------------------------------------------------------------
# 1. xsimd (Unified C++20 SIMD Intrinsics: AVX2 / AVX-512 / ARM64 NEON)
# ------------------------------------------------------------------------------
if(SPECTILE_USE_SYSTEM_PACKAGES)
    find_package(xsimd QUIET)
endif()

if(NOT xsimd_FOUND AND SPECTILE_FETCH_EXTERNAL_DEPS)
    message(STATUS "[SpecTile] Fetching xsimd 13.0.0 (Header-only SIMD abstraction)...")
    FetchContent_Declare(
        xsimd
        GIT_REPOSITORY https://github.com/xtensor-stack/xsimd.git
        GIT_TAG        13.0.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(xsimd)
endif()

# ------------------------------------------------------------------------------
# 2. JUCE 8 & clap-juce-extensions (Only when building Plugin targets)
# ------------------------------------------------------------------------------
option(SPECTILE_BUILD_CLAP "Build CLAP plugin format via clap-juce-extensions" OFF)

if(SPECTILE_BUILD_PLUGIN)
    set(_spectile_restore_host_proc FALSE)
    set(_spectile_arm64_x64_cross FALSE)
    if(WIN32 AND CMAKE_GENERATOR MATCHES "^Visual Studio" AND CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "ARM64" AND CMAKE_VS_PLATFORM_NAME STREQUAL "x64")
        # Avoid juceaide sub-configure defaulting to Platform=ARM64 on ARM64 Windows hosts without ARM64 MSVC tools
        set(_spectile_saved_host_proc "${CMAKE_HOST_SYSTEM_PROCESSOR}")
        set(CMAKE_HOST_SYSTEM_PROCESSOR "x64")
        set(_spectile_restore_host_proc TRUE)
        set(_spectile_arm64_x64_cross TRUE)
    endif()

    set(_spectile_restore_cl FALSE)
    if(MSVC)
        # Ensure juceaide's internal Debug sub-build embeds debug info (/Z7) instead of spawning mspdbsrv.exe (/Zi)
        if(DEFINED ENV{_CL_})
            set(_spectile_had_cl TRUE)
            set(_spectile_saved_cl "$ENV{_CL_}")
            if(NOT "$ENV{_CL_}" MATCHES "/Z[7iI]")
                set(ENV{_CL_} "$ENV{_CL_} /Z7")
                set(_spectile_restore_cl TRUE)
            endif()
        else()
            set(_spectile_had_cl FALSE)
            set(ENV{_CL_} "/Z7")
            set(_spectile_restore_cl TRUE)
        endif()
    endif()

    if(EXISTS "${CMAKE_SOURCE_DIR}/external/JUCE/CMakeLists.txt")
        message(STATUS "[SpecTile] Using local JUCE 8 from ${CMAKE_SOURCE_DIR}/external/JUCE...")
        if(_spectile_arm64_x64_cross AND NOT EXISTS "${CMAKE_BINARY_DIR}/JUCE/tools/extras/Build/juceaide/juceaide.vcxproj")
            file(REMOVE_RECURSE "${CMAKE_BINARY_DIR}/JUCE/tools")
            execute_process(
                COMMAND "${CMAKE_COMMAND}"
                    -S "${CMAKE_SOURCE_DIR}/external/JUCE"
                    -B "${CMAKE_BINARY_DIR}/JUCE/tools"
                    -G "${CMAKE_GENERATOR}"
                    -A "${CMAKE_VS_PLATFORM_NAME}"
                    "-DCMAKE_BUILD_TYPE=Debug"
                    "-DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}"
                    "-DJUCE_BUILD_HELPER_TOOLS=ON"
            )
        endif()
        add_subdirectory("${CMAKE_SOURCE_DIR}/external/JUCE" "${CMAKE_BINARY_DIR}/JUCE" EXCLUDE_FROM_ALL)
        set(JUCE_FOUND TRUE)
    elseif(SPECTILE_USE_SYSTEM_PACKAGES)
        find_package(JUCE CONFIG QUIET)
    endif()

    if(NOT JUCE_FOUND)
        message(STATUS "[SpecTile] Local external/JUCE not found; fetching JUCE 8.0.4...")
        FetchContent_Declare(
            JUCE
            GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
            GIT_TAG        8.0.4
            GIT_SHALLOW    TRUE
        )
        FetchContent_MakeAvailable(JUCE)
    endif()

    if(_spectile_restore_cl)
        if(_spectile_had_cl)
            set(ENV{_CL_} "${_spectile_saved_cl}")
        else()
            unset(ENV{_CL_})
        endif()
    endif()
    if(_spectile_restore_host_proc)
        set(CMAKE_HOST_SYSTEM_PROCESSOR "${_spectile_saved_host_proc}")
    endif()

    if(SPECTILE_BUILD_CLAP AND SPECTILE_FETCH_EXTERNAL_DEPS)
        message(STATUS "[SpecTile] Fetching clap-juce-extensions...")
        FetchContent_Declare(
            clap_juce_extensions
            GIT_REPOSITORY https://github.com/free-audio/clap-juce-extensions.git
            GIT_TAG        main
            GIT_SHALLOW    TRUE
        )
        FetchContent_MakeAvailable(clap_juce_extensions)
    endif()
endif()

# ------------------------------------------------------------------------------
# 3. Catch2 v3 (Unit Test Oracle against Python Golden Tensors)
# ------------------------------------------------------------------------------
if(SPECTILE_BUILD_TESTS AND SPECTILE_FETCH_EXTERNAL_DEPS)
    if(SPECTILE_USE_SYSTEM_PACKAGES)
        find_package(Catch2 3 QUIET)
    endif()

    if(NOT Catch2_FOUND)
        message(STATUS "[SpecTile] Fetching Catch2 v3.7.1...")
        FetchContent_Declare(
            Catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2.git
            GIT_TAG        v3.7.1
            GIT_SHALLOW    TRUE
        )
        FetchContent_MakeAvailable(Catch2)
        list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
    endif()
endif()

# ------------------------------------------------------------------------------
# 4. Platform FFT & SIMD Compiler Flag Configuration Function
# ------------------------------------------------------------------------------
function(spectile_configure_target target_name)
    target_compile_features(${target_name} PUBLIC cxx_std_20)
    set_target_properties(${target_name} PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        POSITION_INDEPENDENT_CODE ON
    )

    # Detect CPU Architecture for SIMD flags (AVX2+FMA+F16C on x86_64, NEON on ARM64)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64|amd64)")
        if(MSVC)
            target_compile_options(${target_name} PUBLIC
                /arch:AVX2
                /fp:fast
                /Oi
                /Ot
                /Zc:__cplusplus
                /permissive-
                /W4
                /wd4100 # unreferenced formal parameter
                /wd4324 # structure was padded due to alignment specifier (alignas(64))
            )
            target_compile_definitions(${target_name} PUBLIC
                SPECTILE_SIMD_AVX2=1
                SPECTILE_SIMD_FMA=1
                SPECTILE_SIMD_F16C=1
                NOMINMAX
                _USE_MATH_DEFINES
            )
        else()
            target_compile_options(${target_name} PUBLIC
                -mavx2
                -mfma
                -mf16c
                -ffast-math
                -fno-finite-math-only
                -Wall
                -Wextra
                -Wpedantic
                -Wno-unused-parameter
            )
            target_compile_definitions(${target_name} PUBLIC
                SPECTILE_SIMD_AVX2=1
                SPECTILE_SIMD_FMA=1
                SPECTILE_SIMD_F16C=1
            )
        endif()
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(arm64|aarch64|ARM64)")
        if(NOT MSVC)
            target_compile_options(${target_name} PUBLIC
                -ffast-math
                -fno-finite-math-only
                -Wall
                -Wextra
                -Wpedantic
                -Wno-unused-parameter
            )
        endif()
        target_compile_definitions(${target_name} PUBLIC
            SPECTILE_SIMD_NEON=1
        )
    endif()

    # Platform-specific hardware FFT acceleration
    if(APPLE)
        find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
        target_link_libraries(${target_name} PUBLIC ${ACCELERATE_FRAMEWORK})
        target_compile_definitions(${target_name} PUBLIC SPECTILE_FFT_VDPS_ACCELERATE=1)
    elseif(SPECTILE_ENABLE_IPP)
        find_package(IPP QUIET)
        if(IPP_FOUND)
            target_link_libraries(${target_name} PUBLIC IPP::ipps IPP::ippcore)
            target_compile_definitions(${target_name} PUBLIC SPECTILE_FFT_INTEL_IPP=1)
        endif()
    endif()

    # Sanitizers (Zero-Tolerance QA Gate)
    if(SPECTILE_ENABLE_ASAN AND NOT MSVC)
        target_compile_options(${target_name} PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(${target_name} PUBLIC -fsanitize=address,undefined)
    elseif(SPECTILE_ENABLE_ASAN AND MSVC)
        target_compile_options(${target_name} PUBLIC /fsanitize=address)
    endif()

    if(SPECTILE_ENABLE_TSAN AND NOT MSVC)
        target_compile_options(${target_name} PUBLIC -fsanitize=thread -fno-omit-frame-pointer)
        target_link_options(${target_name} PUBLIC -fsanitize=thread)
    endif()

    if(SPECTILE_ENABLE_RTSAN AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(${target_name} PUBLIC -fsanitize=realtime)
        target_link_options(${target_name} PUBLIC -fsanitize=realtime)
    endif()
endfunction()
