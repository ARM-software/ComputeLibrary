#
# SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
include(FetchContent)

# Set timestamp of extracted contents to time of extraction.
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

fetchcontent_declare(googlebench
    URL         ${CMAKE_CURRENT_SOURCE_DIR}/third_party/benchmark-v1.9.4.zip
    URL_HASH    SHA256=7a273667fbc23480df1306f82bdb960672811dd29a0342bb34e14040307cf820
)

set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(BENCHMARK_INSTALL_DOCS OFF CACHE BOOL "" FORCE)

fetchcontent_makeavailable(googlebench)
