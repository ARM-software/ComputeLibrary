#
# SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
include(FetchContent)

# Set timestamp of extracted contents to time of extraction.
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

fetchcontent_declare(googletest
    URL         ${CMAKE_CURRENT_SOURCE_DIR}/third_party/googletest-v1.14.0.zip
    URL_HASH    SHA256=1f357c27ca988c3f7c6b4bf68a9395005ac6761f034046e9dde0896e3aba00e4
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

fetchcontent_makeavailable(googletest)
