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

fetchcontent_declare(googlebench
    URL         ${CMAKE_CURRENT_SOURCE_DIR}/third_party/benchmark-v1.8.4.zip
    URL_HASH    SHA256=84c49c4c07074f36fbf8b4f182ed7d75191a6fa72756ab4a17848455499f4286
)

fetchcontent_makeavailable(googlebench)
