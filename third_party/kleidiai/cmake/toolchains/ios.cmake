#
# SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

set(CMAKE_SYSTEM_NAME       iOS)
set(CMAKE_OSX_ARCHITECTURES "arm64;arm64e")
set(CMAKE_CROSSCOMPILING    TRUE)

set(CMAKE_XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY "" CACHE STRING "")

add_compile_options(-Wno-shorten-64-to-32)
