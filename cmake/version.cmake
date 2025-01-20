# Copyright (c) 2023-2025 Arm Limited.
#
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

find_package(Git)

if(GIT_FOUND)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    RESULT_VARIABLE RESULT
    OUTPUT_VARIABLE ACL_VERSION_SHA
    OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

if(NOT GIT_FOUND OR RESULT)
  set(ACL_VERSION_HASH "Unknown")
endif()

set(ARM_COMPUTE_SHA_FILE "${PROJECT_SOURCE_DIR}/arm_compute_version.embed")
file(WRITE "${ARM_COMPUTE_SHA_FILE}.tmp" "\"${ACL_VERSION_SHA}\"")

execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  ${ARM_COMPUTE_SHA_FILE}.tmp
  ${ARM_COMPUTE_SHA_FILE}
)
file(REMOVE ${ARM_COMPUTE_SHA_FILE}.tmp)
