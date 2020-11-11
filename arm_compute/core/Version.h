/*
 * Copyright (c) 2020 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#ifndef ARM_COMPUTE_LIBRARY_VERSION_H
#define ARM_COMPUTE_LIBRARY_VERSION_H

#include <string>

/* Macro utilities */
#define ARM_COMPUTE_STRINGIFY2(s) #s
#define ARM_COMPUTE_STRINGIFY(s) ARM_COMPUTE_STRINGIFY2(s)

#define ARM_COMPUTE_VERSION_STR                      \
    ARM_COMPUTE_STRINGIFY(ARM_COMPUTE_VERSION_MAJOR) \
    "." ARM_COMPUTE_STRINGIFY(ARM_COMPUTE_VERSION_MINOR) "." ARM_COMPUTE_STRINGIFY(ARM_COMPUTE_VERSION_PATCH)

namespace arm_compute
{
/** Returns the arm_compute library build information
 *
 * Contains the version number and the build options used to build the library
 *
 * @return The arm_compute library build information
 */
std::string build_information();
} // namespace arm_compute

#undef ARM_COMPUTE_STRINGIFY
#undef ARM_COMPUTE_STRINGIFY2

#endif /* ARM_COMPUTE_LIBRARY_VERSION_H */
