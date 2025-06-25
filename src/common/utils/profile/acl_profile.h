/*
 * Copyright (c) 2025 Arm Limited.
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

#ifndef ACL_SRC_COMMON_UTILS_PROFILE_ACL_PROFILE_H
#define ACL_SRC_COMMON_UTILS_PROFILE_ACL_PROFILE_H

// Define ACL profile categories
#define PROF_CAT_NONE      "NONE"
#define PROF_CAT_CPU       "CPU"
#define PROF_CAT_NEON      "NEON"
#define PROF_CAT_SVE       "SVE"
#define PROF_CAT_SME       "SME"
#define PROF_CAT_GPU       "GPU"
#define PROF_CAT_MEMORY    "MEMORY"
#define PROF_CAT_RUNTIME   "RUNTIME"
#define PROF_CAT_SCHEDULER "SCHEDULER"

// Define ACL profile levels
enum class ProfileLevel
{
    L0 = 0,
    L1 = 1,
    L2 = 2
};

#define ARM_COMPUTE_TRACE_EVENT(category, level, name) \
    (void)category;                                    \
    (void)name;                                        \
    (void)level
#define ARM_COMPUTE_TRACE_EVENT_BEGIN(category, level, name) \
    (void)category;                                          \
    (void)name;                                              \
    (void)level
#define ARM_COMPUTE_TRACE_EVENT_END(category, level) \
    (void)category;                                  \
    (void)level
#define ARM_COMPUTE_PROFILE_INIT() \
    do                             \
    {                              \
    } while (0)
#define ARM_COMPUTE_PROFILE_STATIC_STORAGE() \
    do                                       \
    {                                        \
    } while (0)
#define ARM_COMPUTE_PROFILE_FINISH() \
    do                               \
    {                                \
    } while (0)

#endif // ACL_SRC_COMMON_UTILS_PROFILE_ACL_PROFILE_H
