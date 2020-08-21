/*
 * Copyright (c) 2019 Arm Limited.
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
#ifndef ARM_COMPUTE_MISC_MACROS_H
#define ARM_COMPUTE_MISC_MACROS_H

#if defined(__cplusplus) && (__cplusplus >= 201402L)

#define ARM_COMPUTE_DEPRECATED [[deprecated]]
#define ARM_COMPUTE_DEPRECATED_REL(rel) [[deprecated("Deprecated in : " #rel)]]
#define ARM_COMPUTE_DEPRECATED_REL_REPLACE(rel, replace) [[deprecated("Deprecated in : " #rel " - Use : " #replace)]]

#elif defined(__GNUC__) || defined(__clang__)

#define ARM_COMPUTE_DEPRECATED __attribute__((deprecated))
#define ARM_COMPUTE_DEPRECATED_REL(rel) __attribute__((deprecated("Deprecated in : " #rel)))
#define ARM_COMPUTE_DEPRECATED_REL_REPLACE(rel, replace) __attribute__((deprecated("Deprecated in : " #rel " - Use : " #replace)))

#else // defined(__cplusplus) && (__cplusplus >= 201402L)

#define ARM_COMPUTE_DEPRECATED
#define ARM_COMPUTE_DEPRECATED_REL(rel)
#define ARM_COMPUTE_DEPRECATED_REL_REPLACE(rel, replace)

#endif // defined(__cplusplus) && (__cplusplus >= 201402L)

#endif /* ARM_COMPUTE_MISC_MACROS_H */
