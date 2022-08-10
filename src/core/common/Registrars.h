/*
 * Copyright (c) 2020-2022 Arm Limited.
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
#ifndef SRC_CORE_COMMON_REGISTRARS_H
#define SRC_CORE_COMMON_REGISTRARS_H

#if defined(ENABLE_FP16_KERNELS)

#if defined(ARM_COMPUTE_ENABLE_SVE)
#define REGISTER_FP16_SVE(func_name) &(func_name)
#else /* !defined(ARM_COMPUTE_ENABLE_SVE) */
#define REGISTER_FP16_SVE(func_name) nullptr
#endif /* defined(ARM_COMPUTE_ENABLE_SVE) */

#if defined(ARM_COMPUTE_ENABLE_SVE2)
#define REGISTER_FP16_SVE2(func_name) &(func_name)
#else /* !defined(ARM_COMPUTE_ENABLE_SVE2) */
#define REGISTER_FP16_SVE2(func_name) nullptr
#endif /* defined(ARM_COMPUTE_ENABLE_SVE2) */

#if defined(ARM_COMPUTE_ENABLE_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#define REGISTER_FP16_NEON(func_name) &(func_name)
#else /* !defined(ARM_COMPUTE_ENABLE_NEON) */
#define REGISTER_FP16_NEON(func_name) nullptr
#endif /* defined(ARM_COMPUTE_ENABLE_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) */

#else /* !defined(ENABLE_FP16_KERNELS) */
#define REGISTER_FP16_NEON(func_name) nullptr
#define REGISTER_FP16_SVE(func_name) nullptr
#define REGISTER_FP16_SVE2(func_name) nullptr
#endif /* defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS) */

#if defined(ENABLE_FP32_KERNELS)

#if defined(ARM_COMPUTE_ENABLE_SVE)
#define REGISTER_FP32_SVE(func_name) &(func_name)
#else /* !defined(ARM_COMPUTE_ENABLE_SVE) */
#define REGISTER_FP32_SVE(func_name) nullptr
#endif /* defined(ARM_COMPUTE_ENABLE_SVE) */

#if defined(ARM_COMPUTE_ENABLE_SVE2)
#define REGISTER_FP32_SVE2(func_name) &(func_name)
#else /* !defined(ARM_COMPUTE_ENABLE_SVE2) */
#define REGISTER_FP32_SVE2(func_name) nullptr
#endif /* defined(ARM_COMPUTE_ENABLE_SVE2) */

#if defined(ARM_COMPUTE_ENABLE_NEON)
#define REGISTER_FP32_NEON(func_name) &(func_name)
#else /* !defined(ARM_COMPUTE_ENABLE_NEON) */
#define REGISTER_FP32_NEON(func_name) nullptr
#endif /* defined(ARM_COMPUTE_ENABLE_NEON) */

#else /* defined(ENABLE_FP32_KERNELS) */
#define REGISTER_FP32_NEON(func_name) nullptr
#define REGISTER_FP32_SVE(func_name) nullptr
#define REGISTER_FP32_SVE2(func_name) nullptr
#endif /* defined(ENABLE_FP32_KERNELS) */

#if defined(ENABLE_QASYMM8_SIGNED_KERNELS)

#define REGISTER_QASYMM8_SIGNED_NEON(func_name) &(func_name)

#if defined(ARM_COMPUTE_ENABLE_SVE)
#define REGISTER_QASYMM8_SIGNED_SVE(func_name) &(func_name)
#else /* !defined(ARM_COMPUTE_ENABLE_SVE) */
#define REGISTER_QASYMM8_SIGNED_SVE(func_name) nullptr
#endif /* defined(ARM_COMPUTE_ENABLE_SVE) */

#if defined(ARM_COMPUTE_ENABLE_SVE2)
#define REGISTER_QASYMM8_SIGNED_SVE2(func_name) &(func_name)
#else /* !defined(ARM_COMPUTE_ENABLE_SVE2) */
#define REGISTER_QASYMM8_SIGNED_SVE2(func_name) nullptr
#endif /* defined(ARM_COMPUTE_ENABLE_SVE2) */

#else /* defined(ENABLE_QASYMM8_SIGNED_KERNELS) */
#define REGISTER_QASYMM8_SIGNED_NEON(func_name) nullptr
#define REGISTER_QASYMM8_SIGNED_SVE(func_name) nullptr
#define REGISTER_QASYMM8_SIGNED_SVE2(func_name) nullptr
#endif /* defined(ENABLE_QASYMM8_SIGNED_KERNELS) */

#if defined(ENABLE_QASYMM8_KERNELS)
#define REGISTER_QASYMM8_NEON(func_name) &(func_name)

#if defined(ARM_COMPUTE_ENABLE_SVE)
#define REGISTER_QASYMM8_SVE(func_name) &(func_name)
#else /* !defined(ARM_COMPUTE_ENABLE_SVE) */
#define REGISTER_QASYMM8_SVE(func_name) nullptr
#endif /* defined(ARM_COMPUTE_ENABLE_SVE) */

#if defined(ARM_COMPUTE_ENABLE_SVE2)
#define REGISTER_QASYMM8_SVE2(func_name) &(func_name)
#else /* !defined(ARM_COMPUTE_ENABLE_SVE2) */
#define REGISTER_QASYMM8_SVE2(func_name) nullptr
#endif /* defined(ARM_COMPUTE_ENABLE_SVE2) */

#else /* defined(ENABLE_QASYMM8_KERNELS) */
#define REGISTER_QASYMM8_NEON(func_name) nullptr
#define REGISTER_QASYMM8_SVE(func_name) nullptr
#define REGISTER_QASYMM8_SVE2(func_name) nullptr
#endif /* defined(ENABLE_QASYMM8_KERNELS) */

#if defined(ENABLE_QSYMM16_KERNELS)

#define REGISTER_QSYMM16_NEON(func_name) &(func_name)

#if defined(ARM_COMPUTE_ENABLE_SVE)
#define REGISTER_QSYMM16_SVE(func_name) &(func_name)
#else /* !defined(ARM_COMPUTE_ENABLE_SVE) */
#define REGISTER_QSYMM16_SVE(func_name) nullptr
#endif /* defined(ARM_COMPUTE_ENABLE_SVE) */

#if defined(ARM_COMPUTE_ENABLE_SVE2)
#define REGISTER_QSYMM16_SVE2(func_name) &(func_name)
#else /* !defined(ARM_COMPUTE_ENABLE_SVE2) */
#define REGISTER_QSYMM16_SVE2(func_name) nullptr
#endif /* defined(ARM_COMPUTE_ENABLE_SVE2) */

#else /* defined(ENABLE_QSYMM16_KERNELS) */
#define REGISTER_QSYMM16_NEON(func_name) nullptr
#define REGISTER_QSYMM16_SVE(func_name) nullptr
#define REGISTER_QSYMM16_SVE2(func_name) nullptr
#endif /* defined(ENABLE_QSYMM16_KERNELS) */

#if defined(ENABLE_QASYMM8_KERNELS) || defined(ENABLE_QASYMM8_SIGNED_KERNELS)
#define REGISTER_Q8_NEON(func_name) &(func_name)
#else /* !defined(ENABLE_QASYMM8_KERNELS) && !defined(ENABLE_QASYMM8_SIGNED_KERNELS) */
#define REGISTER_Q8_NEON(func_name) nullptr
#endif /* defined(ENABLE_QASYMM8_KERNELS) || defined(ENABLE_QASYMM8_SIGNED_KERNELS) */

#if defined(ENABLE_INTEGER_KERNELS)

#if defined(ARM_COMPUTE_ENABLE_SVE)
#define REGISTER_INTEGER_SVE(func_name) &(func_name)
#else /* !defined(ARM_COMPUTE_ENABLE_SVE) */
#define REGISTER_INTEGER_SVE(func_name) nullptr
#endif /* defined(ARM_COMPUTE_ENABLE_SVE) */

#if defined(ARM_COMPUTE_ENABLE_SVE2)
#define REGISTER_INTEGER_SVE2(func_name) &(func_name)
#else /* !defined(ARM_COMPUTE_ENABLE_SVE2) */
#define REGISTER_INTEGER_SVE2(func_name) nullptr
#endif /* defined(ARM_COMPUTE_ENABLE_SVE2) */

#if defined(ARM_COMPUTE_ENABLE_NEON)
#define REGISTER_INTEGER_NEON(func_name) &(func_name)
#else /* !defined(ARM_COMPUTE_ENABLE_NEON) */
#define REGISTER_INTEGER_NEON(func_name) nullptr
#endif /* defined(ARM_COMPUTE_ENABLE_NEON) */

#else /* defined(ENABLE_INTEGER_KERNELS) */
#define REGISTER_INTEGER_NEON(func_name) nullptr
#define REGISTER_INTEGER_SVE(func_name) nullptr
#define REGISTER_INTEGER_SVE2(func_name) nullptr
#endif /* defined(ENABLE_INTEGER_KERNELS) */

#if defined(ARM_COMPUTE_ENABLE_BF16)
#define REGISTER_BF16_NEON(func_name) &(func_name)
#else /* !(defined(ARM_COMPUTE_ENABLE_BF16))*/
#define REGISTER_BF16_NEON(func_name) nullptr
#endif /* defined(ARM_COMPUTE_ENABLE_BF16)*/

#endif /* SRC_CORE_COMMON_REGISTRARS_H */
