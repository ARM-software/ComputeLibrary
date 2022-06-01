/*
 * Copyright (c) 2022 Arm Limited.
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

#include "asmlib.hpp"
#include "convolution_parameters.hpp"
#include "convolver.hpp"
#include "interleave_indirect.hpp"
#include "bfloat.hpp"

#include <alloca.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <tuple>
#include <type_traits>
#include <vector>

#include <arm_neon.h>

#include "utils.hpp"

#ifdef ARM_COMPUTE_ENABLE_SVE
#ifdef ARM_COMPUTE_ENABLE_SME
namespace arm_gemm {

#include "interleave_indirect_impl.hpp"

#include "indirect-interleaves/list-sve.hpp"

/**** Instantiate needed implementations ****/

/* FP32: SME implementations (height 1VL, 2VL, 4VL) */
template void IndirectInterleave<2, 1, VLType::SME>(float *, const float * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<2, 1, VLType::SME>(float *, const float *, size_t, const convolver<float> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<2, 1, VLType::SME>(float *, const float *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<1, 1, VLType::SME>(float *, const float * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<1, 1, VLType::SME>(float *, const float *, size_t, const convolver<float> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<1, 1, VLType::SME>(float *, const float *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<4, 1, VLType::SME>(float *, const float * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<4, 1, VLType::SME>(float *, const float *, size_t, const convolver<float> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<4, 1, VLType::SME>(float *, const float *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

/* BF16: SME implementations (height 1VL, 2VL, 4VL) */
template void IndirectInterleave<2, 2, VLType::SME>(bfloat16 *, const bfloat16 * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<2, 2, VLType::SME>(bfloat16 *, const bfloat16 *, size_t, const convolver<bfloat16> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<2, 2, VLType::SME>(bfloat16 *, const bfloat16 *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<1, 2, VLType::SME>(bfloat16 *, const bfloat16 * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<1, 2, VLType::SME>(bfloat16 *, const bfloat16 *, size_t, const convolver<bfloat16> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<1, 2, VLType::SME>(bfloat16 *, const bfloat16 *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<4, 2, VLType::SME>(bfloat16 *, const bfloat16 * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<4, 2, VLType::SME>(bfloat16 *, const bfloat16 *, size_t, const convolver<bfloat16> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<4, 2, VLType::SME>(bfloat16 *, const bfloat16 *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

/* BF16: SME implementations narrow accumulators (no blocking) (height 1VL, 2VL) */
template void IndirectInterleave<2, 1, VLType::SME>(bfloat16 *, const bfloat16 * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<2, 1, VLType::SME>(bfloat16 *, const bfloat16 *, size_t, const convolver<bfloat16> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<2, 1, VLType::SME>(bfloat16 *, const bfloat16 *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<1, 1, VLType::SME>(bfloat16 *, const bfloat16 * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<1, 1, VLType::SME>(bfloat16 *, const bfloat16 *, size_t, const convolver<bfloat16> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<1, 1, VLType::SME>(bfloat16 *, const bfloat16 *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

/* FP16: SME implementations narrow accumulators (no blocking) (height 1VL, 2VL) */
template void IndirectInterleave<2, 1, VLType::SME>(__fp16 *, const __fp16 * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<2, 1, VLType::SME>(__fp16 *, const __fp16 *, size_t, const convolver<__fp16> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<2, 1, VLType::SME>(__fp16 *, const __fp16 *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<1, 1, VLType::SME>(__fp16 *, const __fp16 * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<1, 1, VLType::SME>(__fp16 *, const __fp16 *, size_t, const convolver<__fp16> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<1, 1, VLType::SME>(__fp16 *, const __fp16 *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

/* FP32 fast-mode: SME implementations */
template void IndirectInterleave<1, 2, VLType::SME>(bfloat16 *, const float * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<1, 2, VLType::SME>(bfloat16 *, const float *, size_t, const convolver<float> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<1, 2, VLType::SME>(bfloat16 *, const float *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<2, 2, VLType::SME>(bfloat16 *, const float * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<2, 2, VLType::SME>(bfloat16 *, const float *, size_t, const convolver<float> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<2, 2, VLType::SME>(bfloat16 *, const float *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<4, 2, VLType::SME>(bfloat16 *, const float * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<4, 2, VLType::SME>(bfloat16 *, const float *, size_t, const convolver<float> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<4, 2, VLType::SME>(bfloat16 *, const float *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

/* INT8: SME implementation (height 1VL, 2VL, 4VL) */
template void IndirectInterleave<1, 4, VLType::SME>(int8_t *, const int8_t * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<1, 4, VLType::SME>(int8_t *, const int8_t *, size_t, const convolver<int8_t> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<1, 4, VLType::SME>(int8_t *, const int8_t *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<2, 4, VLType::SME>(int8_t *, const int8_t * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<2, 4, VLType::SME>(int8_t *, const int8_t *, size_t, const convolver<int8_t> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<2, 4, VLType::SME>(int8_t *, const int8_t *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<4, 4, VLType::SME>(int8_t *, const int8_t * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<4, 4, VLType::SME>(int8_t *, const int8_t *, size_t, const convolver<int8_t> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<4, 4, VLType::SME>(int8_t *, const int8_t *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

/* UINT8: SME implementation (height 1VL, 2VL, 4VL) */
template void IndirectInterleave<1, 4, VLType::SME>(uint8_t *, const uint8_t * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<1, 4, VLType::SME>(uint8_t *, const uint8_t *, size_t, const convolver<uint8_t> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<1, 4, VLType::SME>(uint8_t *, const uint8_t *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<2, 4, VLType::SME>(uint8_t *, const uint8_t * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<2, 4, VLType::SME>(uint8_t *, const uint8_t *, size_t, const convolver<uint8_t> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<2, 4, VLType::SME>(uint8_t *, const uint8_t *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<4, 4, VLType::SME>(uint8_t *, const uint8_t * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<4, 4, VLType::SME>(uint8_t *, const uint8_t *, size_t, const convolver<uint8_t> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<4, 4, VLType::SME>(uint8_t *, const uint8_t *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

} // namespace arm_gemm

#endif // ARM_COMPUTE_ENABLE_SME
#endif // ARM_COMPUTE_ENABLE_SVE
