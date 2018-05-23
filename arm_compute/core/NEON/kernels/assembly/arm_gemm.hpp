/*
 * Copyright (c) 2018 ARM Limited.
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
#pragma once

#include <memory>

#include "arm_gemm_local.hpp"
#include "gemm_common.hpp"

namespace arm_gemm {

template<typename Top, typename Tret>
using UniqueGemmCommon = std::unique_ptr<GemmCommon<Top, Tret> >;

template<typename Top, typename Tret>
UniqueGemmCommon<Top, Tret> gemm(const CPUInfo &ci,
                                 const unsigned int M, const unsigned int N, const unsigned int K,
                                 const unsigned int nbatches, const unsigned int nmulti,
                                 const bool trA, const bool trB, const Tret alpha, const Tret beta,
                                 const int maxthreads, const bool pretransposed_hint);
} // namespace arm_gemm
