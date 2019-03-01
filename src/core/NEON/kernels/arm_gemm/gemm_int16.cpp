/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#ifdef __aarch64__

#include "arm_gemm.hpp"
#include "gemm_common.hpp"
#include "gemm_implementation.hpp"
#include "gemm_interleaved.hpp"

#include "kernels/a64_gemm_s16_12x8.hpp"

namespace arm_gemm {

static const GemmImplementation<int16_t, int32_t> gemm_s16_methods[] = {
{
    GemmMethod::GEMM_INTERLEAVED,
    "gemm_s16_12x8",
    nullptr,
    nullptr,
    [](const GemmArgs<int32_t> &args) { return new GemmInterleaved<gemm_s16_12x8, int16_t, int32_t>(args); }
},
{
    GemmMethod::DEFAULT,
    "",
    nullptr,
    nullptr,
    nullptr
}
};

template<>
const GemmImplementation<int16_t, int32_t> *gemm_implementation_list<int16_t, int32_t>() {
    return gemm_s16_methods;
}

/* Explicitly instantiate the external functions for these types. */
template UniqueGemmCommon<int16_t, int32_t> gemm<int16_t, int32_t>(const GemmArgs<int32_t> &args);
template KernelDescription get_gemm_method<int16_t, int32_t>(const GemmArgs<int32_t> &args);
template bool method_is_compatible<int16_t, int32_t>(GemmMethod method, const GemmArgs<int32_t> &args);
template std::vector<KernelDescription> get_compatible_kernels<int16_t, int32_t> (const GemmArgs<int32_t> &args);

} // namespace arm_gemm

#endif // __aarch64__