/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#include "kernels/a64_gemm_s8_12x8.hpp"
#include "kernels/a64_gemm_s8_4x4.hpp"

namespace arm_gemm {

class GemmImpl_gemm_s8_interleaved_dot : public GemmImplementation<int8_t, int32_t> {
public:
    bool is_supported(const GemmArgs<int32_t> &args) override {
        return args._ci->has_dotprod();
    }

    UniqueGemmCommon<int8_t, int32_t> instantiate(const GemmArgs<int32_t> &args) override {
        return UniqueGemmCommon<int8_t, int32_t>(new GemmInterleaved<gemm_s8_12x8, int8_t, int32_t>(args));
    }

    GemmImpl_gemm_s8_interleaved_dot() : GemmImplementation<int8_t, int32_t>(GemmMethod::GEMM_INTERLEAVED_DOT) { }
};

class GemmImpl_gemm_s8_interleaved : public GemmImplementation<int8_t, int32_t> {
public:
    UniqueGemmCommon<int8_t, int32_t> instantiate(const GemmArgs<int32_t> &args) override {
        return UniqueGemmCommon<int8_t, int32_t>(new GemmInterleaved<gemm_s8_4x4, int8_t, int32_t>(args));
    }

    GemmImpl_gemm_s8_interleaved() : GemmImplementation<int8_t, int32_t>(GemmMethod::GEMM_INTERLEAVED) { }
};

static std::vector<GemmImplementation<int8_t, int32_t> *> gemm_s8_methods = {
    new GemmImpl_gemm_s8_interleaved_dot(),
    new GemmImpl_gemm_s8_interleaved()
};

template<>
std::vector<GemmImplementation<int8_t, int32_t> *> &gemm_implementation_list<int8_t, int32_t>() {
    return gemm_s8_methods;
}

/* Explicitly instantiate the external functions for these types. */
template UniqueGemmCommon<int8_t, int32_t> gemm<int8_t, int32_t>(GemmArgs<int32_t> &args, GemmConfig *cfg);
template GemmMethod get_gemm_method<int8_t, int32_t>(GemmArgs<int32_t> &args);
template bool method_is_compatible<int8_t, int32_t>(GemmMethod method, GemmArgs<int32_t> &args);

} // namespace arm_gemm

#endif // __aarch64__
