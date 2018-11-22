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

#include "kernels/a64_gemm_u16_12x8.hpp"
#include "kernels/a64_gemm_u8_12x8.hpp"
#include "kernels/a64_gemm_u8_4x4.hpp"
#include "kernels/sve_interleaved_u8u32_dot_3VLx8.hpp"

namespace arm_gemm {

#ifdef __ARM_FEATURE_SVE
class GemmImpl_gemm_u8_interleaved_dot : public GemmImplementation<uint8_t, uint32_t> {
public:
    UniqueGemmCommon<uint8_t, uint32_t> instantiate(const GemmArgs<uint32_t> &args) override {
        return UniqueGemmCommon<uint8_t, uint32_t>(new GemmInterleaved<interleaved_u8u32_dot_3VLx8, uint8_t, uint32_t>(args));
    }

    GemmImpl_gemm_u8_interleaved_dot() : GemmImplementation<uint8_t, uint32_t>(GemmMethod::GEMM_INTERLEAVED_DOT) { }
};
#else
class GemmImpl_gemm_u8_interleaved_dot : public GemmImplementation<uint8_t, uint32_t> {
public:
    bool is_supported(const GemmArgs<uint32_t> &args) override {
        return args._ci->has_dotprod();
    }

    UniqueGemmCommon<uint8_t, uint32_t> instantiate(const GemmArgs<uint32_t> &args) override {
        return UniqueGemmCommon<uint8_t, uint32_t>(new GemmInterleaved<gemm_u8_12x8, uint8_t, uint32_t>(args));
    }

    GemmImpl_gemm_u8_interleaved_dot() : GemmImplementation<uint8_t, uint32_t>(GemmMethod::GEMM_INTERLEAVED_DOT) { }
};
#endif

class GemmImpl_gemm_u8_interleaved : public GemmImplementation<uint8_t, uint32_t> {
public:
    UniqueGemmCommon<uint8_t, uint32_t> instantiate(const GemmArgs<uint32_t> &args) override {
        return UniqueGemmCommon<uint8_t, uint32_t>(new GemmInterleaved<gemm_u8_4x4, uint8_t, uint32_t>(args));
    }

    GemmImpl_gemm_u8_interleaved() : GemmImplementation<uint8_t, uint32_t>(GemmMethod::GEMM_INTERLEAVED) { }
};

static GemmImpl_gemm_u8_interleaved_dot gemm_u8_interleaved_dot_impl{};
static GemmImpl_gemm_u8_interleaved gemm_u8_interleaved_impl{};

static std::vector<GemmImplementation<uint8_t, uint32_t> *> gemm_u8_methods = {
    &gemm_u8_interleaved_dot_impl,
    &gemm_u8_interleaved_impl
};

template<>
std::vector<GemmImplementation<uint8_t, uint32_t> *> &gemm_implementation_list<uint8_t, uint32_t>() {
    return gemm_u8_methods;
}

/* Explicitly instantiate the external functions for these types. */
template UniqueGemmCommon<uint8_t, uint32_t> gemm<uint8_t, uint32_t>(GemmArgs<uint32_t> &args, GemmConfig *cfg);
template GemmMethod get_gemm_method<uint8_t, uint32_t>(GemmArgs<uint32_t> &args);
template bool method_is_compatible<uint8_t, uint32_t>(GemmMethod method, GemmArgs<uint32_t> &args);

} // namespace arm_gemm

#endif // __aarch64__
