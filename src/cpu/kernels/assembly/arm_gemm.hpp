/*
 * Copyright (c) 2018-2022 Arm Limited.
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

#include <cstring>
#include <memory>
#include <vector>

#include "arm_gemm_local.hpp"
#include "gemm_common.hpp"

namespace arm_gemm
{
enum class GemmMethod
{
    DEFAULT,
    GEMV_BATCHED,
    GEMV_PRETRANSPOSED,
    GEMV_NATIVE_TRANSPOSED,
    GEMM_NATIVE,
    GEMM_HYBRID,
    GEMM_INTERLEAVED,
    GEMM_INTERLEAVED_2D,
    QUANTIZE_WRAPPER,
    QUANTIZE_WRAPPER_2D,
    GEMM_HYBRID_QUANTIZED
};

enum class WeightFormat
{
    UNSPECIFIED    = 0x1,
    ANY            = 0x2,
    OHWI           = 0x100100,
    OHWIo2         = 0x100200,
    OHWIo4         = 0x100400,
    OHWIo8         = 0x100800,
    OHWIo16        = 0x101000,
    OHWIo32        = 0x102000,
    OHWIo64        = 0x104000,
    OHWIo128       = 0x108000,
    OHWIo4i2       = 0x200400,
    OHWIo4i2_bf16  = 0x200410,
    OHWIo8i2       = 0x200800,
    OHWIo8i2_bf16  = 0x200810,
    OHWIo16i2      = 0x201000,
    OHWIo16i2_bf16 = 0x201010,
    OHWIo32i2      = 0x202000,
    OHWIo32i2_bf16 = 0x202010,
    OHWIo64i2      = 0x204000,
    OHWIo64i2_bf16 = 0x204010,
    OHWIo4i4       = 0x400400,
    OHWIo4i4_bf16  = 0x400410,
    OHWIo8i4       = 0x400800,
    OHWIo8i4_bf16  = 0x400810,
    OHWIo16i4      = 0x401000,
    OHWIo16i4_bf16 = 0x401010,
    OHWIo32i4      = 0x402000,
    OHWIo32i4_bf16 = 0x402010,
    OHWIo64i4      = 0x404000,
    OHWIo64i4_bf16 = 0x404010,
    OHWIo2i8       = 0x800200,
    OHWIo4i8       = 0x800400,
    OHWIo8i8       = 0x800800,
    OHWIo16i8      = 0x801000,
    OHWIo32i8      = 0x802000,
    OHWIo64i8      = 0x804000
};

struct KernelDescription
{
    GemmMethod  method         = GemmMethod::DEFAULT;
    std::string name           = "";
    bool        is_default     = false;
    uint64_t    cycle_estimate = 0;

    KernelDescription(GemmMethod m, std::string n, bool d = false, uint64_t c = 0)
        : method(m), name(n), is_default(d), cycle_estimate(c)
    {
    }
    KernelDescription() noexcept
    {
    }
};

struct GemmConfig
{
    GemmMethod   method           = GemmMethod::DEFAULT;
    std::string  filter           = "";
    unsigned int inner_block_size = 0;
    unsigned int outer_block_size = 0;
    WeightFormat weight_format    = WeightFormat::ANY;

    GemmConfig(GemmMethod method)
        : method(method)
    {
    }
    GemmConfig()
    {
    }
};

struct Activation
{
    enum class Type
    {
        None,
        ReLU,
        BoundedReLU
    };

    Type  type;
    float param1;
    float param2;

    Activation(Type type = Type::None, float p1 = 0.0f, float p2 = 0.0f)
        : type(type), param1(p1), param2(p2)
    {
    }
};

struct GemmArgs
{
public:
    const CPUInfo    *_ci;
    unsigned int      _Msize; // num of tiles
    unsigned int      _Nsize; // output channels
    unsigned int      _Ksize; // input channels
    unsigned int      _Ksections;
    unsigned int      _nbatches;
    unsigned int      _nmulti; // n_gemms to be performed
    bool              _indirect_input;
    Activation        _act;
    int               _maxthreads;
    bool              _fixed_format;
    bool              _fast_mode;
    const GemmConfig *_cfg;

    GemmArgs(const CPUInfo *ci, unsigned int M, unsigned int N,
             unsigned int K, unsigned int Ksections, unsigned int nbatches,
             unsigned int nmulti, bool indirect_input, Activation act, const int maxthreads,
             bool fixed_format = false, bool fast_mode = false, const GemmConfig *cfg = nullptr)
        : _ci(ci), _Msize(M), _Nsize(N), _Ksize(K), _Ksections(Ksections), _nbatches(nbatches), _nmulti(nmulti), _indirect_input(indirect_input), _act(act), _maxthreads(maxthreads),
          _fixed_format(fixed_format), _fast_mode(fast_mode), _cfg(cfg)
    {
    }
};

struct Requantize32
{
public:
    const int32_t *bias                     = nullptr;
    size_t         bias_multi_stride        = 0;
    int32_t        a_offset                 = 0;
    int32_t        b_offset                 = 0;
    int32_t        c_offset                 = 0;
    bool           per_channel_requant      = false;
    int32_t        per_layer_left_shift     = 0;
    int32_t        per_layer_right_shift    = 0;
    int32_t        per_layer_mul            = 0;
    const int32_t *per_channel_left_shifts  = nullptr;
    const int32_t *per_channel_right_shifts = nullptr;
    const int32_t *per_channel_muls         = nullptr;
    int32_t        minval                   = 0;
    int32_t        maxval                   = 0;

    Requantize32() = default;

    // Constructor for per-tensor quantization
    Requantize32(const int32_t *bias, size_t bias_multi_stride,
                 int32_t a_offset, int32_t b_offset, int32_t c_offset,
                 int32_t requant_shift, int32_t requant_mul, int32_t minv, int32_t maxv)
        : bias(bias), bias_multi_stride(bias_multi_stride), a_offset(a_offset), b_offset(b_offset), c_offset(c_offset), per_channel_requant(false), per_layer_left_shift(std::max<int32_t>(requant_shift, 0)),
          per_layer_right_shift(std::min<int32_t>(requant_shift, 0)), per_layer_mul(requant_mul), minval(minv), maxval(maxv)
    {
    }

    // Constructor for per-channel quantization
    Requantize32(const int32_t *bias, size_t bias_multi_stride,
                 int32_t a_offset, int32_t b_offset, int32_t c_offset,
                 const int32_t *requant_left_shifts,
                 const int32_t *requant_right_shifts,
                 const int32_t *requant_muls,
                 int32_t minv, int32_t maxv)
        : bias(bias), bias_multi_stride(bias_multi_stride), a_offset(a_offset), b_offset(b_offset), c_offset(c_offset), per_channel_requant(true), per_channel_left_shifts(requant_left_shifts),
          per_channel_right_shifts(requant_right_shifts), per_channel_muls(requant_muls), minval(minv), maxval(maxv)
    {
    }
};

struct Nothing
{
};

template <typename Top, typename Tret>
using UniqueGemmCommon = std::unique_ptr<GemmCommon<Top, Tret>>;

/* Low level API calls.
 * These are implemented as 'GemmArgs' versions, or with the arguments explicitly listed. */

/* get_gemm_method(): Given the templated types and provided parameters,
 * which is the preferred method to implement this GEMM?  */
template <typename Top, typename Tret, class OutputStage = Nothing>
KernelDescription get_gemm_method(const GemmArgs &args, const OutputStage & = {});

template <typename Top, typename Tret, class OutputStage = Nothing>
UniqueGemmCommon<Top, Tret> gemm(const GemmArgs &args, const OutputStage & = {});

template <typename Top, typename Tret, class OutputStage = Nothing>
std::vector<KernelDescription> get_compatible_kernels(const GemmArgs &args, const OutputStage & = {});

template <typename Top, typename Tret, class OutputStage = Nothing>
bool has_opt_gemm(WeightFormat &weight_format, const GemmArgs &args, const OutputStage & = {});

} // namespace arm_gemm
