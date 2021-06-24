/*
 * Copyright (c) 2021 Arm Limited.
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
#include "src/common/cpuinfo/CpuIsaInfo.h"

#include "arm_compute/core/Error.h"
#include "src/common/cpuinfo/CpuModel.h"

/* Arm Feature flags */
#define ARM_COMPUTE_CPU_FEATURE_HWCAP_HALF (1 << 1)
#define ARM_COMPUTE_CPU_FEATURE_HWCAP_NEON (1 << 12)

/* Arm64 Feature flags */
#define ARM_COMPUTE_CPU_FEATURE_HWCAP_ASIMD (1 << 1)
#define ARM_COMPUTE_CPU_FEATURE_HWCAP_FPHP (1 << 9)
#define ARM_COMPUTE_CPU_FEATURE_HWCAP_ASIMDHP (1 << 10)
#define ARM_COMPUTE_CPU_FEATURE_HWCAP_ASIMDDP (1 << 20)
#define ARM_COMPUTE_CPU_FEATURE_HWCAP_SVE (1 << 22)
#define ARM_COMPUTE_CPU_FEATURE_HWCAP2_SVE2 (1 << 1)
#define ARM_COMPUTE_CPU_FEATURE_HWCAP2_SVEI8MM (1 << 9)
#define ARM_COMPUTE_CPU_FEATURE_HWCAP2_SVEF32MM (1 << 10)
#define ARM_COMPUTE_CPU_FEATURE_HWCAP2_SVEBF16 (1 << 12)
#define ARM_COMPUTE_CPU_FEATURE_HWCAP2_I8MM (1 << 13)
#define ARM_COMPUTE_CPU_FEATURE_HWCAP2_BF16 (1 << 14)

namespace arm_compute
{
namespace cpuinfo
{
namespace
{
inline bool is_feature_supported(uint64_t features, uint64_t feature_mask)
{
    return (features & feature_mask);
}

#if defined(__arm__)
void decode_hwcaps(CpuIsaInfo &isa, const uint32_t hwcaps, const uint32_t hwcaps2)
{
    ARM_COMPUTE_UNUSED(hwcaps2);
    isa.fp16 = is_feature_supported(hwcaps, ARM_COMPUTE_CPU_FEATURE_HWCAP_HALF);
    isa.neon = is_feature_supported(hwcaps, ARM_COMPUTE_CPU_FEATURE_HWCAP_NEON);
}
#elif defined(__aarch64__)
void decode_hwcaps(CpuIsaInfo &isa, const uint32_t hwcaps, const uint32_t hwcaps2)
{
    // High-level SIMD support
    isa.neon = is_feature_supported(hwcaps, ARM_COMPUTE_CPU_FEATURE_HWCAP_ASIMD);
    isa.sve  = is_feature_supported(hwcaps, ARM_COMPUTE_CPU_FEATURE_HWCAP_SVE);
    isa.sve2 = is_feature_supported(hwcaps2, ARM_COMPUTE_CPU_FEATURE_HWCAP2_SVE2);

    // Data-type support
    isa.fp16    = is_feature_supported(hwcaps, ARM_COMPUTE_CPU_FEATURE_HWCAP_FPHP | ARM_COMPUTE_CPU_FEATURE_HWCAP_ASIMDHP);
    isa.bf16    = is_feature_supported(hwcaps2, ARM_COMPUTE_CPU_FEATURE_HWCAP2_BF16);
    isa.svebf16 = is_feature_supported(hwcaps2, ARM_COMPUTE_CPU_FEATURE_HWCAP2_SVEBF16);

    // Instruction extensions
    isa.dot      = is_feature_supported(hwcaps, ARM_COMPUTE_CPU_FEATURE_HWCAP_ASIMDDP);
    isa.i8mm     = is_feature_supported(hwcaps2, ARM_COMPUTE_CPU_FEATURE_HWCAP2_I8MM);
    isa.svei8mm  = is_feature_supported(hwcaps2, ARM_COMPUTE_CPU_FEATURE_HWCAP2_SVEI8MM);
    isa.svef32mm = is_feature_supported(hwcaps2, ARM_COMPUTE_CPU_FEATURE_HWCAP2_SVEF32MM);
}
#else  /* defined(__aarch64__) */
void decode_hwcaps(CpuIsaInfo &isa, const uint32_t hwcaps, const uint32_t hwcaps2)
{
    ARM_COMPUTE_UNUSED(isa, hwcaps, hwcaps2);
}
#endif /* defined(__aarch64__) */

void decode_regs(CpuIsaInfo &isa, const uint64_t isar0, const uint64_t isar1, const uint64_t pfr0, const uint64_t svefr0)
{
    auto is_supported = [](uint64_t feature_reg, uint8_t feature_pos) -> bool
    {
        return ((feature_reg >> feature_pos) & 0xf);
    };

    // High-level SIMD support
    isa.sve  = is_supported(pfr0, 32);
    isa.sve2 = is_supported(svefr0, 0);

    // Data-type support
    isa.fp16    = is_supported(pfr0, 16);
    isa.bf16    = is_supported(isar1, 44);
    isa.svebf16 = is_supported(svefr0, 20);

    // Instruction extensions
    isa.dot      = is_supported(isar0, 44);
    isa.i8mm     = is_supported(isar1, 48);
    isa.svei8mm  = is_supported(svefr0, 44);
    isa.svef32mm = is_supported(svefr0, 52);
}

/** Handle features from whitelisted models in case of problematic kernels
 *
 * @param[in, out] isa   ISA to update
 * @param[in]      model CPU model type
 */
void whitelisted_model_features(CpuIsaInfo &isa, CpuModel model)
{
    if(isa.dot == false)
    {
        isa.dot = model_supports_dot(model);
    }
    if(isa.fp16 == false)
    {
        isa.fp16 = model_supports_fp16(model);
    }
    if(isa.sve == false)
    {
        isa.sve = model_supports_sve(model);
    }
}
} // namespace

CpuIsaInfo init_cpu_isa_from_hwcaps(uint32_t hwcaps, uint32_t hwcaps2, uint32_t midr)
{
    CpuIsaInfo isa;

    decode_hwcaps(isa, hwcaps, hwcaps2);

    const CpuModel model = midr_to_model(midr);
    whitelisted_model_features(isa, model);

    return isa;
}

CpuIsaInfo init_cpu_isa_from_regs(uint64_t isar0, uint64_t isar1, uint64_t pfr0, uint64_t svefr0, uint64_t midr)
{
    CpuIsaInfo isa;

    decode_regs(isa, isar0, isar1, pfr0, svefr0);

    const CpuModel model = midr_to_model(midr);
    whitelisted_model_features(isa, model);

    return isa;
}
} // namespace cpuinfo
} // namespace arm_compute