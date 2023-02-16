/*
 * Copyright (c) 2021-2022 Arm Limited.
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
#ifndef SRC_COMMON_CPUINFO_CPUISAINFO_H
#define SRC_COMMON_CPUINFO_CPUISAINFO_H

#include <cstdint>

namespace arm_compute
{
namespace cpuinfo
{
/** CPU ISA (Instruction Set Architecture) information
 *
 * Contains ISA related information around the Arm architecture
 */
struct CpuIsaInfo
{
    /* SIMD extension support */
    bool neon{ false };
    bool sve{ false };
    bool sve2{ false };
    bool sme{ false };
    bool sme2{ false };

    /* Data-type extensions support */
    bool fp16{ false };
    bool bf16{ false };
    bool svebf16{ false };

    /* Instruction support */
    bool dot{ false };
    bool i8mm{ false };
    bool svei8mm{ false };
    bool svef32mm{ false };
};

/** Identify ISA related information through system information
 *
 * @param[in] hwcaps  HWCAPS feature information
 * @param[in] hwcaps2 HWCAPS2 feature information
 * @param[in] midr    MIDR value
 *
 * @return CpuIsaInfo A populated ISA feature structure
 */
CpuIsaInfo init_cpu_isa_from_hwcaps(uint32_t hwcaps, uint32_t hwcaps2, uint32_t midr);

/** Identify ISA related information through register information
 *
 * @param[in] isar0  Value of Instruction Set Attribute Register 0 (ID_AA64ISAR0_EL1)
 * @param[in] isar1  Value of Instruction Set Attribute Register 1 (ID_AA64ISAR1_EL1)
 * @param[in] pfr0   Value of Processor Feature Register 0 (ID_AA64PFR0_EL1)
 * @param[in] pfr1   Value of Processor Feature Register 1 (ID_AA64PFR1_EL1)
 * @param[in] svefr0 Value of SVE feature ID register 0 (ID_AA64ZFR0_EL1)
 * @param[in] midr   Value of Main ID Register (MIDR)
 *
 * @return CpuIsaInfo A populated ISA feature structure
 */
CpuIsaInfo init_cpu_isa_from_regs(uint64_t isar0, uint64_t isar1, uint64_t pfr0, uint64_t pfr1, uint64_t svefr0, uint64_t midr);
} // namespace cpuinfo
} // namespace arm_compute

#endif /* SRC_COMMON_CPUINFO_CPUISAINFO_H */
