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
#ifndef SRC_COMMON_CPUINFO_H
#define SRC_COMMON_CPUINFO_H

#include "src/common/cpuinfo/CpuIsaInfo.h"
#include "src/common/cpuinfo/CpuModel.h"

#include <string>
#include <vector>

namespace arm_compute
{
namespace cpuinfo
{
/** Aggregate class that contains CPU related information
 *
 * Contains information about the numbers of the CPUs, the model of each CPU,
 * ISA related information and more
 *
 * @note We can safely assume that the ISA is common between different clusters of cores
 */
class CpuInfo
{
public:
    /** Default constructor */
    CpuInfo() = default;
    /** Construct a new Cpu Info object
     *
     * @param[in] isa  ISA capabilities information
     * @param[in] cpus CPU models information
     */
    CpuInfo(CpuIsaInfo isa, std::vector<CpuModel> cpus);
    /** CpuInfo builder function from system related information
     *
     * @return CpuInfo A populated CpuInfo structure
     */
    static CpuInfo build();

public:
    bool has_neon() const
    {
        return _isa.neon;
    }
    bool has_sve() const
    {
        return _isa.sve;
    }
    bool has_sve2() const
    {
        return _isa.sve2;
    }
    bool has_sme() const
    {
        return _isa.sme;
    }
    bool has_sme2() const
    {
        return _isa.sme2;
    }
    bool has_fp16() const
    {
        return _isa.fp16;
    }
    bool has_bf16() const
    {
        return _isa.bf16;
    }
    bool has_svebf16() const
    {
        return _isa.svebf16;
    }
    bool has_dotprod() const
    {
        return _isa.dot;
    }
    bool has_i8mm() const
    {
        return _isa.i8mm;
    }
    bool has_svei8mm() const
    {
        return _isa.svei8mm;
    }
    bool has_svef32mm() const
    {
        return _isa.svef32mm;
    }

    const CpuIsaInfo &isa() const
    {
        return _isa;
    }
    const std::vector<CpuModel> &cpus() const
    {
        return _cpus;
    }

    CpuModel cpu_model(uint32_t cpuid) const;
    CpuModel cpu_model() const;
    uint32_t num_cpus() const;

private:
    CpuIsaInfo            _isa{};
    std::vector<CpuModel> _cpus{};
};

/** Some systems have both big and small cores, this fuction computes the minimum number of cores
 *  that are exactly the same on the system. To maximize performance the library attempts to process
 *  workloads concurrently using as many threads as big cores are available on the system.
 *
 * @return The minumum number of common cores.
 */
uint32_t num_threads_hint();
} // namespace cpuinfo
} // namespace arm_compute
#endif /* SRC_COMMON_CPUINFO_H */
