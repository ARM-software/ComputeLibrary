/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CPP_TYPES_H
#define ARM_COMPUTE_CPP_TYPES_H

#include "arm_compute/core/Error.h"

#include <memory>

namespace arm_compute
{
#define ARM_COMPUTE_CPU_MODEL_LIST \
    X(GENERIC)                     \
    X(GENERIC_FP16)                \
    X(GENERIC_FP16_DOT)            \
    X(A35)                         \
    X(A53)                         \
    X(A55r0)                       \
    X(A55r1)                       \
    X(A73)                         \
    X(KLEIN)                       \
    X(X1)

/** CPU models types
 *
 * @note We only need to detect CPUs we have microarchitecture-specific code for.
 * @note Architecture features are detected via HWCAPs.
 */
enum class CPUModel
{
#define X(model) model,
    ARM_COMPUTE_CPU_MODEL_LIST
#undef X
};

class CPUInfo final
{
public:
    /** Constructor */
    CPUInfo();
    ~CPUInfo();

    /** Disable copy constructor and assignment operator to avoid copying the vector of CPUs each time
     *  CPUInfo is initialized once in the IScheduler and ThreadInfo will get a pointer to it.
     */
    CPUInfo &operator=(const CPUInfo &cpuinfo) = delete;
    CPUInfo(const CPUInfo &cpuinfo)            = delete;
    CPUInfo &operator=(CPUInfo &&cpuinfo) = default;
    CPUInfo(CPUInfo &&cpuinfo)            = default;

    /** Checks if the cpu model supports fp16.
     *
     * @return true of the cpu supports fp16, false otherwise
     */
    bool has_fp16() const;
    /** Checks if the cpu model supports bf16.
     *
     * @return true of the cpu supports bf16, false otherwise
     */
    bool has_bf16() const;
    /** Checks if the cpu model supports dot product.
     *
     * @return true of the cpu supports dot product, false otherwise
     */
    bool has_dotprod() const;
    /** Checks if the cpu model supports sve.
     *
     * @return true of the cpu supports sve, false otherwise
     */
    bool has_sve() const;
    /** Gets the cpu model for a given cpuid.
     *
     * @param[in] cpuid the id of the cpu core to be retrieved,
     *
     * @return the @ref CPUModel of the cpuid queiried.
     */
    CPUModel get_cpu_model(unsigned int cpuid) const;
    /** Gets the current thread's cpu model
     *
     * @return Current thread's @ref CPUModel
     */
    CPUModel get_cpu_model() const;
    /** Gets the L1 cache size
     *
     * @return the size of the L1 cache
     */
    unsigned int get_L1_cache_size() const;
    /** Gets the L2 cache size
     *
     * @return the size of the L1 cache
     */
    unsigned int get_L2_cache_size() const;
    /** Set fp16 support
     *
     * @param[in] fp16 whether the cpu supports fp16.
     */
    void set_fp16(const bool fp16);
    /** Set dot product support
     *
     * @param[in] dotprod whether the cpu supports dot product.
     */
    void set_dotprod(const bool dotprod);

    /** Return the maximum number of CPUs present
     *
     * @return Number of CPUs
     */
    unsigned int get_cpu_num() const;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

/** Information about executing thread and CPU. */
struct ThreadInfo
{
    int            thread_id{ 0 };
    int            num_threads{ 1 };
    const CPUInfo *cpu_info{ nullptr };
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPP_TYPES_H */
