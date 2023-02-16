/*
 * Copyright (c) 2017-2022 Arm Limited.
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
namespace cpuinfo
{
struct CpuIsaInfo;
} // namespace cpuinfo

#define ARM_COMPUTE_CPU_MODEL_LIST \
    X(GENERIC)                     \
    X(GENERIC_FP16)                \
    X(GENERIC_FP16_DOT)            \
    X(A53)                         \
    X(A55r0)                       \
    X(A55r1)                       \
    X(A35)                         \
    X(A73)                         \
    X(A76)                         \
    X(A510)                        \
    X(X1)                          \
    X(V1)                          \
    X(A64FX)                       \
    X(N1)

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
protected:
    CPUInfo();
    ~CPUInfo();

public:
    /** Access the KernelLibrary singleton.
     * This method has been deprecated and will be removed in future releases
     * @return The KernelLibrary instance.
     */
    static CPUInfo &get();

    /* Delete move and copy constructors and assignment operator
    s */
    CPUInfo(CPUInfo const &) = delete;            // Copy construct
    CPUInfo(CPUInfo &&)      = delete;            // Move construct
    CPUInfo &operator=(CPUInfo const &) = delete; // Copy assign
    CPUInfo &operator=(CPUInfo &&) = delete;      // Move assign

    /** Checks if the cpu model supports fp16.
     *
     * @return true if the cpu supports fp16, false otherwise
     */
    bool has_fp16() const;
    /** Checks if the cpu model supports bf16.
     *
     * @return true if the cpu supports bf16, false otherwise
     */
    bool has_bf16() const;
    /** Checks if the cpu model supports bf16.
     *
     * @return true if the cpu supports bf16, false otherwise
     */
    bool has_svebf16() const;
    /** Checks if the cpu model supports dot product.
     *
     * @return true if the cpu supports dot product, false otherwise
     */
    bool has_dotprod() const;
    /** Checks if the cpu model supports floating-point matrix multiplication.
     *
     * @return true if the cpu supports floating-point matrix multiplication, false otherwise
     */
    bool has_svef32mm() const;
    /** Checks if the cpu model supports integer matrix multiplication.
     *
     * @return true if the cpu supports integer matrix multiplication, false otherwise
     */
    bool has_i8mm() const;
    /** Checks if the cpu model supports integer matrix multiplication.
     *
     * @return true if the cpu supports integer matrix multiplication, false otherwise
     */
    bool has_svei8mm() const;
    /** Checks if the cpu model supports sve.
     *
     * @return true if the cpu supports sve, false otherwise
     */
    bool has_sve() const;
    /** Checks if the cpu model supports sve2.
     *
     * @return true if the cpu supports sve2, false otherwise
     */
    bool has_sve2() const;
    /** Checks if the cpu model supports sme.
     *
     * @return true if the cpu supports sme, false otherwise
     */
    bool has_sme() const;
    /** Checks if the cpu model supports sme2.
     *
     * @return true if the cpu supports sme2, false otherwise
     */
    bool has_sme2() const;
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
    /** Gets the current cpu's ISA information
     *
     * @return Current cpu's ISA information
     */
    cpuinfo::CpuIsaInfo get_isa() const;
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
