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
#ifndef __ARM_COMPUTE_CPP_TYPES_H__
#define __ARM_COMPUTE_CPP_TYPES_H__

#include "arm_compute/core/Error.h"

#include <string>
#include <vector>

namespace arm_compute
{
/** CPU models - we only need to detect CPUs we have
 * microarchitecture-specific code for.
 *
 * Architecture features are detected via HWCAPs.
 */
enum class CPUModel
{
    GENERIC,
    GENERIC_FP16,
    GENERIC_FP16_DOT,
    A53,
    A55r0,
    A55r1
};

/** Convert a cpumodel value to a string
 *
 * @param val CPUModel value to be converted
 *
 * @return String representing the corresponding CPUModel.
 */
inline std::string cpu_model_to_string(CPUModel val)
{
    switch(val)
    {
        case CPUModel::GENERIC:
        {
            return std::string("GENERIC");
        }
        case CPUModel::GENERIC_FP16:
        {
            return std::string("GENERIC_FP16");
        }
        case CPUModel::GENERIC_FP16_DOT:
        {
            return std::string("GENERIC_FP16_DOT");
        }
        case CPUModel::A53:
        {
            return std::string("A53");
        }
        case CPUModel::A55r0:
        {
            return std::string("A55r0");
        }
        case CPUModel::A55r1:
        {
            return std::string("A55r1");
        }
        default:
        {
            ARM_COMPUTE_ERROR("Invalid CPUModel.");
            return std::string("GENERIC");
        }
    }
}

class CPUInfo final
{
public:
    /** Constructor */
    CPUInfo();

    /** Disable copy constructor and assignment operator to avoid copying the vector of CPUs each time
     *  CPUInfo is initialized once in the IScheduler and ThreadInfo will get a pointer to it.
     */
    CPUInfo &operator=(const CPUInfo &cpuinfo) = delete;
    CPUInfo(const CPUInfo &cpuinfo)            = delete;
    CPUInfo &operator=(const CPUInfo &&cpuinfo) = delete;
    CPUInfo(const CPUInfo &&cpuinfo)            = delete;

    /** Checks if the cpu model supports fp16.
     *
     * @return true of the cpu supports fp16, false otherwise
     */
    bool has_fp16() const;
    /** Checks if the cpu model supports dot product.
     *
     * @return true of the cpu supports dot product, false otherwise
     */
    bool has_dotprod() const;
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
    /** Set the L1 cache size
     *
     * @param[in] size the new size to be set.
     */
    void set_L1_cache_size(unsigned int size);
    /** Set the L2 cache size
     *
     * @param[in] size the new size to be set.
     */
    void set_L2_cache_size(unsigned int size);
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
    /** Set the cpumodel for a given cpu core
     *
     * @param[in] cpuid the id of the core to be set.
     * @param[in] model the @ref CPUModel to be set.
     */
    void set_cpu_model(unsigned int cpuid, CPUModel model);
    /** Set max number of cpus
     *
     * @param[in] cpu_count the number of CPUs in the system.
     */
    void set_cpu_num(unsigned int cpu_count);

    /** Return the maximum number of CPUs present
     *
     * @return Number of CPUs
     */
    unsigned int get_cpu_num() const;

private:
    std::vector<CPUModel> _percpu        = {};
    bool                  _fp16          = false;
    bool                  _dotprod       = false;
    unsigned int          _L1_cache_size = 32768;
    unsigned int          _L2_cache_size = 262144;
};

/** Information about executing thread and CPU. */
struct ThreadInfo
{
    int            thread_id{ 0 };
    int            num_threads{ 1 };
    const CPUInfo *cpu_info{ nullptr };
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CPP_TYPES_H__ */
