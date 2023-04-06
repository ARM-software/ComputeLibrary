/*
 * Copyright (c) 2017-2021, 2023 Arm Limited.
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
#ifndef ARM_COMPUTE_ISCHEDULER_H
#define ARM_COMPUTE_ISCHEDULER_H

#include "arm_compute/core/CPP/CPPTypes.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/experimental/Types.h"

#include <functional>
#include <limits>

namespace arm_compute
{
class ICPPKernel;
class ITensor;
class Window;

/** Scheduler interface to run kernels */
class IScheduler
{
public:
    /** Strategies available to split a workload */
    enum class StrategyHint
    {
        STATIC,  /**< Split the workload evenly among the threads */
        DYNAMIC, /**< Split the workload dynamically using a bucket system */
    };

    /** Function to be used and map a given thread id to a logical core id
     *
     * Mapping function expects the thread index and total number of cores as input,
     * and returns the logical core index to bind against
     */
    using BindFunc = std::function<int(int, int)>;

    /** When arm_compute::ISchedular::Hints::_split_dimension is initialized with this value
     * then the schedular is free to break down the problem space over as many dimensions
     * as it wishes
     */
    static constexpr unsigned int split_dimensions_all = std::numeric_limits<unsigned>::max();

    /** Scheduler hints
     *
     * Collection of preferences set by the function regarding how to split a given workload
     */
    class Hints
    {
    public:
        /** Constructor
         *
         * @param[in] split_dimension Dimension along which to split the kernel's execution window.
         * @param[in] strategy        (Optional) Split strategy.
         * @param[in] threshold       (Optional) Dynamic scheduling capping threshold.
         */
        Hints(unsigned int split_dimension, StrategyHint strategy = StrategyHint::STATIC, int threshold = 0)
            : _split_dimension(split_dimension), _strategy(strategy), _threshold(threshold)
        {
        }
        /** Set the split_dimension hint
         *
         * @param[in] split_dimension Dimension along which to split the kernel's execution window.
         *
         * @return the Hints object
         */
        Hints &set_split_dimension(unsigned int split_dimension)
        {
            _split_dimension = split_dimension;
            return *this;
        }
        /** Return the prefered split dimension
         *
         * @return The split dimension
         */
        unsigned int split_dimension() const
        {
            return _split_dimension;
        }

        /** Set the strategy hint
         *
         * @param[in] strategy Prefered strategy to use to split the workload
         *
         * @return the Hints object
         */
        Hints &set_strategy(StrategyHint strategy)
        {
            _strategy = strategy;
            return *this;
        }
        /** Return the prefered strategy to use to split workload.
         *
         * @return The strategy
         */
        StrategyHint strategy() const
        {
            return _strategy;
        }
        /** Return the granule capping threshold to be used by dynamic scheduling.
         *
         * @return The capping threshold
         */
        int threshold() const
        {
            return _threshold;
        }

    private:
        unsigned int _split_dimension{};
        StrategyHint _strategy{};
        int          _threshold{};
    };
    /** Signature for the workloads to execute */
    using Workload = std::function<void(const ThreadInfo &)>;
    /** Default constructor. */
    IScheduler();

    /** Destructor. */
    virtual ~IScheduler() = default;

    /** Sets the number of threads the scheduler will use to run the kernels.
     *
     * @param[in] num_threads If set to 0, then one thread per CPU core available on the system will be used, otherwise the number of threads specified.
     */
    virtual void set_num_threads(unsigned int num_threads) = 0;

    /** Sets the number of threads the scheduler will use to run the kernels but also using a binding function to pin the threads to given logical cores
     *
     * @param[in] num_threads If set to 0, then one thread per CPU core available on the system will be used, otherwise the number of threads specified.
     * @param[in] func        Binding function to use.
     */
    virtual void set_num_threads_with_affinity(unsigned int num_threads, BindFunc func);

    /** Returns the number of threads that the SingleThreadScheduler has in its pool.
     *
     * @return Number of threads available in SingleThreadScheduler.
     */
    virtual unsigned int num_threads() const = 0;

    /** Runs the kernel in the same thread as the caller synchronously.
     *
     * @param[in] kernel Kernel to execute.
     * @param[in] hints  Hints for the scheduler.
     */
    virtual void schedule(ICPPKernel *kernel, const Hints &hints) = 0;

    /** Runs the kernel in the same thread as the caller synchronously.
     *
     * @param[in] kernel  Kernel to execute.
     * @param[in] hints   Hints for the scheduler.
     * @param[in] window  Window to use for kernel execution.
     * @param[in] tensors Vector containing the tensors to operate on.
     */
    virtual void schedule_op(ICPPKernel *kernel, const Hints &hints, const Window &window, ITensorPack &tensors) = 0;

    /** Execute all the passed workloads
     *
     * @note There is no guarantee regarding the order in which the workloads will be executed or whether or not they will be executed in parallel.
     *
     * @param[in] workloads List of workloads to run
     * @param[in] tag       String that can be used by profiling tools to identify the workloads run by the scheduler (Can be null).
     */
    virtual void run_tagged_workloads(std::vector<Workload> &workloads, const char *tag);

    /** Get CPU info.
     *
     * @return CPU info.
     */
    CPUInfo &cpu_info();
    /** Get a hint for the best possible number of execution threads
     *
     * @warning In case we can't work out the best number of threads,
     *          std::thread::hardware_concurrency() is returned else 1 in case of bare metal builds
     *
     * @return Best possible number of execution threads to use
     */
    unsigned int num_threads_hint() const;

protected:
    /** Execute all the passed workloads
     *
     * @note there is no guarantee regarding the order in which the workloads will be executed or whether or not they will be executed in parallel.
     *
     * @param[in] workloads Array of workloads to run
     */
    virtual void run_workloads(std::vector<Workload> &workloads) = 0;

    /** Common scheduler logic to execute the given kernel
     *
     * @param[in] kernel  Kernel to execute.
     * @param[in] hints   Hints for the scheduler.
     * @param[in] window  Window to use for kernel execution.
     * @param[in] tensors Vector containing the tensors to operate on.
     */
    void schedule_common(ICPPKernel *kernel, const Hints &hints, const Window &window, ITensorPack &tensors);

    /** Adjust the number of windows to the optimize performance
     * (used for small workloads where smaller number of threads might improve the performance)
     *
     * @param[in] window           Window to use for kernel execution
     * @param[in] split_dimension  Axis of dimension to split
     * @param[in] init_num_windows Initial number of sub-windows to split
     * @param[in] kernel           Kernel to execute
     * @param[in] cpu_info         The CPU platform used to create the context.
     *
     * @return Adjusted number of windows
     */
    std::size_t adjust_num_of_windows(const Window &window, std::size_t split_dimension, std::size_t init_num_windows, const ICPPKernel &kernel, const CPUInfo &cpu_info);

private:
    unsigned int _num_threads_hint = {};
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_ISCHEDULER_H */
