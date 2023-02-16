/*
 * Copyright (c) 2022-2023 Arm Limited.
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
#ifndef ARM_COMPUTE_DYNAMIC_FUSION_RUNTIME_GPU_CL_CLWORKLOADRUNTIME
#define ARM_COMPUTE_DYNAMIC_FUSION_RUNTIME_GPU_CL_CLWORKLOADRUNTIME

#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/dynamic_fusion/sketch/MemoryDescriptor.h"
#include <map>
#include <memory>

namespace arm_compute
{
/** Forward declaration */
class CLTensor;
namespace experimental
{
namespace dynamic_fusion
{
/** Forward declaration */
class GpuWorkloadSketch;

/** OpenCL runtime to run a workload
 */
class ClWorkloadRuntime
{
public:
    ClWorkloadRuntime();
    ~ClWorkloadRuntime();
    /** Configure @ref ClWorkloadRuntime
     * @note A runtime cannot be re-configured
     *
     * @param[in] sketch @ref GpuWorkloadSketch with which to configure
     */
    Status configure(const GpuWorkloadSketch &sketch);
    /** Perform run workload
     * @note If the runtime is not configured, this method will not perform any action
     *
     * @param[in,out] tensors Tensors required by the run workloads
     *
     * @return Status If the run is successful
     */
    Status run(const std::vector<CLTensor *> &tensors);
    /** Get auxiliary tensors of the workload and their memory requirement
     */
    std::vector<std::tuple<CLTensor *, TensorInfo, AuxMemoryInfo>> get_auxiliary_tensors();

private:
    /** Enqueue prepare workload
     * @note If the runtime is not configured, this method will not perform any action
     */
    void prepare();
    struct Implementation;
    std::unique_ptr<Implementation> _impl;
};

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif /* ARM_COMPUTE_DYNAMIC_FUSION_RUNTIME_GPU_CL_CLWORKLOADRUNTIME */
