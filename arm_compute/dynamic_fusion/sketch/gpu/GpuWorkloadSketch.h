/*
 * Copyright (c) 2022 Arm Limited.
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
#ifndef ARM_COMPUTE_DYNAMIC_FUSION_SKETCH_GPU_GPUWORKLOADSKETCH
#define ARM_COMPUTE_DYNAMIC_FUSION_SKETCH_GPU_GPUWORKLOADSKETCH

#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/dynamic_fusion/sketch/gpu/GpuWorkloadContext.h"

#include <memory>

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
/** A descriptor of a workload of operators
 *
 * A "workload" is a basic unit of computation to schedule and perform. It contains one or more operators that can be "fused" together.
 * Note that a workload may still contain multiple kernels.
 */
class GpuWorkloadSketch
{
public:
    /** Global context used for the creation of a workload */
    using Context = GpuWorkloadContext;
    /** Internal opaque implementation */
    class Implementation;

public:
    /** Constructor
     *
     * @param[in] context Gpu context for the creation of a workload
     */
    explicit GpuWorkloadSketch(GpuWorkloadContext *context);
    /** Destructor */
    ~GpuWorkloadSketch();
    /** Get the implementation */
    Implementation &implementation();
    /** Get the implementation */
    const Implementation &implementation() const;
    /** Get the gpu workload context of this sketch */
    const GpuWorkloadContext *gpu_context() const;
    /** Create a @ref TensorInfo associated with the workload sketch.
     *
     * @return TensorInfo   Newly created tensor info
     */
    template <typename... Args>
    TensorInfo create_tensor_info(Args &&... args)
    {
        auto tensor_info = TensorInfo(std::forward<Args>(args)...);
        register_new_tensor(tensor_info);
        return tensor_info;
    }
    /** Create a default @ref TensorInfo associated with the workload sketch
     * It is usually used by user input or output tensors
     *
     * @return TensorInfo   Newly created tensor info
     */
    TensorInfo create_tensor_info();

private:
    /** Register a new tensor by setting a new id to it and register its memory descriptor in the sketch
     *
     * @param[in,out] tensor_info @ref ITensorInfo that will be registered
     */
    void register_new_tensor(ITensorInfo &tensor_info);
    std::unique_ptr<Implementation> _impl; /**< Internal opaque implementation*/
};

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif /* ARM_COMPUTE_DYNAMIC_FUSION_SKETCH_GPU_GPUWORKLOADSKETCH */
