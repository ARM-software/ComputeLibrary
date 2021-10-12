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
#ifndef ARM_COMPUTE_CPU_POOL2D_H
#define ARM_COMPUTE_CPU_POOL2D_H

#include "arm_compute/core/experimental/Types.h"
#include "src/core/common/Macros.h"
#include "src/cpu/ICpuOperator.h"

#include <memory>

namespace arm_compute
{
// Forward Declarations
struct PoolingLayerInfo;

namespace cpu
{
/** Basic function to simulate a pooling layer with the specified pooling operation. This function calls the following kernels:
 *
 * -# @ref NEFillBorderKernel (executed if padding size is different from zero)
 * -# @ref kernels::CpuPool2dKernel
 * -# @ref kernels::CpuPool2dAssemblyWrapperKernel
 */
class CpuPool2d : public ICpuOperator
{
public:
    CpuPool2d();
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuPool2d);
    ~CpuPool2d();
    /** Set the src and dst tensors.
     *
     * @note F16 is supported for pool sizes 2 and 3 only
     *
     * @param[in, out] src       Source tensor info. (Written to only when padding != 0) Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[out]     dst       Destination tensor info. Data types supported: same as @p src.
     * @param[in]      pool_info Contains pooling operation information described in @ref PoolingLayerInfo.
     * @param[out]     indices   (optional) The indices of the maximal values. Data type supported: U32.
     */
    void configure(ITensorInfo *src, ITensorInfo *dst, const PoolingLayerInfo &pool_info, ITensorInfo *indices = nullptr);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to CpuPool2d::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *dst, const PoolingLayerInfo &pool_info, const ITensorInfo *indices = nullptr);

    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;
    experimental::MemoryRequirements workspace() const override;

private:
    std::unique_ptr<INEKernel> _pooling_layer_kernel;
    std::unique_ptr<INEKernel> _asm_glue;

    bool                             _is_global_pooling_layer;
    DataLayout                       _data_layout;
    experimental::MemoryRequirements _aux_mem{};
};
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_POOL2D_H */
