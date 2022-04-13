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
#ifndef ARM_COMPUTE_CPU_POOL3D_H
#define ARM_COMPUTE_CPU_POOL3D_H

#include "arm_compute/core/experimental/Types.h"
#include "src/core/common/Macros.h"
#include "src/cpu/ICpuOperator.h"

#include <memory>

namespace arm_compute
{
namespace cpu
{
/** Basic function to simulate a pooling layer with the specified pooling operation. This function calls the following kernels:
 *
 * -# @ref kernels::CpuPool3dKernel
 */
class CpuPool3d : public ICpuOperator
{
public:
    CpuPool3d();
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuPool3d);
    ~CpuPool3d();
    /** Set the src and dst tensors.
     *
     *
     * @param[in]  src       Source tensor info. Data types supported: F16/F32/QASYMM8/QASYMM8_SIGNED.
     * @param[out] dst       Destination tensor info. Data types supported: same as @p src.
     * @param[in]  pool_info Contains pooling operation information described in @ref Pooling3dLayerInfo.
     */
    void configure(const ITensorInfo *src, ITensorInfo *dst, const Pooling3dLayerInfo &pool_info);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to CpuPool3d::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *dst, const Pooling3dLayerInfo &pool_info);

    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;
    experimental::MemoryRequirements workspace() const override;

private:
    experimental::MemoryRequirements _aux_mem{};
};
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_POOL3D_H */
