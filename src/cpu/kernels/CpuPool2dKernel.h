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
#ifndef ARM_COMPUTE_CPU_POOL2D_KERNEL_H
#define ARM_COMPUTE_CPU_POOL2D_KERNEL_H

#include "arm_compute/core/Types.h"
#include "src/core/common/Macros.h"
#include "src/cpu/ICpuKernel.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
/** Interface for the pooling layer kernel */
class CpuPool2dKernel : public ICpuKernel<CpuPool2dKernel>
{
private:
    using PoolingKernelPtr = std::add_pointer<void(const ITensor *, ITensor *, ITensor *, PoolingLayerInfo &, const Window &, const Window &)>::type;

public:
    CpuPool2dKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuPool2dKernel);
    /** Configure kernel for a given list of arguments
     *
     * @note F16 are supported for pool sizes 2 and 3 only
     *
     * @param[in]  src       Source tensor info. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[out] dst       Destination tensor info. Data types supported: Same as @p src.
     * @param[in]  pool_info Contains pooling operation information described in @ref PoolingLayerInfo.
     * @param[out] indices   (optional) The indices of the maximal values. Data type supported: U32.
     */
    void configure(ITensorInfo *src, ITensorInfo *dst, const PoolingLayerInfo &pool_info, ITensorInfo *indices = nullptr);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to CpuPool2dKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *dst, const PoolingLayerInfo &pool_info, const ITensorInfo *indices = nullptr);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;
    const char *name() const override;

    struct PoolingKernel
    {
        const char                      *name;
        const PoolDataTypeISASelectorPtr is_selected;
        PoolingKernelPtr                 ukernel;
    };

    static const std::vector<PoolingKernel> &get_available_kernels();

private:
    PoolingLayerInfo _pool_info{};
    DataLayout       _data_layout{ DataLayout::UNKNOWN };
    unsigned int     _num_elems_processed_per_iteration{ 0 };
    Size2D           _pool_size{};
    int              _pool_stride_x{};
    PoolingKernelPtr _run_method{ nullptr };
    std::string      _name{};
};
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_POOL2D_KERNEL_H */
