/*
 * Copyright (c) 2017-2020, 2024-2025 Arm Limited.
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

#ifndef ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_OPERATORS_CPUPOOL2D_H
#define ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_OPERATORS_CPUPOOL2D_H

#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h" // required for PoolingLayerInfo
#include "arm_compute/runtime/NEON/INEOperator.h"

#include <memory>

namespace arm_compute
{
namespace experimental
{
namespace op
{

/*
 * A shallow wrapper for arm_compute::cpu::CpuPool2d.
 * Any new features should be added to arm_compute::cpu::CpuPool2d,
 * and arm_compute::experimental::op::CpuPool2d should remain a shallow wrapper.
 * Note this is not thread safe therefore no thread safety tests provided.
 */
class CpuPool2d : public INEOperator
{
public:
    /** Constructor **/
    CpuPool2d();
    /** Prevent instances of this class from being copied (As this class contains pointers) **/
    CpuPool2d(const CpuPool2d &) = delete;
    /** Prevent copy assignment **/
    CpuPool2d &operator=(const CpuPool2d &) = delete;
    /** Default move constructor **/
    CpuPool2d(CpuPool2d &&) = default;
    /** Default move assignment **/
    CpuPool2d &operator=(CpuPool2d &&) = default;
    /** Default destructor **/
    ~CpuPool2d() override;
    /** Set the src and dst tensors.
     *
     * Valid data layouts:
     * - NHWC
     * - NCHW
     *
     * Valid data type configurations:
     * |src            |dst            |
     * |:--------------|:--------------|
     * |QASYMM8        |QASYMM8        |
     * |QASYMM8_SIGNED |QASYMM8_SIGNED |
     * |F16            |F16            |
     * |F32            |F32            |
     *
     * @note F16 is supported for pool sizes 2 and 3 only
     * @note Source tensor is padded with -inf for MAX pooling and 0 otherwise
     *       Cases where pooling region is completely outside input tensor are only supported for floating point data type
     *
     * @param[in, out] src       Source tensor info. (Written to only when padding != 0) Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[out]     dst       Destination tensor info. Data types supported: same as @p src.
     * @param[in]      pool_info Contains pooling operation information described in @ref PoolingLayerInfo.
     * @param[out]     indices   (optional) The indices of the maximal values. Data type supported: U32.
     */
    void
    configure(ITensorInfo *src, ITensorInfo *dst, const PoolingLayerInfo &pool_info, ITensorInfo *indices = nullptr);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref CpuPool2d::configure()
     */

    static Status validate(const ITensorInfo      *src,
                           const ITensorInfo      *dst,
                           const PoolingLayerInfo &pool_info,
                           const ITensorInfo      *indices = nullptr);

    // Inherited methods overridden:
    void                             run(ITensorPack &tensors) override;
    experimental::MemoryRequirements workspace() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace op
} //namespace experimental
} // namespace arm_compute
#endif // ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_OPERATORS_CPUPOOL2D_H
