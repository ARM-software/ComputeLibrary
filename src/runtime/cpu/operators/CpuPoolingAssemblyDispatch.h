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
#ifndef ARM_COMPUTE_CPU_POOLING_ASSEMBLY_DISPATCH_H
#define ARM_COMPUTE_CPU_POOLING_ASSEMBLY_DISPATCH_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/Tensor.h"
#include "src/runtime/cpu/ICpuOperator.h"

namespace arm_compute
{
namespace cpu
{
class ITensor;

/** Basic function to run pooling assembly kernels */
class CpuPoolingAssemblyDispatch : public ICpuOperator
{
public:
    /** Constructor */
    CpuPoolingAssemblyDispatch(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied */
    CpuPoolingAssemblyDispatch(const CpuPoolingAssemblyDispatch &) = delete;
    /** Default move constructor */
    CpuPoolingAssemblyDispatch(CpuPoolingAssemblyDispatch &&) = default;
    /** Prevent instances of this class from being copied */
    CpuPoolingAssemblyDispatch &operator=(const CpuPoolingAssemblyDispatch &) = delete;
    /** Default move assignment operator */
    CpuPoolingAssemblyDispatch &operator=(CpuPoolingAssemblyDispatch &&) = default;
    /** Destructor */
    ~CpuPoolingAssemblyDispatch();

    /** If supported create an assembly routine, else fallback to Compute Library function.
     *
     * @param[in]  src  Source tensor info. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[out] dst  Destination tensor info to store the result of pooling. Data types supported: same as @p src.
     * @param[in]  info Pooling meta-data
     */
    void configure(const ITensorInfo *src, ITensorInfo *dst, const PoolingLayerInfo &info);

    /** Indicates whether or not this function can be used to process the given parameters.
     *
     * @param[in] src  Source tensor info. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in] dst  Destination tensor to store the result of pooling. Data types supported: same as @p src.
     * @param[in] info Pooling meta-data
     *
     * @return a status.
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *dst, const PoolingLayerInfo &info);
    /** Was the function successfully configured ?
     *
     * @return True if the function is configured and ready to run
     */
    bool is_configured() const;
    // Run method overriden
    void run(ITensorPack &tensors) override;

private:
    arm_compute::MemoryGroup _memory_group;

    arm_compute::Tensor _workspace;
    bool                _is_global_pooling_layer;
};
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_POOLING_ASSEMBLY_DISPATCH_H */
