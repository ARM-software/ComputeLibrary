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
#ifndef ARM_COMPUTE_NEPOOLINGASSEMBLYDISPATCH_H
#define ARM_COMPUTE_NEPOOLINGASSEMBLYDISPATCH_H

#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/NEON/INEOperator.h"
#include "arm_compute/runtime/Tensor.h"
#include "src/core/NEON/INEKernel.h"

namespace arm_compute
{
// Forward Declarations
class ITensor;
struct PoolingLayerInfo;

/** Assembly kernel glue */
class NEPoolingAssemblyDispatch : public IFunction
{
public:
    /** Constructor */
    NEPoolingAssemblyDispatch(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEPoolingAssemblyDispatch(const NEPoolingAssemblyDispatch &) = delete;
    /** Default move constructor */
    NEPoolingAssemblyDispatch(NEPoolingAssemblyDispatch &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEPoolingAssemblyDispatch &operator=(const NEPoolingAssemblyDispatch &) = delete;
    /** Default move assignment operator */
    NEPoolingAssemblyDispatch &operator=(NEPoolingAssemblyDispatch &&);
    /** Destructor */
    ~NEPoolingAssemblyDispatch();

    /** If supported create an assembly routine, else fallback to Compute Library function.
     *
     * @param[in]  input  Input tensor. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[out] output Output tensor to store the result of pooling. Data types supported: same as @p input.
     * @param[in]  info   Pooling meta-data
     */
    void configure(const ITensor *input, ITensor *output, const PoolingLayerInfo &info);

    /** Indicates whether or not this function can be used to process the given parameters.
     *
     * @param[in] input  Input tensor info. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in] output Output tensor to store the result of pooling. Data types supported: same as @p input.
     * @param[in] info   Pooling meta-data
     *
     * @return a status.
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const PoolingLayerInfo &info);

    /** Was the function successfully configured ?
     *
     * @return True if the function is configured and ready to run
     */
    bool is_configured() const;

    // Inherited methods overridden:
    void run() override;

private:
    /** Helper function to allocate memory for the workspace needed by the
     * assembly kernels
     *
     * @param[in] workspace_size Total size of the workspace.
     * @param[in] alignment      Alignment requirement in bytes.
     */
    void allocate_workspace(size_t workspace_size, size_t alignment);

    struct Impl;
    std::unique_ptr<Impl> _impl;

    MemoryGroup _memory_group{};
    Tensor      _workspace{};
};

namespace experimental
{
/** Basic function to run pooling assembly kernels */
class NEPoolingAssemblyDispatch : public INEOperator
{
public:
    /** Constructor */
    NEPoolingAssemblyDispatch() = default;
    /** Prevent instances of this class from being copied */
    NEPoolingAssemblyDispatch(const NEPoolingAssemblyDispatch &) = delete;
    /** Default move constructor */
    NEPoolingAssemblyDispatch(NEPoolingAssemblyDispatch &&) = default;
    /** Prevent instances of this class from being copied */
    NEPoolingAssemblyDispatch &operator=(const NEPoolingAssemblyDispatch &) = delete;
    /** Default move assignment operator */
    NEPoolingAssemblyDispatch &operator=(NEPoolingAssemblyDispatch &&) = default;
    /** Destructor */
    ~NEPoolingAssemblyDispatch();

    /** If supported create an assembly routine, else fallback to Compute Library function.
     *
     * @param[in]  input  Input tensor. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[out] output Output tensor to store the result of pooling. Data types supported: same as @p input.
     * @param[in]  info   Pooling meta-data
     */
    void configure(const ITensorInfo *input, ITensorInfo *output, const PoolingLayerInfo &info);

    /** Indicates whether or not this function can be used to process the given parameters.
     *
     * @param[in] input  Input tensor info. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in] output Output tensor to store the result of pooling. Data types supported: same as @p input.
     * @param[in] info   Pooling meta-data
     *
     * @return a status.
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const PoolingLayerInfo &info);
    /** Was the function successfully configured ?
     *
     * @return True if the function is configured and ready to run
     */
    bool is_configured() const;
    // Run method overriden
    void run(ITensorPack &tensors) override;

private:
    bool _is_global_pooling_layer{ false };
};
} // namespace experimental
} // namespace arm_compute
#endif /* ARM_COMPUTE_NEPOOLINGASSEMBLYDISPATCH_H */
