/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CLL2NORMALIZELAYER_H
#define ARM_COMPUTE_CLL2NORMALIZELAYER_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/ICLSimpleFunction.h"
#include "arm_compute/runtime/CL/functions/CLReductionOperation.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/MemoryGroup.h"

#include <cstdint>
#include <memory>

namespace arm_compute
{
class CLCompileContext;
class CLL2NormalizeLayerKernel;
class ICLTensor;
class ITensorInfo;

/** Basic function to perform a L2 normalization on a given axis.
 *
 * This function runs the following kernels:
 * -# @ref CLReductionOperation
 * -# @ref CLL2NormalizeLayerKernel
 */
class CLL2NormalizeLayer : public IFunction
{
public:
    /** Constructor */
    CLL2NormalizeLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Default Destructor */
    ~CLL2NormalizeLayer();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLL2NormalizeLayer(const CLL2NormalizeLayer &) = delete;
    /** Default move constructor */
    CLL2NormalizeLayer(CLL2NormalizeLayer &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLL2NormalizeLayer &operator=(const CLL2NormalizeLayer &) = delete;
    /** Default move assignment operator */
    CLL2NormalizeLayer &operator=(CLL2NormalizeLayer &&) = default;

    /** Set the input and output tensors.
     *
     * Valid data layouts:
     * - NHWC
     * - NCHW
     *
     * Valid data type configurations:
     * |src      |dst       |
     * |:--------|:---------|
     * |F16      |F16       |
     * |F32      |F32       |
     *
     * @param[in]  input   Source tensor. Data types supported: F16/F32. Data layouts supported: NCHW/NHWC.
     * @param[out] output  Destination tensor. Data types and data layouts supported: Same as @p input.
     * @param[in]  axis    Axis along which to reduce. Negative values wrap around. Maximum supported actual reduction axis : 2
     * @param[in]  epsilon (Optional) Lower bound value for the normalization.
     */
    void configure(ICLTensor *input, ICLTensor *output, int axis, float epsilon = 1e-12f);
    /** Set the input and output tensors.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Source tensor. Data types supported: F16/F32. Data layouts supported: NCHW/NHWC.
     * @param[out] output          Destination tensor. Data types and data layouts supported: Same as @p input.
     * @param[in]  axis            Axis along which to reduce. Negative values wrap around. Maximum supported actual reduction axis : 2
     * @param[in]  epsilon         (Optional) Lower bound value for the normalization.
     */
    void configure(const CLCompileContext &compile_context, ICLTensor *input, ICLTensor *output, int axis, float epsilon = 1e-12f);

    /** Static function to check if given info will lead to a valid configuration of @ref CLL2NormalizeLayer.
     *
     * @param[in] input   Source tensor info. Data types supported: F16/F32. Data layouts supported: NCHW/NHWC.
     * @param[in] output  Destination tensor info. Data types and data layouts supported: Same as @p input.
     * @param[in] axis    Axis along which to reduce. Negative values wrap around. Maximum supported actual reduction axis : 2
     * @param[in] epsilon (Optional) Lower bound value for the normalization.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, int axis, float epsilon = 1e-12f);

    // Inherited methods overridden:
    void run() override;

private:
    MemoryGroup                               _memory_group;
    CLReductionOperation                      _reduce_func;
    std::unique_ptr<CLL2NormalizeLayerKernel> _normalize_kernel;
    CLTensor                                  _sumsq;
};
}
#endif /*ARM_COMPUTE_CLL2NORMALIZELAYER_H */
