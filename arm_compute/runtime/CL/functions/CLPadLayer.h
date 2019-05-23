/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLPADLAYER_H__
#define __ARM_COMPUTE_CLPADLAYER_H__

#include "arm_compute/core/CL/kernels/CLCopyKernel.h"
#include "arm_compute/core/CL/kernels/CLMemsetKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/functions/CLConcatenateLayer.h"

#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/functions/CLStridedSlice.h"
#include "arm_compute/runtime/IFunction.h"

namespace arm_compute
{
class ICLTensor;

/** Basic function to pad a tensor. This function calls the following OpenCL kernels:
 *
 *  -# @ref CLMemsetKernel
 *  -# @ref CLFillBorderKernel
 *  -# @ref CLCopyKernel
 */
class CLPadLayer : public IFunction
{
public:
    /** Default constructor*/
    CLPadLayer();

    /** Initialize the function
     *
     * @param[in]  input          Source tensor. Data types supported: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32.
     * @param[out] output         Output tensor. Data type supported: same as @p input
     * @param[in]  padding        The padding for each spatial dimension of the input tensor. The pair padding[i]
     *                            specifies the front and the end padding in the i-th dimension.
     * @param[in]  constant_value (Optional) Constant value to be used for the padding.
     * @param[in]  mode           (Optional) Controls whether the padding should be filled with @p constant_value using CONSTANT,
     *                            or reflect the input, either including the border values (SYMMETRIC) or not (REFLECT). Only CONSTANT
     *                              is currently supported.
     */
    void configure(ICLTensor *input, ICLTensor *output, const PaddingList &padding, PixelValue constant_value = PixelValue(), PaddingMode mode = PaddingMode::CONSTANT);

    /**  Static function to check if given info will lead to a valid configuration of @ref CLPadLayer.
     *
     * @param[in] input          Source tensor info. Data types supported: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32.
     * @param[in] output         Output tensor info. Data type supported: same as @p input
     * @param[in] padding        The padding for each spatial dimension of the input tensor. The pair padding[i]
     *                           specifies the front and the end padding in the i-th dimension.
     * @param[in] constant_value (Optional) Constant value to be used for the padding
     * @param[in] mode           (Optional) Controls whether the padding should be filled with @p constant_value using CONSTANT,
     *                            or reflect the input, either including the border values (SYMMETRIC) or not (REFLECT). Only CONSTANT
     *                              is currently supported.
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const PaddingList &padding, PixelValue constant_value = PixelValue(), PaddingMode mode = PaddingMode::CONSTANT);

    // Inherited methods overridden:
    void run() override;

private:
    void configure_constant_mode(ICLTensor *input, ICLTensor *output, const PaddingList &padding, const PixelValue constant_value);
    void configure_reflect_symmetric_mode(ICLTensor *input, ICLTensor *output);

    CLCopyKernel                    _copy_kernel;
    PaddingMode                     _mode;
    PaddingList                     _padding;
    CLMemsetKernel                  _memset_kernel;
    size_t                          _num_dimensions;
    std::vector<CLStridedSlice>     _slice_functions;
    std::vector<CLConcatenateLayer> _concat_functions;
    std::vector<CLTensor>           _slice_results;
    std::vector<CLTensor>           _concat_results;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_PADLAYER_H__ */
