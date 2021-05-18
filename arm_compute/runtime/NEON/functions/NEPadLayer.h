/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_NEPADLAYER_H
#define ARM_COMPUTE_NEPADLAYER_H

#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/NEON/functions/NEConcatenateLayer.h"
#include "arm_compute/runtime/NEON/functions/NECopy.h"
#include "arm_compute/runtime/NEON/functions/NEStridedSlice.h"
#include "arm_compute/runtime/SubTensor.h"

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/Tensor.h"
#include <memory>

namespace arm_compute
{
class NEPadLayerKernel;

/** Basic function to pad a tensor. This function calls the following functions/kernels:
 *
 *  - For padding mode = PaddingMode::CONSTANT:
 *      -# @ref NEPadLayerKernel
 *  - Otherwise:
 *      -# @ref NECopy
 *      -# @ref NEStridedSlice
 *      -# @ref NEConcatenateLayer
 *
 */
class NEPadLayer : public IFunction
{
public:
    /** Default Constructor */
    NEPadLayer();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEPadLayer(const NEPadLayer &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEPadLayer &operator=(const NEPadLayer &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEPadLayer(NEPadLayer &&) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEPadLayer &operator=(NEPadLayer &&) = delete;
    /** Default destructor */
    ~NEPadLayer();
    /** Initialize the function
     *
     * Valid data layouts:
     * - NHWC
     * - NCHW
     *
     * Valid data type configurations:
     * |src      |dst       |
     * |:--------|:---------|
     * |All      |All       |
     *
     * @param[in]  input          Source tensor. Data types supported: All.
     * @param[out] output         Output tensor. Data type supported: same as @p input
     * @param[in]  padding        The padding for each spatial dimension of the input tensor. The pair padding[i]
     *                            specifies the front and the end padding in the i-th dimension.
     * @param[in]  constant_value (Optional) Constant value to be used for the padding
     * @param[in]  mode           (Optional) Controls whether the padding should be filled with @p constant_value using CONSTANT,
     *                            or reflect the input, either including the border values (SYMMETRIC) or not (REFLECT).
     */
    void configure(ITensor *input, ITensor *output, const PaddingList &padding, const PixelValue constant_value = PixelValue(), const PaddingMode mode = PaddingMode::CONSTANT);
    /**  Static function to check if given info will lead to a valid configuration of @ref NEPadLayer.
     *
     * @param[in] input          Source tensor info. Data types supported: All.
     * @param[in] output         Output tensor info. Data type supported: same as @p input
     * @param[in] padding        The padding for each spatial dimension of the input tensor. The pair padding[i]
     *                           specifies the front and the end padding in the i-th dimension.
     * @param[in] constant_value (Optional) Constant value to be used for the padding
     * @param[in] mode           (Optional) Controls whether the padding should be filled with @p constant_value using CONSTANT,
     *                     or reflect the input, either including the border values (SYMMETRIC) or not (REFLECT).
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const PaddingList &padding, const PixelValue constant_value = PixelValue(), const PaddingMode mode = PaddingMode::CONSTANT);

    // Inherited methods overridden:
    void run() override;

private:
    /** Configure kernels for when constant padding is used.
     *
     * @param[in]  input          Source tensor. Data types supported: All.
     * @param[out] output         Output tensor. Data type supported: same as @p input
     * @param[in]  padding        The padding for each spatial dimension of the input tensor. The pair padding[i]
     *                            specifies the front and the end padding in the i-th dimension.
     * @param[in]  constant_value Constant value to be used for the padding
     */
    void configure_constant_mode(ITensor *input, ITensor *output, const PaddingList &padding, const PixelValue constant_value);
    /** Configure functions for when reflect or symmetric padding is used.
     *
     * @param[in]  input  Source tensor. Data types supported: All.
     * @param[out] output Output tensor. Data type supported: same as @p input
     */
    void configure_reflect_symmetric_mode(ITensor *input, ITensor *output);

private:
    NECopy                            _copy_function;
    std::unique_ptr<NEPadLayerKernel> _pad_kernel;
    PaddingMode                       _mode;
    PaddingList                       _padding;
    uint32_t                          _num_dimensions;
    std::vector<NEStridedSlice>       _slice_functions;
    std::vector<NEConcatenateLayer>   _concat_functions;
    std::vector<Tensor>               _slice_results;
    std::vector<Tensor>               _concat_results;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_NEPADLAYER_H */
