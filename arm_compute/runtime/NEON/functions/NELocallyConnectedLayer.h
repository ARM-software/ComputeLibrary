/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_NELOCALLYCONNECTEDLAYER_H
#define ARM_COMPUTE_NELOCALLYCONNECTEDLAYER_H

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/NEON/functions/NECol2Im.h"
#include "arm_compute/runtime/NEON/functions/NEIm2Col.h"
#include "arm_compute/runtime/Tensor.h"

#include <memory>

namespace arm_compute
{
class INETensor;
class NEWeightsReshapeKernel;
class NELocallyConnectedMatrixMultiplyKernel;

/** Basic function to compute the locally connected layer. This function calls the following NEON kernels:
 *
 * -# @ref NEWeightsReshapeKernel (executed only once for each configuration)
 * -# @ref NEIm2ColKernel
 * -# @ref NELocallyConnectedMatrixMultiplyKernel
 * -# @ref NECol2ImKernel
 */
class NELocallyConnectedLayer : public IFunction
{
public:
    /** Default constructor */
    NELocallyConnectedLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NELocallyConnectedLayer(const NELocallyConnectedLayer &) = delete;
    /** Prevent instances of this class from being moved (As this class contains pointers) */
    NELocallyConnectedLayer(NELocallyConnectedLayer &&) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NELocallyConnectedLayer &operator=(const NELocallyConnectedLayer &) = delete;
    /** Prevent instances of this class from being moved (As this class contains pointers) */
    NELocallyConnectedLayer &operator=(NELocallyConnectedLayer &&) = delete;
    /** Default destructor */
    ~NELocallyConnectedLayer();
    /** Set the input and output tensors.
     *
     * @param[in]  input     Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
     *                       while every optional dimension from 4 and above represent a batch of inputs.
     *                       Data types supported: F16, F32.
     * @param[in]  weights   Weights tensor. Weights are 5D tensor with dimensions [kernel_x, kernel_y, IFM, OFM, num_patches]. Data type supported:Same as @p input.
     * @param[in]  biases    Biases tensor. Shared biases supported. Biases are 2D tensor with dimensions [OFM, num_patches]. Data type supported:Same as @p input.
     * @param[out] output    Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                       Data types supported: Same as @p input.
     * @param[in]  conv_info Contains padding and stride information described in @ref PadStrideInfo.
     */
    ARM_COMPUTE_DEPRECATED_REL(20.11)
    void configure(const ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const PadStrideInfo &conv_info);
    /** Static function to check if given info will lead to a valid configuration of @ref NELocallyConnectedLayer
     *
     * @param[in] input     Input tensor info. 3 lower dimensions represent a single input [width, height, IFM],
     *                      while every optional dimension from 4 and above represent a batch of inputs.
     *                      Data types supported: F16, F32.
     * @param[in] weights   Weights tensor info. Weights are 5D tensor with dimensions [kernel_x, kernel_y, IFM, OFM, num_patches]. Data type supported:Same as @p input.
     * @param[in] biases    Biases tensor info. Shared biases supported. Biases are 2D tensor with dimensions [OFM, num_patches]. Data type supported:Same as @p input.
     * @param[in] output    Output tensor info. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                      Data types supported: Same as @p input.
     * @param[in] conv_info Contains padding and stride information described in @ref PadStrideInfo.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info);

    // Inherited methods overridden:
    void run() override;
    void prepare() override;

private:
    MemoryGroup                                             _memory_group;
    NEIm2Col                                                _input_im2col;
    std::unique_ptr<NEWeightsReshapeKernel>                 _weights_reshape_kernel;
    std::unique_ptr<NELocallyConnectedMatrixMultiplyKernel> _mm_kernel;
    NECol2Im                                                _output_col2im;
    Tensor                                                  _input_im2col_reshaped;
    Tensor                                                  _weights_reshaped;
    Tensor                                                  _gemm_output;
    bool                                                    _is_prepared;
    const ITensor                                          *_original_weights;
};
}
#endif /* ARM_COMPUTE_NELOCALLYCONNECTEDLAYER_H */
