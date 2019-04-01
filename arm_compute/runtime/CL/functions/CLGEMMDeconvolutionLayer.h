/*
 * Copyright (c) 2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLGEMMDECONVOLUTIONLAYER_H__
#define __ARM_COMPUTE_CLGEMMDECONVOLUTIONLAYER_H__

#include "arm_compute/core/CL/kernels/CLDeconvolutionReshapeOutputKernel.h"
#include "arm_compute/runtime/CL/CLMemoryGroup.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/functions/CLConvolutionLayer.h"
#include "arm_compute/runtime/CL/functions/CLPermute.h"
#include "arm_compute/runtime/CL/functions/CLReshapeLayer.h"
#include "arm_compute/runtime/CL/functions/CLSlice.h"
#include "arm_compute/runtime/CL/functions/CLTranspose.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"

#include <memory>

namespace arm_compute
{
class ICLTensor;
/** Function to run the deconvolution layer through a call to GEMM.
 *
 * Deconvolution Layer is the backward pass of Convolution Layer. First we transform the input depending on the stride and pad info and then perform a 1x1
 * convolution pass. Input stride defines how many zeroes we should put between each element of the input, pad is the amount of padding and finally a is a user
 * specified value where a < stride - 1, that increases the padding top and right of the input image.
 *
 *  The relation between input to output is as follows:
 *  \f[
 *       width\_output = (width\_input - 1) \cdot stride\_x - 2 \cdot padding\_x + kernel\_x
 *  \f]
 *  \f[
 *       height\_output = (height\_input - 1) \cdot stride\_y - 2 \cdot padding\_y + kernel\_y
 *  \f]
 *
 *  where:
 *      width_input is the size of the first input dimension.
 *      height_input is the size of the second input dimension.
 *      width_output is the size of the first output dimension.
 *      height_output is the size of the second output dimension.
 *      kernel_x and kernel_y are the convolution sizes in x and y.
 *      stride_x and stride_y is the input stride of the first and second dimension.
 *
 * The weights used by Deconvolution are supposed to be the same as the ones used for Convolution.
 *
 * This function calls the following OpenCL kernels/functions:
 *
 * -# @ref CLGEMMLowpMatrixMultiplyCore
 * -# @ref CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint
 * -# @ref CLPermute
 * -# @ref CLPermute
 * -# @ref CLReshapeLayer
 * -# @ref CLTranspose
 * -# @ref CLDeconvolutionReshapeOutputKernel
 * -# @ref CLSlice
 */
class CLGEMMDeconvolutionLayer : public IFunction
{
public:
    /** Constructor */
    CLGEMMDeconvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLGEMMDeconvolutionLayer(const CLGEMMDeconvolutionLayer &) = delete;
    /** Default move constructor */
    CLGEMMDeconvolutionLayer(CLGEMMDeconvolutionLayer &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLGEMMDeconvolutionLayer &operator=(const CLGEMMDeconvolutionLayer &) = delete;
    /** Default move assignment operator */
    CLGEMMDeconvolutionLayer &operator=(CLGEMMDeconvolutionLayer &&) = default;
    /** Set the input, weights, biases and output tensors.
     *
     * @param[in,out] input       Input tensor. 3 lower dimensions represent a single input, and an optional 4th dimension for batch of inputs. Data types supported: F16/F32. Data layout supported: NHWC
     * @param[in]     weights     The 4d weights with dimensions [width, height, IFM, OFM]. Data type supported: Same as @p input. Data layout supported: same as @p input.
     * @param[in]     bias        (Optional) The biases have one dimension. Data type supported: Same as @p input. Data layout supported: same as @p input.
     * @param[out]    output      Output tensor. The output has the same number of dimensions as the @p input. Data layout supported: same as @p input.
     * @param[in]     deconv_info Contains padding and policies to be used in the deconvolution, this is described in @ref PadStrideInfo. This function supports only stride_x = weights.width && stride_y = weights.height. Moreover, padding is not supported.
     */
    void configure(const ICLTensor *input, const ICLTensor *weights, const ICLTensor *bias, ICLTensor *output, const PadStrideInfo &deconv_info);
    /** Static function to check if given info will lead to a valid configuration of @ref CLDeconvolutionLayer
     *
     * @param[in] input       Input tensor info. 3 lower dimensions represent a single input, and an optional 4th dimension for batch of inputs. Data types supported: F16/F32. Data layout supported: NHWC
     * @param[in] weights     The 4d weights info with dimensions [width, height, IFM, OFM]. Data type supported: Same as @p input. Data layout supported: same as @p input.
     * @param[in] bias        (Optional) The biases have one dimension. Data type supported: Same as @p input. Data layout supported: same as @p input.
     * @param[in] output      Output tensor info. The output has the same number of dimensions as the @p input. Data layout supported: same as @p input.
     * @param[in] deconv_info Contains padding and policies to be used in the deconvolution, this is described in @ref PadStrideInfo.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *bias, const ITensorInfo *output, const PadStrideInfo &deconv_info);

    // Inherited methods overridden:
    void run() override;
    void prepare() override;

private:
    CLMemoryGroup _memory_group;

    CLGEMM                                              _mm_gemm;
    CLGEMMLowpMatrixMultiplyCore                        _mm_gemmlowp;
    CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint _gemmlowp_output_stage;
    CLPermute                                           _permute_input_to_nhwc;
    CLPermute                                           _permute_weights_to_nhwc;
    CLReshapeLayer                                      _reshape_weights;
    CLTranspose                                         _transpose_weights;
    CLDeconvolutionReshapeOutputKernel                  _deconv_reshape;
    CLSlice                                             _slice_gemm;

    CLTensor _gemmlowp_final;
    CLTensor _reshaped_weights;
    CLTensor _reshaped_weights_t;
    CLTensor _permuted_input;
    CLTensor _permuted_weights;
    CLTensor _gemm_output;
    CLTensor _slice_gemm_input;

    const ICLTensor *_original_weights;
    bool             _is_prepared;
    bool             _padded_input;
    bool             _is_nchw;
    bool             _is_quantized;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CLGEMMDECONVOLUTIONLAYER_H__ */
