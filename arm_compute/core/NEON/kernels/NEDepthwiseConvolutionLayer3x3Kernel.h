/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEDEPTHWISECONVOLUTIONKERNEL3x3_H__
#define __ARM_COMPUTE_NEDEPTHWISECONVOLUTIONKERNEL3x3_H__

#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/NEON/kernels/convolution/depthwise/depthwise.hpp"

#include <memory>

namespace arm_compute
{
class ITensor;

/** Interface for the kernel to run a 3x3 depthwise convolution on a tensor. */
class NEDepthwiseConvolutionLayer3x3Kernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEDepthwiseConvolutionLayer3x3Kernel";
    }
    /** Default constructor */
    NEDepthwiseConvolutionLayer3x3Kernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEDepthwiseConvolutionLayer3x3Kernel(const NEDepthwiseConvolutionLayer3x3Kernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEDepthwiseConvolutionLayer3x3Kernel &operator=(const NEDepthwiseConvolutionLayer3x3Kernel &) = delete;
    /** Default Move Constructor. */
    NEDepthwiseConvolutionLayer3x3Kernel(NEDepthwiseConvolutionLayer3x3Kernel &&) = default;
    /** Default move assignment operator */
    NEDepthwiseConvolutionLayer3x3Kernel &operator=(NEDepthwiseConvolutionLayer3x3Kernel &&) = default;
    /** Initialize the function's source, destination, conv and border_size.
     *
     * @param[in]  input            Source tensor. DataType supported: QASYMM8, F32.
     * @param[in]  weights          Weights tensor. This is a 3D tensor with dimensions [3, 3, IFM]. Data type supported: Same as @p input.
     * @param[out] output           Destination tensor. Data type supported: Same as @p input.
     * @param[in]  conv_info        Padding and stride information to use for the convolution.
     * @param[in]  depth_multiplier (Optional) Multiplier to apply to the input's depth in order to retrieve the output's depth. Defaults to 1.
     * @param[in]  data_layout      (Optional) Data layout of the input and weights tensor
     */
    void configure(const ITensor *input, const ITensor *weights, ITensor *output, const PadStrideInfo &conv_info, unsigned int depth_multiplier = 1, DataLayout data_layout = DataLayout::NCHW);
    /** Static method that checks if optimized execution is supported for the given parameters
     *
     * @param[in] input_shape      Input shape
     * @param[in] conv_info        Padding and stride information to use for the convolution.
     * @param[in] dt               Data type of the input and weights
     * @param[in] data_layout      (Optional) Data layout of the input and weights tensor
     * @param[in] depth_multiplier (Optional) Multiplier to apply to the input's depth in order to retrieve the output's depth. Defaults to 1.
     *
     * @return True if the optimized kernels can be executed else false
     */
    static bool is_optimized_execution_possible(TensorShape input_shape, PadStrideInfo conv_info, DataType dt, unsigned int depth_multiplier = 1, DataLayout data_layout = DataLayout::NCHW);
    /** Generates the convolver object */
    void generate_convolver();

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;
    BorderSize border_size() const override;

private:
    void configure_generic();
    void configure_optimized();
    void run_generic(const Window &window, const ThreadInfo &info);
    void run_optimized(const Window &window, const ThreadInfo &info);
    /** Creates an optimized backend convolver object
     *
     * @note Convolver of strides 1,2 and convolution size of 3 is currently supported
     *
     * @param[in] conv_info     Padding and stride information to use for the convolution
     * @param[in] w             Weights tensor
     * @param[in] in            Input tensor
     * @param[in] out           Output tensor
     * @param[in] setup_strides (Optional) Boolean to enable setting the strides of the tensors
     *                           in the convolver in case of padding. Defaults to false
     *
     * @return  A convolver object or nullptr if the configuration is not supported
     */
    std::unique_ptr<depthwise::IDepthwiseConvolution> create_convolver_object(PadStrideInfo  conv_info,
                                                                              const ITensor *w,
                                                                              const ITensor *in,
                                                                              ITensor       *out,
                                                                              bool           setup_strides = false);

private:
    BorderSize                                        _border_size;
    const ITensor                                    *_input;
    ITensor                                          *_output;
    const ITensor                                    *_weights;
    PadStrideInfo                                     _conv_info;
    std::unique_ptr<depthwise::IDepthwiseConvolution> _convolver;
    unsigned int                                      _num_elems_written_per_iteration;
    bool                                              _run_optimized;
    unsigned int                                      _depth_multiplier;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_NEDEPTHWISECONVOLUTIONKERNEL3x3_H__ */
