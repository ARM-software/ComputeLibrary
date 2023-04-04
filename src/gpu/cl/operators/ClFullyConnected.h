/*
 * Copyright (c) 2017-2021, 2023 Arm Limited.
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
#ifndef ARM_COMPUTE_CL_FULLY_CONNECTED_H
#define ARM_COMPUTE_CL_FULLY_CONNECTED_H

#include "arm_compute/core/TensorInfo.h"

#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/IClOperator.h"

#include <memory>

namespace arm_compute
{
namespace opencl
{
// Forward declarations
class ClConvertFullyConnectedWeights;
class ClFlatten;
class ClGemm;
class ClGemmLowpMatrixMultiplyCore;
class ClTranspose;

/** Basic function to compute a Fully Connected layer on OpenCL. This function calls the following OpenCL kernels:
 *
 *  -# @ref opencl::kernels::ClIm2ColKernel (called when the input comes from a convolutional layer)
 *  -# @ref CLTranspose (if @p are_weights_reshaped is set to false and transpose_weights is set to true ) (called once)
 *  -# @ref opencl::ClGemm or @ref CLGEMMLowpMatrixMultiplyCore (if quantized asymmetric)
 *
 * @note  The fully connected layer accepts "weights" tensors only with 2 dimensions.
 */
class ClFullyConnected : public IClOperator
{
public:
    ClFullyConnected();
    ~ClFullyConnected();
    /** Set the input and output tensors.
     *
     * Valid data layouts:
     * - NHWC
     * - NCHW
     *
     * Valid data type configurations:
     * |src0           |src1               |src2   |dst            |
     * |:--------------|:------------------|:------|:--------------|
     * |F16            |F16                |F16    |F16            |
     * |F32            |F32                |F32    |F32            |
     * |QASYMM8        |QASYMM8            |S32    |QASYMM8        |
     * |QASYMM8_SIGNED |QASYMM8_SIGNED     |S32    |QASYMM8_SIGNED |
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  src             Source tensor. Data type supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in]  weights         Weights tensor. The weights must be 2 dimensional.
     *                             If this function is called after a Convolution Layer, the (transposed) weights will have as many rows as the product of the first 3 input's dimensions.
     *                             If it is called after another FullyConnected Layer, the (transposed) weights will have as many rows as the input's first dimension.
     *                             Data type supported: Same as @p src.
     * @param[in]  biases          Bias tensor. Can be nullptr. Data type supported:Same as @p src.
     * @param[out] dst             Destination tensor. Its shape should be equal to the output of a matrix multiplication between:
     *                             - The output of im2col on the input and the (transposed) 2D weights, if the function is called after a Convolution Layer
     *                             - The input tensor and the (transposed) 2D weights, if the function is called after another FullyConnected Layer.
     *                             Data type supported: Same as @p src.
     * @param[in]  fc_info         (Optional) Fully connected layer additional info
     */
    void configure(const CLCompileContext &compile_context, ITensorInfo *src, ITensorInfo *weights, ITensorInfo *biases, ITensorInfo *dst,
                   FullyConnectedLayerInfo fc_info = FullyConnectedLayerInfo());
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to ClFullyConnected::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *dst,
                           FullyConnectedLayerInfo fc_info = FullyConnectedLayerInfo());

    // Inherited methods overriden
    void run(ITensorPack &tensors) override;
    void prepare(ITensorPack &tensors) override;
    experimental::MemoryRequirements workspace() const override;

private:
    void configure_fc_fc(const CLCompileContext &compile_context, ITensorInfo *src, ITensorInfo *weights, ITensorInfo *bias, ITensorInfo *dst, const FullyConnectedLayerInfo &fc_info);
    void configure_conv_fc(const CLCompileContext &compile_context, ITensorInfo *src, ITensorInfo *weights, ITensorInfo *bias, ITensorInfo *dst, const FullyConnectedLayerInfo &fc_info);
    void configure_mm(const CLCompileContext &compile_context, ITensorInfo *src, ITensorInfo *weights, ITensorInfo *bias, ITensorInfo *dst, const FullyConnectedLayerInfo &fc_info);

private:
    enum AuxTensorIdx
    {
        TransposedWeights = 10,
        ConvertedWeights  = 11,
        FlattenedSrc      = 12,
        Count             = 13
    };

    std::unique_ptr<ClConvertFullyConnectedWeights> _convert_weights;
    std::unique_ptr<ClFlatten>                      _flatten;
    std::unique_ptr<ClTranspose>                    _reshape_weights;
    std::unique_ptr<ClGemm>                         _mm_gemm;
    std::unique_ptr<ClGemmLowpMatrixMultiplyCore>   _mm_gemmlowp;

    experimental::MemoryRequirements _aux_mem{};

    TensorInfo _flattened_src{};
    TensorInfo _converted_weights{};
    TensorInfo _reshaped_weights{};

    TensorInfo _weights_to_use{};
    int        _weights_to_use_idx{ ACL_SRC_1 };

    bool _are_weights_converted{ true };
    bool _are_weights_reshaped{ true };
    bool _is_fc_after_conv{ true };
    bool _is_quantized{ false };
    bool _is_prepared{ false };
    bool _dynamic_weights{ false };

#ifdef ARM_COMPUTE_ASSERTS_ENABLED
    int  _asrt_run_count{};
    int  _asrt_prepare_count{};
#endif // ARM_COMPUTE_ASSERTS_ENABLED
};
} // namespace opencl
} // namespace arm_compute
#endif /* ARM_COMPUTE_CL_FULLY_CONNECTED_H */
