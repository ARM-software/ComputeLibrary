/*
 * Copyright (c) 2021-2023 Arm Limited.
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
#ifndef ARM_COMPUTE_CPU_FULLY_CONNECTED_H
#define ARM_COMPUTE_CPU_FULLY_CONNECTED_H

#include "src/cpu/ICpuOperator.h"

#include "arm_compute/core/TensorInfo.h"

#include <memory>

namespace arm_compute
{
namespace cpu
{
// Forward declarations
class CpuConvertFullyConnectedWeights;
class CpuFlatten;
class CpuGemm;
class CpuGemmLowpMatrixMultiplyCore;
namespace kernels
{
class CpuTransposeKernel;
} // namespace kernels
/** Basic function to compute a Fully Connected layer. This function calls the following kernels:
 *  -# @ref kernels::CpuIm2ColKernel (called when the input comes from a convolutional layer)
 *  -# @ref kernels::CpuTransposeKernel (if @p are_weights_reshaped is set to false and transpose_weights is set to true ) (called once)
 *  -# @ref CpuGemm or @ref CpuGemmLowpMatrixMultiplyCore (if quantized asymmetric)
 *  -# @ref kernels::CpuGemmMatrixAdditionKernel or @ref CpuGemmLowpOutputStage (if quantized asymmetric) (if @p biases is not equal to nullptr)
 *
 * @note  The fully connected layer accepts "weights" tensors only with 2 dimensions.
 */
class CpuFullyConnected : public ICpuOperator
{
public:
    /** Constructor */
    CpuFullyConnected();
    /** Destructor */
    ~CpuFullyConnected();
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
     * @param[in]  src          Source tensor info. Data type supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in]  weights      Weights tensor info. The weights must be 2 dimensional.
     *                          If this function is called after a Convolution Layer, the (transposed) weights will have as many rows as the product of the first 3 input's dimensions.
     *                          If it is called after another FullyConnected Layer, the (transposed) weights will have as many rows as the input's first dimension.
     *                          Data type supported: Same as @p src.
     * @param[in]  biases       Bias tensor info. Can be nullptr. Data type supported: Same as @p weights, S32 if @p weights is QASYMM8/QASYMM8_SIGNED.
     * @param[out] dst          Destination tensor info. Its shape should be equal to the output of a matrix multiplication between:
     *                          - The output of im2col on the input and the (transposed) 2D weights, if the function is called after a Convolution Layer
     *                          - The input tensor and the (transposed) 2D weights, if the function is called after another FullyConnected Layer.
     *                          Data type supported: Same as @p src.
     * @param[in]  fc_info      (Optional) Fully connected layer additional info
     * @param[in]  weights_info (Optional) Stores neccessary compute information when weights are already reshaped
     */
    void configure(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, ITensorInfo *dst,
                   FullyConnectedLayerInfo fc_info = FullyConnectedLayerInfo(), const WeightsInfo &weights_info = WeightsInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref CpuFullyConnected
     *
     * Similar to @ref CpuFullyConnected::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *dst,
                           FullyConnectedLayerInfo fc_info = FullyConnectedLayerInfo(), const WeightsInfo &weights_info = WeightsInfo());

    /** Static function that queries whether there exists fixed-format kernel and if it exists it will return in the first argument in what format
     * weights are expected to be reshaped as defined by WeightFormat class. Apart from the first argument the rest of the arguments are the same
     * as in @ref CpuFullyConnectedLayer::validate() except that all arguments are required.
     *
     * @return a status
     */
    static Status has_opt_impl(arm_compute::WeightFormat &expected_weight_format, const ITensorInfo *src, const ITensorInfo *weights,
                               const ITensorInfo *biases, const ITensorInfo *dst,
                               FullyConnectedLayerInfo fc_info, WeightsInfo weights_info);

    //Inherited methods override
    void                             run(ITensorPack &tensors) override;
    void                             prepare(ITensorPack &tensors) override;
    experimental::MemoryRequirements workspace() const override;

private:
    void configure_fc_fc(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, ITensorInfo *dst, const ActivationLayerInfo &act);
    void configure_conv_fc(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, ITensorInfo *dst, const ActivationLayerInfo &act);
    void configure_mm(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, ITensorInfo *dst, const ActivationLayerInfo &act);

    enum AuxTensorIdx
    {
        AsmGemmWorkspace = 0,
        Pretranspose,
        GemmTemp1, // Both CpuGemm and CpuGemmLowpMatrixMultiplyCore
        GemmTemp2, // Both CpuGemm and CpuGemmLowpMatrixMultiplyCore
        GemmTemp3, // Both CpuGemm and CpuGemmLowpMatrixMultiplyCore
        GemmTemp4, // CpuGemmLowpMatrixMultiplyCore only
        GemmTemp5, // CpuGemmLowpMatrixMultiplyCore only
        GemmTemp6, // CpuGemmLowpMatrixMultiplyCore only
        GemmTemp7, // CpuGemmLowpMatrixMultiplyCore only
        TransposedWeights,
        ConvertedWeights,
        FlattenedSrc,
        Count
    };

    std::unique_ptr<CpuFlatten>                      _flatten;
    std::unique_ptr<CpuConvertFullyConnectedWeights> _convert_weights;
    std::unique_ptr<kernels::CpuTransposeKernel>     _transpose_weights;
    std::unique_ptr<CpuGemm>                         _mm_gemm;
    std::unique_ptr<CpuGemmLowpMatrixMultiplyCore>   _mm_gemmlowp;

    TensorInfo   _flattened_src;
    TensorInfo   _converted_weights;
    TensorInfo   _reshaped_weights;
    TensorInfo   _trans_weights;
    AuxTensorIdx _trans_weights_idx;

    experimental::MemoryRequirements _aux_mem;

    bool                      _needs_weights_conversion;
    bool                      _needs_weights_reshape;
    bool                      _is_fc_after_conv;
    bool                      _is_quantized_asymmetric;
    bool                      _is_prepared;
    bool                      _enable_fast_math;
    bool                      _fixed_format;
    arm_compute::WeightFormat _weight_format;
    bool                      _dynamic_weights;

#ifdef ARM_COMPUTE_ASSERTS_ENABLED
    int                       _asrt_run_count{};
    int                       _asrt_prepare_count{};
#endif // ARM_COMPUTE_ASSERTS_ENABLED
};
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_FULLY_CONNECTED_H */
