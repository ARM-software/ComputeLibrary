/*
 * Copyright (c) 2017-2023 Arm Limited.
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
#ifndef ARM_COMPUTE_NEFULLYCONNECTEDLAYER_H
#define ARM_COMPUTE_NEFULLYCONNECTEDLAYER_H

#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/IWeightsManager.h"

#include "arm_compute/runtime/NEON/functions/NETranspose.h"
#include "arm_compute/runtime/Tensor.h"

#include <memory>

namespace arm_compute
{
namespace weights_transformations
{
/** Basic function to manage the reshape weights generated from @ref NETranspose */
class NEFullyConnectedLayerReshapeWeightsManaged : public ITransformWeights
{
public:
    void run() override
    {
        _output.allocator()->allocate();
        _func.run();
        _reshape_run = true;
    }

    void release() override
    {
        _output.allocator()->free();
    }

    ITensor *get_weights() override
    {
        return &_output;
    }

    uint32_t uid() override
    {
        return _uid;
    }

    void configure(const ITensor *input)
    {
        _func.configure(input, &_output);
    }

private:
    static constexpr uint32_t _uid = 0x0;
    Tensor                    _output{};
    NETranspose               _func{};
};
} // namespace weights_transformations

/** Basic function to compute a Fully Connected layer. This function calls the following kernels:
 *  -# @ref cpu::kernels::CpuIm2ColKernel (called when the input comes from a convolutional layer)
 *  -# @ref NETranspose (if @p are_weights_reshaped is set to false and transpose_weights is set to true ) (called once)
 *  -# @ref NEGEMM or @ref NEGEMMLowpMatrixMultiplyCore (if quantized asymmetric)
 *  -# @ref cpu::kernels::CpuGemmMatrixAdditionKernel or @ref NEGEMMLowpOutputStage (if quantized asymmetric) (if @p biases is not equal to nullptr)
 *
 * @note  The fully connected layer accepts "weights" tensors only with 2 dimensions.
 */
class NEFullyConnectedLayer : public IFunction
{
public:
    /** Constructor */
    NEFullyConnectedLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr, IWeightsManager *weights_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEFullyConnectedLayer(const NEFullyConnectedLayer &) = delete;
    /** Prevent instances of this class from being moved (As this class contains pointers) */
    NEFullyConnectedLayer(NEFullyConnectedLayer &&) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEFullyConnectedLayer &operator=(const NEFullyConnectedLayer &) = delete;
    /** Prevent instances of this class from being moved (As this class contains pointers) */
    NEFullyConnectedLayer &operator=(NEFullyConnectedLayer &&) = delete;
    /** Default destructor */
    ~NEFullyConnectedLayer();
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
     * @param[in]  input        Source tensor. Data type supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in]  weights      Weights tensor. The weights must be 2 dimensional.
     *                          If this function is called after a Convolution Layer, the (transposed) weights will have as many rows as the product of the first 3 input's dimensions.
     *                          If it is called after another FullyConnected Layer, the (transposed) weights will have as many rows as the input's first dimension.
     *                          Data type supported: Same as @p input.
     * @param[in]  biases       Bias tensor. Can be nullptr. Data type supported: Same as @p weights, S32 if @p weights is QASYMM8/QASYMM8_SIGNED.
     * @param[out] output       Destination tensor. Its shape should be equal to the output of a matrix multiplication between:
     *                          - The output of im2col on the input and the (transposed) 2D weights, if the function is called after a Convolution Layer
     *                          - The input tensor and the (transposed) 2D weights, if the function is called after another FullyConnected Layer.
     *                          Data type supported: Same as @p input.
     * @param[in]  fc_info      (Optional) Fully connected layer additional info
     * @param[in]  weights_info (Optional) Stores neccessary compute information when weights are already reshaped
     */
    void configure(const ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output,
                   FullyConnectedLayerInfo fc_info = FullyConnectedLayerInfo(), const WeightsInfo &weights_info = WeightsInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref NEFullyConnectedLayer
     *
     * Similar to @ref NEFullyConnectedLayer::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output,
                           FullyConnectedLayerInfo fc_info = FullyConnectedLayerInfo(), const WeightsInfo &weights_info = WeightsInfo());

    /** Static function that queries whether fixed-format kernel exists for a given problem description
     *
     * @param[out] expected_weight_format Format in which weights should be for found fixed format kernel
     * @param[in]  input                  Source tensor
     * @param[in]  weights                Weights tensor.
     * @param[in]  biases                 Bias tensor. Can be nullptr. Data type supported: Same as @p weights, S32 if @p weights is QASYMM8/QASYMM8_SIGNED.
     * @param[in]  output                 Destination tensor
     * @param[in]  fc_info                Fully connected layer additional info
     * @param[in]  weights_info           Describes weights shape
     *
     * @return a status
     */
    static Status has_opt_impl(arm_compute::WeightFormat &expected_weight_format, const ITensorInfo *input, const ITensorInfo *weights,
                               const ITensorInfo *biases, const ITensorInfo *output, const FullyConnectedLayerInfo &fc_info, const WeightsInfo &weights_info);

    //Inherited methods override
    void run() override;
    void prepare() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_NEFULLYCONNECTEDLAYER_H */
