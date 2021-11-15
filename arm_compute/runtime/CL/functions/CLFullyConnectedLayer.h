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
#ifndef ARM_COMPUTE_CLFULLYCONNECTEDLAYER_H
#define ARM_COMPUTE_CLFULLYCONNECTEDLAYER_H

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/IWeightsManager.h"
#include "arm_compute/runtime/MemoryGroup.h"

namespace arm_compute
{
/** Basic function to compute a Fully Connected layer on OpenCL. This function calls the following OpenCL kernels:
 *
 *  -# @ref opencl::kernels::ClIm2ColKernel (called when the input comes from a convolutional layer)
 *  -# @ref CLTranspose (if @p are_weights_reshaped is set to false and transpose_weights is set to true ) (called once)
 *  -# @ref opencl::ClGemm or @ref CLGEMMLowpMatrixMultiplyCore (if quantized asymmetric)
 *
 * @note  The fully connected layer accepts "weights" tensors only with 2 dimensions.
 */
class CLFullyConnectedLayer : public IFunction
{
public:
    /** Constructor */
    CLFullyConnectedLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr, IWeightsManager *weights_manager = nullptr);
    /** Default destructor */
    ~CLFullyConnectedLayer();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLFullyConnectedLayer(const CLFullyConnectedLayer &) = delete;
    /** Default move constructor */
    CLFullyConnectedLayer(CLFullyConnectedLayer &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLFullyConnectedLayer &operator=(const CLFullyConnectedLayer &) = delete;
    /** Default move assignment operator */
    CLFullyConnectedLayer &operator=(CLFullyConnectedLayer &&) = default;
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
     * @param[in]  input           Source tensor. Data type supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in]  weights         Weights tensor. The weights must be 2 dimensional.
     *                             If this function is called after a Convolution Layer, the (transposed) weights will have as many rows as the product of the first 3 input's dimensions.
     *                             If it is called after another FullyConnected Layer, the (transposed) weights will have as many rows as the input's first dimension.
     *                             Data type supported: Same as @p input.
     * @param[in]  biases          Bias tensor. Can be nullptr. Data type supported:Same as @p input.
     * @param[out] output          Destination tensor. Its shape should be equal to the output of a matrix multiplication between:
     *                             - The output of im2col on the input and the (transposed) 2D weights, if the function is called after a Convolution Layer
     *                             - The input tensor and the (transposed) 2D weights, if the function is called after another FullyConnected Layer.
     *                             Data type supported: Same as @p input.
     * @param[in]  fc_info         (Optional) Fully connected layer additional info
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output,
                   FullyConnectedLayerInfo fc_info = FullyConnectedLayerInfo());
    /** Set the input and output tensors.
     *
     * Similar to @ref CLFullyConnectedLayer
     */
    void configure(const ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output,
                   FullyConnectedLayerInfo fc_info = FullyConnectedLayerInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref CLFullyConnectedLayer
     *
     * Similar to @ref CLFullyConnectedLayer
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output,
                           FullyConnectedLayerInfo fc_info = FullyConnectedLayerInfo());

    //Inherited methods override
    void run() override;
    void prepare() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_CLFULLYCONNECTEDLAYER_H */
