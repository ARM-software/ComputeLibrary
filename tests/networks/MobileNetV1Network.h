/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_TEST_MODEL_OBJECTS_MOBILENETV1_H__
#define __ARM_COMPUTE_TEST_MODEL_OBJECTS_MOBILENETV1_H__

#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/Utils.h"

#include "utils/Utils.h"

#include <memory>

using namespace arm_compute;
using namespace arm_compute::test;

namespace arm_compute
{
namespace test
{
namespace networks
{
/** MobileNet model object */
template <typename TensorType,
          typename Accessor,
          typename ActivationLayerFunction,
          typename BatchNormalizationLayerFunction,
          typename ConvolutionLayerFunction,
          typename DirectConvolutionLayerFunction,
          typename DepthwiseConvolutionFunction,
          typename ReshapeFunction,
          typename PoolingLayerFunction,
          typename SoftmaxLayerFunction>
class MobileNetV1Network
{
public:
    void init(unsigned int input_spatial_size, int batches)
    {
        _batches            = batches;
        _input_spatial_size = input_spatial_size;

        // Currently supported sizes
        ARM_COMPUTE_ERROR_ON(input_spatial_size != 128 && input_spatial_size != 224);

        // Initialize input, output
        input.allocator()->init(TensorInfo(TensorShape(input_spatial_size, input_spatial_size, 3U, _batches), 1, DataType::F32));
        output.allocator()->init(TensorInfo(TensorShape(1001U, _batches), 1, DataType::F32));
        // Initialize weights and biases
        w_conv3x3.allocator()->init(TensorInfo(TensorShape(3U, 3U, 3U, 32U), 1, DataType::F32));
        mean_conv3x3.allocator()->init(TensorInfo(TensorShape(32U), 1, DataType::F32));
        var_conv3x3.allocator()->init(TensorInfo(TensorShape(32U), 1, DataType::F32));
        beta_conv3x3.allocator()->init(TensorInfo(TensorShape(32U), 1, DataType::F32));
        gamma_conv3x3.allocator()->init(TensorInfo(TensorShape(32U), 1, DataType::F32));
        depthwise_conv_block_init(0, 32, 32);
        depthwise_conv_block_init(1, 32, 64);
        depthwise_conv_block_init(2, 64, 64);
        depthwise_conv_block_init(3, 64, 128);
        depthwise_conv_block_init(4, 128, 256);
        depthwise_conv_block_init(5, 256, 512);
        depthwise_conv_block_init(6, 512, 512);
        depthwise_conv_block_init(7, 512, 512);
        depthwise_conv_block_init(8, 512, 512);
        depthwise_conv_block_init(9, 512, 512);
        depthwise_conv_block_init(10, 512, 512);
        depthwise_conv_block_init(11, 512, 1024);
        depthwise_conv_block_init(12, 1024, 1024);
        w_conv1c.allocator()->init(TensorInfo(TensorShape(1U, 1U, 1024U, 1001U), 1, DataType::F32));
        b_conv1c.allocator()->init(TensorInfo(TensorShape(1001U), 1, DataType::F32));
        // Init reshaped output
        reshape_out.allocator()->init(TensorInfo(TensorShape(1001U, _batches), 1, DataType::F32));
    }

    /** Build the model. */
    void build()
    {
        // Configure Layers
        conv3x3.configure(&input, &w_conv3x3, nullptr, &conv_out[0], PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR));
        conv3x3_bn.configure(&conv_out[0], nullptr, &mean_conv3x3, &var_conv3x3, &beta_conv3x3, &gamma_conv3x3, 0.001f);
        conv3x3_act.configure(&conv_out[0], nullptr, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f));
        depthwise_conv_block_build(0, PadStrideInfo(1, 1, 1, 1), PadStrideInfo(1, 1, 0, 0));
        depthwise_conv_block_build(1, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0));
        depthwise_conv_block_build(2, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0));
        depthwise_conv_block_build(3, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0));
        depthwise_conv_block_build(4, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0));
        depthwise_conv_block_build(5, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0));
        depthwise_conv_block_build(6, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0));
        depthwise_conv_block_build(7, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0));
        depthwise_conv_block_build(8, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0));
        depthwise_conv_block_build(9, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0));
        depthwise_conv_block_build(10, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0));
        depthwise_conv_block_build(11, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0));
        depthwise_conv_block_build(12, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0));
        pool.configure(&conv_out[13], &pool_out, PoolingLayerInfo(PoolingType::AVG));
        conv1c.configure(&pool_out, &w_conv1c, &b_conv1c, &conv_out[14], PadStrideInfo(1, 1, 0, 0));
        reshape.configure(&conv_out[14], &reshape_out);
        smx.configure(&reshape_out, &output);
    }

    void allocate()
    {
        input.allocator()->allocate();
        output.allocator()->allocate();

        w_conv3x3.allocator()->allocate();
        mean_conv3x3.allocator()->allocate();
        var_conv3x3.allocator()->allocate();
        beta_conv3x3.allocator()->allocate();
        gamma_conv3x3.allocator()->allocate();

        ARM_COMPUTE_ERROR_ON(w_conv.size() != w_dwc.size());
        for(unsigned int i = 0; i < w_conv.size(); ++i)
        {
            w_dwc[i].allocator()->allocate();
            bn_mean[2 * i].allocator()->allocate();
            bn_var[2 * i].allocator()->allocate();
            bn_beta[2 * i].allocator()->allocate();
            bn_gamma[2 * i].allocator()->allocate();
            w_conv[i].allocator()->allocate();
            bn_mean[2 * i + 1].allocator()->allocate();
            bn_var[2 * i + 1].allocator()->allocate();
            bn_beta[2 * i + 1].allocator()->allocate();
            bn_gamma[2 * i + 1].allocator()->allocate();
        }
        w_conv1c.allocator()->allocate();
        b_conv1c.allocator()->allocate();

        // Allocate intermediate buffers
        for(auto &o : conv_out)
        {
            o.allocator()->allocate();
        }
        for(auto &o : dwc_out)
        {
            o.allocator()->allocate();
        }
        pool_out.allocator()->allocate();
        reshape_out.allocator()->allocate();
    }

    /** Fills the trainable parameters and input with random data. */
    void fill_random()
    {
        unsigned int                     seed_idx = 0;
        std::uniform_real_distribution<> distribution(-1, 1);
        library->fill(Accessor(input), distribution, seed_idx++);

        library->fill(Accessor(w_conv3x3), distribution, seed_idx++);
        library->fill(Accessor(mean_conv3x3), distribution, seed_idx++);
        library->fill(Accessor(var_conv3x3), distribution, seed_idx++);
        library->fill(Accessor(beta_conv3x3), distribution, seed_idx++);
        library->fill(Accessor(gamma_conv3x3), distribution, seed_idx++);

        ARM_COMPUTE_ERROR_ON(w_conv.size() != w_dwc.size());
        for(unsigned int i = 0; i < w_conv.size(); ++i)
        {
            library->fill(Accessor(w_dwc[i]), distribution, seed_idx++);
            library->fill(Accessor(bn_mean[2 * i]), distribution, seed_idx++);
            library->fill(Accessor(bn_var[2 * i]), distribution, seed_idx++);
            library->fill(Accessor(bn_beta[2 * i]), distribution, seed_idx++);
            library->fill(Accessor(bn_gamma[2 * i]), distribution, seed_idx++);
            library->fill(Accessor(w_conv[i]), distribution, seed_idx++);
            library->fill(Accessor(bn_mean[2 * i + 1]), distribution, seed_idx++);
            library->fill(Accessor(bn_var[2 * i + 1]), distribution, seed_idx++);
            library->fill(Accessor(bn_beta[2 * i + 1]), distribution, seed_idx++);
            library->fill(Accessor(bn_gamma[2 * i + 1]), distribution, seed_idx++);
        }
        library->fill(Accessor(w_conv1c), distribution, seed_idx++);
        library->fill(Accessor(b_conv1c), distribution, seed_idx++);
    }

    /** Feed input to network from file.
     *
     * @param name File name of containing the input data.
     */
    void feed(std::string name)
    {
        library->fill_layer_data(Accessor(input), name);
    }

    /** Get the classification results.
     *
     * @return Vector containing the classified labels
     */
    std::vector<unsigned int> get_classifications()
    {
        std::vector<unsigned int> classified_labels;
        Accessor                  output_accessor(output);

        Window window;
        window.set(Window::DimX, Window::Dimension(0, 1, 1));
        for(unsigned int d = 1; d < output_accessor.shape().num_dimensions(); ++d)
        {
            window.set(d, Window::Dimension(0, output_accessor.shape()[d], 1));
        }

        execute_window_loop(window, [&](const Coordinates & id)
        {
            int               max_idx = 0;
            float             val     = 0;
            const void *const out_ptr = output_accessor(id);
            for(unsigned int l = 0; l < output_accessor.shape().x(); ++l)
            {
                float curr_val = reinterpret_cast<const float *>(out_ptr)[l];
                if(curr_val > val)
                {
                    max_idx = l;
                    val     = curr_val;
                }
            }
            classified_labels.push_back(max_idx);
        });
        return classified_labels;
    }

    /** Clear all allocated memory from the tensor objects */
    void clear()
    {
        input.allocator()->free();
        output.allocator()->free();

        w_conv3x3.allocator()->free();
        mean_conv3x3.allocator()->free();
        var_conv3x3.allocator()->free();
        beta_conv3x3.allocator()->free();
        gamma_conv3x3.allocator()->free();

        ARM_COMPUTE_ERROR_ON(w_conv.size() != w_dwc.size());
        for(unsigned int i = 0; i < w_conv.size(); ++i)
        {
            w_dwc[i].allocator()->free();
            bn_mean[2 * i].allocator()->free();
            bn_var[2 * i].allocator()->free();
            bn_beta[2 * i].allocator()->free();
            bn_gamma[2 * i].allocator()->free();
            w_conv[i].allocator()->free();
            bn_mean[2 * i + 1].allocator()->free();
            bn_var[2 * i + 1].allocator()->free();
            bn_beta[2 * i + 1].allocator()->free();
            bn_gamma[2 * i + 1].allocator()->free();
        }
        w_conv1c.allocator()->free();
        b_conv1c.allocator()->free();

        // Free intermediate buffers
        for(auto &o : conv_out)
        {
            o.allocator()->free();
        }
        for(auto &o : dwc_out)
        {
            o.allocator()->free();
        }
        pool_out.allocator()->free();
        reshape_out.allocator()->free();
    }

    /** Runs the model */
    void run()
    {
        conv3x3.run();
        conv3x3_bn.run();
        conv3x3_act.run();
        depthwise_conv_block_run(0);
        depthwise_conv_block_run(1);
        depthwise_conv_block_run(2);
        depthwise_conv_block_run(3);
        depthwise_conv_block_run(4);
        depthwise_conv_block_run(5);
        depthwise_conv_block_run(6);
        depthwise_conv_block_run(7);
        depthwise_conv_block_run(8);
        depthwise_conv_block_run(9);
        depthwise_conv_block_run(10);
        depthwise_conv_block_run(11);
        depthwise_conv_block_run(12);
        pool.run();
        conv1c.run();
        reshape.run();
        smx.run();
    }

    /** Sync the results */
    void sync()
    {
        sync_if_necessary<TensorType>();
        sync_tensor_if_necessary<TensorType>(output);
    }

private:
    void depthwise_conv_block_init(unsigned int idx, unsigned int ifm, unsigned int ofm)
    {
        // Depthwise Convolution weights
        w_dwc[idx].allocator()->init(TensorInfo(TensorShape(3U, 3U, ifm), 1, DataType::F32));
        // Batch normalization parameters
        bn_mean[2 * idx].allocator()->init(TensorInfo(TensorShape(ifm), 1, DataType::F32));
        bn_var[2 * idx].allocator()->init(TensorInfo(TensorShape(ifm), 1, DataType::F32));
        bn_beta[2 * idx].allocator()->init(TensorInfo(TensorShape(ifm), 1, DataType::F32));
        bn_gamma[2 * idx].allocator()->init(TensorInfo(TensorShape(ifm), 1, DataType::F32));
        // Convolution weights
        w_conv[idx].allocator()->init(TensorInfo(TensorShape(1U, 1U, ifm, ofm), 1, DataType::F32));
        // Batch normalization parameters
        bn_mean[2 * idx + 1].allocator()->init(TensorInfo(TensorShape(ofm), 1, DataType::F32));
        bn_var[2 * idx + 1].allocator()->init(TensorInfo(TensorShape(ofm), 1, DataType::F32));
        bn_beta[2 * idx + 1].allocator()->init(TensorInfo(TensorShape(ofm), 1, DataType::F32));
        bn_gamma[2 * idx + 1].allocator()->init(TensorInfo(TensorShape(ofm), 1, DataType::F32));
    }
    void depthwise_conv_block_build(unsigned int idx, PadStrideInfo dwc_ps, PadStrideInfo conv_ps)
    {
        // Configure depthwise convolution block
        dwc3x3[idx].configure(&conv_out[idx], &w_dwc[idx], nullptr, &dwc_out[idx], dwc_ps);
        bn[2 * idx].configure(&dwc_out[idx], nullptr, &bn_mean[2 * idx], &bn_var[2 * idx], &bn_beta[2 * idx], &bn_gamma[2 * idx], 0.001f);
        act[2 * idx].configure(&dwc_out[idx], nullptr, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f));
        // Configure pointwise convolution block
        conv1x1[idx].configure(&dwc_out[idx], &w_conv[idx], nullptr, &conv_out[idx + 1], conv_ps);
        bn[2 * idx + 1].configure(&conv_out[idx + 1], nullptr, &bn_mean[2 * idx + 1], &bn_var[2 * idx + 1], &bn_beta[2 * idx + 1], &bn_gamma[2 * idx + 1], 0.001f);
        act[2 * idx + 1].configure(&conv_out[idx], nullptr, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f));
    }
    void depthwise_conv_block_run(unsigned int idx)
    {
        dwc3x3[idx].run();
        bn[2 * idx].run();
        act[2 * idx].run();
        conv1x1[idx].run();
        bn[2 * idx + 1].run();
        act[2 * idx + 1].run();
    }

private:
    unsigned int _batches{ 0 };
    unsigned int _input_spatial_size{ 0 };

    ConvolutionLayerFunction        conv3x3{};
    BatchNormalizationLayerFunction conv3x3_bn{};
    ActivationLayerFunction         conv3x3_act{};
    std::array<ActivationLayerFunction, 26>         act{ {} };
    std::array<BatchNormalizationLayerFunction, 26> bn{ {} };
    std::array<DepthwiseConvolutionFunction, 13>    dwc3x3{ {} };
    std::array<DirectConvolutionLayerFunction, 13>  conv1x1{ {} };
    DirectConvolutionLayerFunction conv1c{};
    PoolingLayerFunction           pool{};
    ReshapeFunction                reshape{};
    SoftmaxLayerFunction           smx{};

    TensorType w_conv3x3{}, mean_conv3x3{}, var_conv3x3{}, beta_conv3x3{}, gamma_conv3x3{};
    std::array<TensorType, 13> w_conv{ {} };
    std::array<TensorType, 13> w_dwc{ {} };
    std::array<TensorType, 26> bn_mean{ {} };
    std::array<TensorType, 26> bn_var{ {} };
    std::array<TensorType, 26> bn_beta{ {} };
    std::array<TensorType, 26> bn_gamma{ {} };
    TensorType w_conv1c{}, b_conv1c{};

    TensorType input{}, output{};

    std::array<TensorType, 15> conv_out{ {} };
    std::array<TensorType, 13> dwc_out{ {} };
    TensorType pool_out{};
    TensorType reshape_out{};
};
} // namespace networks
} // namespace test
} // namespace arm_compute
#endif //__ARM_COMPUTE_TEST_MODEL_OBJECTS_MOBILENETV1_H__
