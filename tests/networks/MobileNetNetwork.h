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
#ifndef __ARM_COMPUTE_TEST_MODEL_OBJECTS_MOBILENET_H__
#define __ARM_COMPUTE_TEST_MODEL_OBJECTS_MOBILENET_H__

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
          typename ConvolutionLayerFunction,
          typename DirectConvolutionLayerFunction,
          typename DepthwiseConvolutionLayerFunction,
          typename ReshapeFunction,
          typename PoolingLayerFunction>
class MobileNetNetwork
{
public:
    void init(int batches)
    {
        _batches = batches;

        // Initialize input, output
        input.allocator()->init(TensorInfo(TensorShape(224U, 224U, 3U, _batches), 1, DataType::F32));
        output.allocator()->init(TensorInfo(TensorShape(11U, _batches), 1, DataType::F32));
        // Initialize weights and biases
        w_conv3x3.allocator()->init(TensorInfo(TensorShape(3U, 3U, 3U, 16U), 1, DataType::F32));
        b_conv3x3.allocator()->init(TensorInfo(TensorShape(16U), 1, DataType::F32));
        depthwise_conv_block_init(0, 16, 16);
        depthwise_conv_block_init(1, 16, 32);
        depthwise_conv_block_init(2, 32, 32);
        depthwise_conv_block_init(3, 32, 64);
        depthwise_conv_block_init(4, 64, 64);
        depthwise_conv_block_init(5, 64, 128);
        depthwise_conv_block_init(6, 128, 128);
        depthwise_conv_block_init(7, 128, 128);
        depthwise_conv_block_init(8, 128, 128);
        depthwise_conv_block_init(9, 128, 128);
        depthwise_conv_block_init(10, 128, 128);
        depthwise_conv_block_init(11, 128, 256);
        depthwise_conv_block_init(12, 256, 256);
        w_conv[13].allocator()->init(TensorInfo(TensorShape(1U, 1U, 256U, 11U), 1, DataType::F32));
        b_conv[13].allocator()->init(TensorInfo(TensorShape(11U), 1, DataType::F32));
    }

    /** Build the model. */
    void build()
    {
        // Configure Layers
        conv3x3.configure(&input, &w_conv3x3, &b_conv3x3, &conv_out[0], PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR));
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
        pool.configure(&conv_out[13], &pool_out, PoolingLayerInfo(PoolingType::AVG, 7, PadStrideInfo(2, 2, 0, 0)));
        conv1x1[13].configure(&pool_out, &w_conv[13], &b_conv[13], &conv_out[14], PadStrideInfo(1, 1, 0, 0));
        logistic.configure(&conv_out[14], nullptr, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC));
        reshape.configure(&conv_out[14], &output);
    }

    void allocate()
    {
        input.allocator()->allocate();
        output.allocator()->allocate();

        w_conv3x3.allocator()->allocate();
        b_conv3x3.allocator()->allocate();
        for(unsigned int i = 0; i < w_conv.size(); ++i)
        {
            w_conv[i].allocator()->allocate();
            b_conv[i].allocator()->allocate();
        }
        for(unsigned int i = 0; i < w_dwc.size(); ++i)
        {
            w_dwc[i].allocator()->allocate();
            b_dwc[i].allocator()->allocate();
        }
        for(auto &o : conv_out)
        {
            o.allocator()->allocate();
        }
        for(auto &o : dwc_out)
        {
            o.allocator()->allocate();
        }
        pool_out.allocator()->allocate();
    }

    /** Fills the trainable parameters and input with random data. */
    void fill_random()
    {
        unsigned int                     seed_idx = 0;
        std::uniform_real_distribution<> distribution(-1, 1);
        library->fill(Accessor(input), distribution, seed_idx++);

        library->fill(Accessor(w_conv3x3), distribution, seed_idx++);
        library->fill(Accessor(b_conv3x3), distribution, seed_idx++);
        for(unsigned int i = 0; i < w_conv.size(); ++i)
        {
            library->fill(Accessor(w_conv[i]), distribution, seed_idx++);
            library->fill(Accessor(b_conv[i]), distribution, seed_idx++);
        }
        for(unsigned int i = 0; i < w_dwc.size(); ++i)
        {
            library->fill(Accessor(w_dwc[i]), distribution, seed_idx++);
            library->fill(Accessor(b_dwc[i]), distribution, seed_idx++);
        }
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
        b_conv3x3.allocator()->free();
        for(unsigned int i = 0; i < w_conv.size(); ++i)
        {
            w_conv[i].allocator()->free();
            b_conv[i].allocator()->free();
        }
        for(unsigned int i = 0; i < w_dwc.size(); ++i)
        {
            w_dwc[i].allocator()->free();
            b_dwc[i].allocator()->free();
        }
        for(auto &o : conv_out)
        {
            o.allocator()->free();
        }
        for(auto &o : dwc_out)
        {
            o.allocator()->free();
        }
        pool_out.allocator()->free();
    }

    /** Runs the model */
    void run()
    {
        conv3x3.run();
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
        conv1x1[13].run();
        logistic.run();
        reshape.run();
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
        w_dwc[idx].allocator()->init(TensorInfo(TensorShape(3U, 3U, ifm), 1, DataType::F32));
        b_dwc[idx].allocator()->init(TensorInfo(TensorShape(ifm), 1, DataType::F32));
        w_conv[idx].allocator()->init(TensorInfo(TensorShape(1U, 1U, ifm, ofm), 1, DataType::F32));
        b_conv[idx].allocator()->init(TensorInfo(TensorShape(ofm), 1, DataType::F32));
    }
    void depthwise_conv_block_build(unsigned int idx, PadStrideInfo dwc_ps, PadStrideInfo conv_ps)
    {
        dwc3x3[idx].configure(&conv_out[idx], &w_dwc[idx], &b_dwc[idx], &dwc_out[idx], dwc_ps);
        act[2 * idx].configure(&dwc_out[idx], nullptr, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f));
        conv1x1[idx].configure(&dwc_out[idx], &w_conv[idx], &b_conv[idx], &conv_out[idx + 1], conv_ps);
        act[2 * idx + 1].configure(&conv_out[idx], nullptr, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f));
    }
    void depthwise_conv_block_run(unsigned int idx)
    {
        dwc3x3[idx].run();
        act[2 * idx].run();
        conv1x1[idx].run();
        act[2 * idx + 1].run();
    }

private:
    unsigned int _batches{ 0 };

    ConvolutionLayerFunction conv3x3{};
    ActivationLayerFunction  conv3x3_act{};
    std::array<ActivationLayerFunction, 26>           act{ {} };
    std::array<DirectConvolutionLayerFunction, 14>    conv1x1{ {} };
    std::array<DepthwiseConvolutionLayerFunction, 13> dwc3x3{ {} };
    PoolingLayerFunction    pool{};
    ActivationLayerFunction logistic{};
    ReshapeFunction         reshape{};

    TensorType w_conv3x3{}, b_conv3x3{};
    std::array<TensorType, 14> w_conv{ {} }, b_conv{ {} };
    std::array<TensorType, 13> w_dwc{ {} }, b_dwc{ {} };

    TensorType input{}, output{};

    std::array<TensorType, 15> conv_out{ {} };
    std::array<TensorType, 13> dwc_out{ {} };
    TensorType pool_out{};
};
} // namespace networks
} // namespace test
} // namespace arm_compute
#endif //__ARM_COMPUTE_TEST_MODEL_OBJECTS_MOBILENET_H__
