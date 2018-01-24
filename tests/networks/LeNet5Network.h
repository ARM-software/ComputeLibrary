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
#ifndef __ARM_COMPUTE_TEST_MODEL_OBJECTS_LENET5_H__
#define __ARM_COMPUTE_TEST_MODEL_OBJECTS_LENET5_H__

#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/Utils.h"

#include <memory>

using namespace arm_compute;
using namespace arm_compute::test;

namespace arm_compute
{
namespace test
{
namespace networks
{
/** Lenet5 model object */
template <typename TensorType,
          typename Accessor,
          typename ActivationLayerFunction,
          typename ConvolutionLayerFunction,
          typename FullyConnectedLayerFunction,
          typename PoolingLayerFunction,
          typename SoftmaxLayerFunction>
class LeNet5Network
{
public:
    void init(int batches)
    {
        _batches = batches;

        // Initialize input, output, weights and biases
        input.allocator()->init(TensorInfo(TensorShape(28U, 28U, 1U, _batches), 1, DataType::F32));
        output.allocator()->init(TensorInfo(TensorShape(10U, _batches), 1, DataType::F32));
        w[0].allocator()->init(TensorInfo(TensorShape(5U, 5U, 1U, 20U), 1, DataType::F32));
        b[0].allocator()->init(TensorInfo(TensorShape(20U), 1, DataType::F32));
        w[1].allocator()->init(TensorInfo(TensorShape(5U, 5U, 20U, 50U), 1, DataType::F32));
        b[1].allocator()->init(TensorInfo(TensorShape(50U), 1, DataType::F32));
        w[2].allocator()->init(TensorInfo(TensorShape(800U, 500U), 1, DataType::F32));
        b[2].allocator()->init(TensorInfo(TensorShape(500U), 1, DataType::F32));
        w[3].allocator()->init(TensorInfo(TensorShape(500U, 10U), 1, DataType::F32));
        b[3].allocator()->init(TensorInfo(TensorShape(10U), 1, DataType::F32));
    }

    /** Build the model. */
    void build()
    {
        // Initialize intermediate tensors
        // Layer 1
        conv1_out.allocator()->init(TensorInfo(TensorShape(24U, 24U, 20U, _batches), 1, DataType::F32));
        pool1_out.allocator()->init(TensorInfo(TensorShape(12U, 12U, 20U, _batches), 1, DataType::F32));
        // Layer 2
        conv2_out.allocator()->init(TensorInfo(TensorShape(8U, 8U, 50U, _batches), 1, DataType::F32));
        pool2_out.allocator()->init(TensorInfo(TensorShape(4U, 4U, 50U, _batches), 1, DataType::F32));
        // Layer 3
        fc1_out.allocator()->init(TensorInfo(TensorShape(500U, _batches), 1, DataType::F32));
        act1_out.allocator()->init(TensorInfo(TensorShape(500U, _batches), 1, DataType::F32));
        // Layer 6
        fc2_out.allocator()->init(TensorInfo(TensorShape(10U, _batches), 1, DataType::F32));

        // Configure Layers
        conv1.configure(&input, &w[0], &b[0], &conv1_out, PadStrideInfo(1, 1, 0, 0));
        pool1.configure(&conv1_out, &pool1_out, PoolingLayerInfo(PoolingType::MAX, 2, PadStrideInfo(2, 2, 0, 0)));
        conv2.configure(&pool1_out, &w[1], &b[1], &conv2_out, PadStrideInfo(1, 1, 0, 0));
        pool2.configure(&conv2_out, &pool2_out, PoolingLayerInfo(PoolingType::MAX, 2, PadStrideInfo(2, 2, 0, 0)));
        fc1.configure(&pool2_out, &w[2], &b[2], &fc1_out);
        act1.configure(&fc1_out, &act1_out, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        fc2.configure(&act1_out, &w[3], &b[3], &fc2_out);
        smx.configure(&fc2_out, &output);
    }

    void allocate()
    {
        // Allocate tensors
        input.allocator()->allocate();
        output.allocator()->allocate();
        for(auto &wi : w)
        {
            wi.allocator()->allocate();
        }
        for(auto &bi : b)
        {
            bi.allocator()->allocate();
        }
        conv1_out.allocator()->allocate();
        pool1_out.allocator()->allocate();
        conv2_out.allocator()->allocate();
        pool2_out.allocator()->allocate();
        fc1_out.allocator()->allocate();
        act1_out.allocator()->allocate();
        fc2_out.allocator()->allocate();
    }

    /** Fills the trainable parameters and input with random data. */
    void fill_random()
    {
        std::uniform_real_distribution<> distribution(-1, 1);
        library->fill(Accessor(input), distribution, 0);
        for(unsigned int i = 0; i < w.size(); ++i)
        {
            library->fill(Accessor(w[i]), distribution, i + 1);
            library->fill(Accessor(b[i]), distribution, i + 10);
        }
    }

    /** Fills the trainable parameters from binary files
     *
     * @param weights Files names containing the weights data
     * @param biases  Files names containing the bias data
     */
    void fill(std::vector<std::string> weights, std::vector<std::string> biases)
    {
        ARM_COMPUTE_ERROR_ON(weights.size() != w.size());
        ARM_COMPUTE_ERROR_ON(biases.size() != b.size());

        for(unsigned int i = 0; i < weights.size(); ++i)
        {
            library->fill_layer_data(Accessor(w[i]), weights[i]);
            library->fill_layer_data(Accessor(b[i]), biases[i]);
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
        for(auto &wi : w)
        {
            wi.allocator()->free();
        }
        for(auto &bi : b)
        {
            bi.allocator()->free();
        }

        conv1_out.allocator()->free();
        pool1_out.allocator()->free();
        conv2_out.allocator()->free();
        pool2_out.allocator()->free();
        fc1_out.allocator()->free();
        act1_out.allocator()->free();
        fc2_out.allocator()->free();
    }

    /** Runs the model */
    void run()
    {
        // Layer 1
        conv1.run();
        pool1.run();
        // Layer 2
        conv2.run();
        pool2.run();
        // Layer 3
        fc1.run();
        act1.run();
        // Layer 4
        fc2.run();
        // Softmax
        smx.run();
    }

    /** Sync the results */
    void sync()
    {
        sync_if_necessary<TensorType>();
        sync_tensor_if_necessary<TensorType>(output);
    }

private:
    unsigned int _batches{ 0 };

    ActivationLayerFunction     act1{};
    ConvolutionLayerFunction    conv1{}, conv2{};
    FullyConnectedLayerFunction fc1{}, fc2{};
    PoolingLayerFunction        pool1{}, pool2{};
    SoftmaxLayerFunction        smx{};

    TensorType input{}, output{};
    std::array<TensorType, 4> w{ {} }, b{ {} };

    TensorType conv1_out{}, pool1_out{};
    TensorType conv2_out{}, pool2_out{};
    TensorType fc1_out{}, act1_out{};
    TensorType fc2_out{};
};
} // namespace networks
} // namespace test
} // namespace arm_compute
#endif //__ARM_COMPUTE_TEST_MODEL_OBJECTS_LENET5_H__
