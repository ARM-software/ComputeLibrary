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
#ifndef __ARM_COMPUTE_TEST_MODEL_OBJECTS_ALEXNET_H__
#define __ARM_COMPUTE_TEST_MODEL_OBJECTS_ALEXNET_H__

#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/Tensor.h"

#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/Utils.h"

#include <memory>

namespace arm_compute
{
namespace test
{
namespace networks
{
/** AlexNet model object */
template <typename ITensorType,
          typename TensorType,
          typename SubTensorType,
          typename Accessor,
          typename ActivationLayerFunction,
          typename ConvolutionLayerFunction,
          typename DirectConvolutionLayerFunction,
          typename FullyConnectedLayerFunction,
          typename NormalizationLayerFunction,
          typename PoolingLayerFunction,
          typename SoftmaxLayerFunction>
class AlexNetNetwork
{
public:
    void init(DataType data_type, int fixed_point_position, int batches, bool reshaped_weights = false)
    {
        _data_type            = data_type;
        _fixed_point_position = fixed_point_position;
        _batches              = batches;
        _reshaped_weights     = reshaped_weights;

        // Initialize weights and biases
        if(!_reshaped_weights)
        {
            w[0].allocator()->init(TensorInfo(TensorShape(11U, 11U, 3U, 96U), 1, _data_type, _fixed_point_position));
            b[0].allocator()->init(TensorInfo(TensorShape(96U), 1, _data_type, _fixed_point_position));
            w[1].allocator()->init(TensorInfo(TensorShape(5U, 5U, 48U, 256U), 1, _data_type, _fixed_point_position));
            b[1].allocator()->init(TensorInfo(TensorShape(256U), 1, _data_type, _fixed_point_position));
            w[2].allocator()->init(TensorInfo(TensorShape(3U, 3U, 256U, 384U), 1, _data_type, _fixed_point_position));
            b[2].allocator()->init(TensorInfo(TensorShape(384U), 1, _data_type, _fixed_point_position));
            w[3].allocator()->init(TensorInfo(TensorShape(3U, 3U, 192U, 384U), 1, _data_type, _fixed_point_position));
            b[3].allocator()->init(TensorInfo(TensorShape(384U), 1, _data_type, _fixed_point_position));
            w[4].allocator()->init(TensorInfo(TensorShape(3U, 3U, 192U, 256U), 1, _data_type, _fixed_point_position));
            b[4].allocator()->init(TensorInfo(TensorShape(256U), 1, _data_type, _fixed_point_position));
            w[5].allocator()->init(TensorInfo(TensorShape(9216U, 4096U), 1, _data_type, _fixed_point_position));
            b[5].allocator()->init(TensorInfo(TensorShape(4096U), 1, _data_type, _fixed_point_position));
            w[6].allocator()->init(TensorInfo(TensorShape(4096U, 4096U), 1, _data_type, _fixed_point_position));
            b[6].allocator()->init(TensorInfo(TensorShape(4096U), 1, _data_type, _fixed_point_position));
            w[7].allocator()->init(TensorInfo(TensorShape(4096U, 1000U), 1, _data_type, _fixed_point_position));
            b[7].allocator()->init(TensorInfo(TensorShape(1000U), 1, _data_type, _fixed_point_position));

            w11 = std::unique_ptr<SubTensorType>(new SubTensorType(&w[1], TensorShape(5U, 5U, 48U, 128U), Coordinates()));
            w12 = std::unique_ptr<SubTensorType>(new SubTensorType(&w[1], TensorShape(5U, 5U, 48U, 128U), Coordinates(0, 0, 0, 128)));
            b11 = std::unique_ptr<SubTensorType>(new SubTensorType(&b[1], TensorShape(128U), Coordinates()));
            b12 = std::unique_ptr<SubTensorType>(new SubTensorType(&b[1], TensorShape(128U), Coordinates(128)));

            w31 = std::unique_ptr<SubTensorType>(new SubTensorType(&w[3], TensorShape(3U, 3U, 192U, 192U), Coordinates()));
            w32 = std::unique_ptr<SubTensorType>(new SubTensorType(&w[3], TensorShape(3U, 3U, 192U, 192U), Coordinates(0, 0, 0, 192)));
            b31 = std::unique_ptr<SubTensorType>(new SubTensorType(&b[3], TensorShape(192U), Coordinates()));
            b32 = std::unique_ptr<SubTensorType>(new SubTensorType(&b[3], TensorShape(192U), Coordinates(192)));

            w41 = std::unique_ptr<SubTensorType>(new SubTensorType(&w[4], TensorShape(3U, 3U, 192U, 128U), Coordinates()));
            w42 = std::unique_ptr<SubTensorType>(new SubTensorType(&w[4], TensorShape(3U, 3U, 192U, 128U), Coordinates(0, 0, 0, 128)));
            b41 = std::unique_ptr<SubTensorType>(new SubTensorType(&b[4], TensorShape(128U), Coordinates()));
            b42 = std::unique_ptr<SubTensorType>(new SubTensorType(&b[4], TensorShape(128U), Coordinates(128)));
        }
        else
        {
            auto reshape = [&](unsigned int width, unsigned int height, bool convolution_layer) -> TensorShape
            {
                const bool is_optimised = std::is_same<ITensorType, ITensor>::value && NEScheduler::get().cpu_info().CPU >= CPUTarget::ARMV7 && data_type == DataType::F32;

                if(convolution_layer && is_optimised)
                {
                    return TensorShape{ height, width };
                }
                else
                {
                    const int interleave_width = 16 / arm_compute::data_size_from_type(_data_type);

                    return TensorShape{ width * interleave_width, static_cast<unsigned int>(std::ceil(static_cast<float>(height) / interleave_width)) };
                }
            };

            // Create tensor for the reshaped weights
            w[0].allocator()->init(TensorInfo(reshape(366U, 96U, true), 1, _data_type, _fixed_point_position));

            // Configure the direct convolution's weights. Direct convolution doesn't need reshape weights
            if(!_is_direct_conv)
            {
                auto w11_tensor = std::unique_ptr<TensorType>(new TensorType());
                auto w12_tensor = std::unique_ptr<TensorType>(new TensorType());
                auto w31_tensor = std::unique_ptr<TensorType>(new TensorType());
                auto w32_tensor = std::unique_ptr<TensorType>(new TensorType());
                auto w41_tensor = std::unique_ptr<TensorType>(new TensorType());
                auto w42_tensor = std::unique_ptr<TensorType>(new TensorType());
                w11_tensor->allocator()->init(TensorInfo(reshape(1248U, 128U, true), 1, _data_type, _fixed_point_position));
                w12_tensor->allocator()->init(TensorInfo(reshape(1248U, 128U, true), 1, _data_type, _fixed_point_position));
                w31_tensor->allocator()->init(TensorInfo(reshape(1920U, 192U, true), 1, _data_type, _fixed_point_position));
                w32_tensor->allocator()->init(TensorInfo(reshape(1920U, 192U, true), 1, _data_type, _fixed_point_position));
                w41_tensor->allocator()->init(TensorInfo(reshape(1920U, 128U, true), 1, _data_type, _fixed_point_position));
                w42_tensor->allocator()->init(TensorInfo(reshape(1920U, 128U, true), 1, _data_type, _fixed_point_position));
                w[2].allocator()->init(TensorInfo(reshape(2560U, 384U, true), 1, _data_type, _fixed_point_position));
                w11 = std::move(w11_tensor);
                w12 = std::move(w12_tensor);
                w31 = std::move(w31_tensor);
                w32 = std::move(w32_tensor);
                w41 = std::move(w41_tensor);
                w42 = std::move(w42_tensor);
            }
            else
            {
                w[1].allocator()->init(TensorInfo(TensorShape(5U, 5U, 48U, 256U), 1, _data_type, _fixed_point_position));
                b[1].allocator()->init(TensorInfo(TensorShape(256U), 1, _data_type, _fixed_point_position));
                w[2].allocator()->init(TensorInfo(TensorShape(3U, 3U, 256U, 384U), 1, _data_type, _fixed_point_position));
                b[2].allocator()->init(TensorInfo(TensorShape(384U), 1, _data_type, _fixed_point_position));
                w[3].allocator()->init(TensorInfo(TensorShape(3U, 3U, 192U, 384U), 1, _data_type, _fixed_point_position));
                b[3].allocator()->init(TensorInfo(TensorShape(384U), 1, _data_type, _fixed_point_position));
                w[4].allocator()->init(TensorInfo(TensorShape(3U, 3U, 192U, 256U), 1, _data_type, _fixed_point_position));
                b[4].allocator()->init(TensorInfo(TensorShape(256U), 1, _data_type, _fixed_point_position));
                w11 = std::unique_ptr<SubTensorType>(new SubTensorType(&w[1], TensorShape(5U, 5U, 48U, 128U), Coordinates()));
                w12 = std::unique_ptr<SubTensorType>(new SubTensorType(&w[1], TensorShape(5U, 5U, 48U, 128U), Coordinates(0, 0, 0, 128)));
                b11 = std::unique_ptr<SubTensorType>(new SubTensorType(&b[1], TensorShape(128U), Coordinates()));
                b12 = std::unique_ptr<SubTensorType>(new SubTensorType(&b[1], TensorShape(128U), Coordinates(128)));

                w31 = std::unique_ptr<SubTensorType>(new SubTensorType(&w[3], TensorShape(3U, 3U, 192U, 192U), Coordinates()));
                w32 = std::unique_ptr<SubTensorType>(new SubTensorType(&w[3], TensorShape(3U, 3U, 192U, 192U), Coordinates(0, 0, 0, 192)));
                b31 = std::unique_ptr<SubTensorType>(new SubTensorType(&b[3], TensorShape(192U), Coordinates()));
                b32 = std::unique_ptr<SubTensorType>(new SubTensorType(&b[3], TensorShape(192U), Coordinates(192)));

                w41 = std::unique_ptr<SubTensorType>(new SubTensorType(&w[4], TensorShape(3U, 3U, 192U, 128U), Coordinates()));
                w42 = std::unique_ptr<SubTensorType>(new SubTensorType(&w[4], TensorShape(3U, 3U, 192U, 128U), Coordinates(0, 0, 0, 128)));
                b41 = std::unique_ptr<SubTensorType>(new SubTensorType(&b[4], TensorShape(128U), Coordinates()));
                b42 = std::unique_ptr<SubTensorType>(new SubTensorType(&b[4], TensorShape(128U), Coordinates(128)));
            }

            b[5].allocator()->init(TensorInfo(TensorShape(4096U), 1, _data_type, _fixed_point_position));
            b[6].allocator()->init(TensorInfo(TensorShape(4096U), 1, _data_type, _fixed_point_position));
            b[7].allocator()->init(TensorInfo(TensorShape(1000U), 1, _data_type, _fixed_point_position));

            if(_batches > 1 && std::is_same<TensorType, Tensor>::value)
            {
                w[5].allocator()->init(TensorInfo(reshape(9216U, 4096U, false), 1, _data_type, _fixed_point_position));
                w[6].allocator()->init(TensorInfo(reshape(4096U, 4096U, false), 1, _data_type, _fixed_point_position));
                w[7].allocator()->init(TensorInfo(reshape(4096U, 1000U, false), 1, _data_type, _fixed_point_position));
            }
            else
            {
                w[5].allocator()->init(TensorInfo(TensorShape(4096U, 9216U), 1, _data_type, _fixed_point_position));
                w[6].allocator()->init(TensorInfo(TensorShape(4096U, 4096U), 1, _data_type, _fixed_point_position));
                w[7].allocator()->init(TensorInfo(TensorShape(1000U, 4096U), 1, _data_type, _fixed_point_position));
            }
        }
    }

    void build()
    {
        input.allocator()->init(TensorInfo(TensorShape(227U, 227U, 3U, _batches), 1, _data_type, _fixed_point_position));
        output.allocator()->init(TensorInfo(TensorShape(1000U, _batches), 1, _data_type, _fixed_point_position));

        // Initialize intermediate tensors
        // Layer 1
        conv1_out.allocator()->init(TensorInfo(TensorShape(55U, 55U, 96U, _batches), 1, _data_type, _fixed_point_position));
        act1_out.allocator()->init(TensorInfo(TensorShape(55U, 55U, 96U, _batches), 1, _data_type, _fixed_point_position));
        norm1_out.allocator()->init(TensorInfo(TensorShape(55U, 55U, 96U, _batches), 1, _data_type, _fixed_point_position));
        pool1_out.allocator()->init(TensorInfo(TensorShape(27U, 27U, 96U, _batches), 1, _data_type, _fixed_point_position));
        pool11_out = std::unique_ptr<SubTensorType>(new SubTensorType(&pool1_out, TensorShape(27U, 27U, 48U, _batches), Coordinates()));
        pool12_out = std::unique_ptr<SubTensorType>(new SubTensorType(&pool1_out, TensorShape(27U, 27U, 48U, _batches), Coordinates(0, 0, 48)));
        // Layer 2
        conv2_out.allocator()->init(TensorInfo(TensorShape(27U, 27U, 256U, _batches), 1, _data_type, _fixed_point_position));
        conv21_out = std::unique_ptr<SubTensorType>(new SubTensorType(&conv2_out, TensorShape(27U, 27U, 128U, _batches), Coordinates()));
        conv22_out = std::unique_ptr<SubTensorType>(new SubTensorType(&conv2_out, TensorShape(27U, 27U, 128U, _batches), Coordinates(0, 0, 128)));
        act2_out.allocator()->init(TensorInfo(TensorShape(27U, 27U, 256U, _batches), 1, _data_type, _fixed_point_position));
        norm2_out.allocator()->init(TensorInfo(TensorShape(27U, 27U, 256U, _batches), 1, _data_type, _fixed_point_position));
        pool2_out.allocator()->init(TensorInfo(TensorShape(13U, 13U, 256U, _batches), 1, _data_type, _fixed_point_position));
        // Layer 3
        conv3_out.allocator()->init(TensorInfo(TensorShape(13U, 13U, 384U, _batches), 1, _data_type, _fixed_point_position));
        act3_out.allocator()->init(TensorInfo(TensorShape(13U, 13U, 384U, _batches), 1, _data_type, _fixed_point_position));
        act31_out = std::unique_ptr<SubTensorType>(new SubTensorType(&act3_out, TensorShape(13U, 13U, 192U, _batches), Coordinates()));
        act32_out = std::unique_ptr<SubTensorType>(new SubTensorType(&act3_out, TensorShape(13U, 13U, 192U, _batches), Coordinates(0, 0, 192)));
        // Layer 4
        conv4_out.allocator()->init(TensorInfo(TensorShape(13U, 13U, 384U, _batches), 1, _data_type, _fixed_point_position));
        conv41_out = std::unique_ptr<SubTensorType>(new SubTensorType(&conv4_out, TensorShape(13U, 13U, 192U, _batches), Coordinates()));
        conv42_out = std::unique_ptr<SubTensorType>(new SubTensorType(&conv4_out, TensorShape(13U, 13U, 192U, _batches), Coordinates(0, 0, 192)));
        act4_out.allocator()->init(TensorInfo(TensorShape(13U, 13U, 384U, _batches), 1, _data_type, _fixed_point_position));
        act41_out = std::unique_ptr<SubTensorType>(new SubTensorType(&act4_out, TensorShape(13U, 13U, 192U, _batches), Coordinates()));
        act42_out = std::unique_ptr<SubTensorType>(new SubTensorType(&act4_out, TensorShape(13U, 13U, 192U, _batches), Coordinates(0, 0, 192)));
        // Layer 5
        conv5_out.allocator()->init(TensorInfo(TensorShape(13U, 13U, 256U, _batches), 1, _data_type, _fixed_point_position));
        conv51_out = std::unique_ptr<SubTensorType>(new SubTensorType(&conv5_out, TensorShape(13U, 13U, 128U, _batches), Coordinates()));
        conv52_out = std::unique_ptr<SubTensorType>(new SubTensorType(&conv5_out, TensorShape(13U, 13U, 128U, _batches), Coordinates(0, 0, 128)));
        act5_out.allocator()->init(TensorInfo(TensorShape(13U, 13U, 256U, _batches), 1, _data_type, _fixed_point_position));
        pool5_out.allocator()->init(TensorInfo(TensorShape(6U, 6U, 256U, _batches), 1, _data_type, _fixed_point_position));
        // Layer 6
        fc6_out.allocator()->init(TensorInfo(TensorShape(4096U, _batches), 1, _data_type, _fixed_point_position));
        act6_out.allocator()->init(TensorInfo(TensorShape(4096U, _batches), 1, _data_type, _fixed_point_position));
        // Layer 7
        fc7_out.allocator()->init(TensorInfo(TensorShape(4096U, _batches), 1, _data_type, _fixed_point_position));
        act7_out.allocator()->init(TensorInfo(TensorShape(4096U, _batches), 1, _data_type, _fixed_point_position));
        // Layer 8
        fc8_out.allocator()->init(TensorInfo(TensorShape(1000U, _batches), 1, _data_type, _fixed_point_position));

        // Configure Layers
        // Layer 1
        TensorType *b0 = _reshaped_weights ? nullptr : &b[0];
        conv1.configure(&input, &w[0], b0, &conv1_out, PadStrideInfo(4, 4, 0, 0), WeightsInfo(_reshaped_weights, 11U, 11U, 96U));
        act1.configure(&conv1_out, &act1_out, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        norm1.configure(&act1_out, &norm1_out, NormalizationLayerInfo(NormType::CROSS_MAP, 5, 0.0001f, 0.75f));
        pool1.configure(&norm1_out, &pool1_out, PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0)));
        // Layer 2
        conv21.configure(pool11_out.get(), w11.get(), b11.get(), conv21_out.get(), PadStrideInfo(1, 1, 2, 2), WeightsInfo(_reshaped_weights, 5U, 5U, 128U));
        conv22.configure(pool12_out.get(), w12.get(), b12.get(), conv22_out.get(), PadStrideInfo(1, 1, 2, 2), WeightsInfo(_reshaped_weights, 5U, 5U, 128U));
        act2.configure(&conv2_out, &act2_out, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        norm2.configure(&act2_out, &norm2_out, NormalizationLayerInfo(NormType::CROSS_MAP, 5, 0.0001f, 0.75f));
        pool2.configure(&norm2_out, &pool2_out, PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0)));
        // Layer 3
        TensorType *b2 = (_reshaped_weights && !_is_direct_conv) ? nullptr : &b[2];
        conv3.configure(&pool2_out, &w[2], b2, &conv3_out, PadStrideInfo(1, 1, 1, 1), WeightsInfo(_reshaped_weights, 3U, 3U, 384U));
        act3.configure(&conv3_out, &act3_out, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        // Layer 4
        conv41.configure(act31_out.get(), w31.get(), b31.get(), conv41_out.get(), PadStrideInfo(1, 1, 1, 1), WeightsInfo(_reshaped_weights, 3U, 3U, 192U));
        conv42.configure(act32_out.get(), w32.get(), b32.get(), conv42_out.get(), PadStrideInfo(1, 1, 1, 1), WeightsInfo(_reshaped_weights, 3U, 3U, 192U));
        act4.configure(&conv4_out, &act4_out, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        // Layer 5
        conv51.configure(act41_out.get(), w41.get(), b41.get(), conv51_out.get(), PadStrideInfo(1, 1, 1, 1), WeightsInfo(_reshaped_weights, 3U, 3U, 128U));
        conv52.configure(act42_out.get(), w42.get(), b42.get(), conv52_out.get(), PadStrideInfo(1, 1, 1, 1), WeightsInfo(_reshaped_weights, 3U, 3U, 128U));
        act5.configure(&conv5_out, &act5_out, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        pool5.configure(&act5_out, &pool5_out, PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0)));
        // Layer 6
        fc6.configure(&pool5_out, &w[5], &b[5], &fc6_out, true, _reshaped_weights);
        act6.configure(&fc6_out, &act6_out, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        // Layer 7
        fc7.configure(&act6_out, &w[6], &b[6], &fc7_out, true, _reshaped_weights);
        act7.configure(&fc7_out, &act7_out, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        // Layer 8
        fc8.configure(&act7_out, &w[7], &b[7], &fc8_out, true, _reshaped_weights);
        // Softmax
        smx.configure(&fc8_out, &output);
    }

    void allocate()
    {
        input.allocator()->allocate();
        output.allocator()->allocate();

        if(!_reshaped_weights)
        {
            for(auto &wi : w)
            {
                wi.allocator()->allocate();
            }

            for(auto &bi : b)
            {
                bi.allocator()->allocate();
            }
        }
        else
        {
            w[0].allocator()->allocate();
            w[2].allocator()->allocate();
            w[5].allocator()->allocate();
            w[6].allocator()->allocate();
            w[7].allocator()->allocate();

            b[5].allocator()->allocate();
            b[6].allocator()->allocate();
            b[7].allocator()->allocate();

            if(!_is_direct_conv)
            {
                dynamic_cast<TensorType *>(w11.get())->allocator()->allocate();
                dynamic_cast<TensorType *>(w12.get())->allocator()->allocate();
                dynamic_cast<TensorType *>(w31.get())->allocator()->allocate();
                dynamic_cast<TensorType *>(w32.get())->allocator()->allocate();
                dynamic_cast<TensorType *>(w41.get())->allocator()->allocate();
                dynamic_cast<TensorType *>(w42.get())->allocator()->allocate();
            }
            else
            {
                b[1].allocator()->allocate();
                b[2].allocator()->allocate();
                b[3].allocator()->allocate();
                b[4].allocator()->allocate();
                w[1].allocator()->allocate();
                w[3].allocator()->allocate();
                w[4].allocator()->allocate();
            }
        }

        conv1_out.allocator()->allocate();
        act1_out.allocator()->allocate();
        norm1_out.allocator()->allocate();
        pool1_out.allocator()->allocate();
        conv2_out.allocator()->allocate();
        act2_out.allocator()->allocate();
        norm2_out.allocator()->allocate();
        pool2_out.allocator()->allocate();
        conv3_out.allocator()->allocate();
        act3_out.allocator()->allocate();
        conv4_out.allocator()->allocate();
        act4_out.allocator()->allocate();
        conv5_out.allocator()->allocate();
        act5_out.allocator()->allocate();
        pool5_out.allocator()->allocate();
        fc6_out.allocator()->allocate();
        act6_out.allocator()->allocate();
        fc7_out.allocator()->allocate();
        act7_out.allocator()->allocate();
        fc8_out.allocator()->allocate();
    }

    /** Fills the trainable parameters and input with random data. */
    void fill_random()
    {
        library->fill_tensor_uniform(Accessor(input), 0);

        if(!_reshaped_weights)
        {
            for(unsigned int i = 0; i < w.size(); ++i)
            {
                library->fill_tensor_uniform(Accessor(w[i]), i + 1);
                library->fill_tensor_uniform(Accessor(b[i]), i + 10);
            }
        }
        else
        {
            library->fill_tensor_uniform(Accessor(w[0]), 1);
            library->fill_tensor_uniform(Accessor(w[2]), 2);

            library->fill_tensor_uniform(Accessor(w[5]), 3);
            library->fill_tensor_uniform(Accessor(b[5]), 4);
            library->fill_tensor_uniform(Accessor(w[6]), 5);
            library->fill_tensor_uniform(Accessor(b[6]), 6);
            library->fill_tensor_uniform(Accessor(w[7]), 7);
            library->fill_tensor_uniform(Accessor(b[7]), 8);

            if(!_is_direct_conv)
            {
                library->fill_tensor_uniform(Accessor(*dynamic_cast<TensorType *>(w11.get())), 9);
                library->fill_tensor_uniform(Accessor(*dynamic_cast<TensorType *>(w12.get())), 10);
                library->fill_tensor_uniform(Accessor(*dynamic_cast<TensorType *>(w31.get())), 11);
                library->fill_tensor_uniform(Accessor(*dynamic_cast<TensorType *>(w32.get())), 12);
                library->fill_tensor_uniform(Accessor(*dynamic_cast<TensorType *>(w41.get())), 13);
                library->fill_tensor_uniform(Accessor(*dynamic_cast<TensorType *>(w42.get())), 14);
            }
            else
            {
                library->fill_tensor_uniform(Accessor(w[1]), 9);
                library->fill_tensor_uniform(Accessor(b[1]), 10);
                library->fill_tensor_uniform(Accessor(w[3]), 11);
                library->fill_tensor_uniform(Accessor(b[3]), 12);
                library->fill_tensor_uniform(Accessor(w[4]), 13);
                library->fill_tensor_uniform(Accessor(b[4]), 14);
            }
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
        ARM_COMPUTE_ERROR_ON(_reshaped_weights);

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
        // Free allocations
        input.allocator()->free();
        output.allocator()->free();

        if(!_reshaped_weights)
        {
            for(auto &wi : w)
            {
                wi.allocator()->free();
            }

            for(auto &bi : b)
            {
                bi.allocator()->free();
            }
        }
        else
        {
            w[0].allocator()->free();
            w[2].allocator()->free();
            w[5].allocator()->free();
            w[6].allocator()->free();
            w[7].allocator()->free();

            b[5].allocator()->free();
            b[6].allocator()->free();
            b[7].allocator()->free();

            if(_is_direct_conv)
            {
                w[3].allocator()->free();
                w[4].allocator()->free();
                b[2].allocator()->free();
                b[3].allocator()->free();
                b[4].allocator()->free();
            }
        }

        w11.reset();
        w12.reset();
        b11.reset();
        b11.reset();
        w31.reset();
        w32.reset();
        b31.reset();
        b32.reset();
        w41.reset();
        w42.reset();
        b41.reset();
        b42.reset();

        conv1_out.allocator()->free();
        act1_out.allocator()->free();
        norm1_out.allocator()->free();
        pool1_out.allocator()->free();
        conv2_out.allocator()->free();
        act2_out.allocator()->free();
        norm2_out.allocator()->free();
        pool2_out.allocator()->free();
        conv3_out.allocator()->free();
        act3_out.allocator()->free();
        conv4_out.allocator()->free();
        act4_out.allocator()->free();
        conv5_out.allocator()->free();
        act5_out.allocator()->free();
        pool5_out.allocator()->free();
        fc6_out.allocator()->free();
        act6_out.allocator()->free();
        fc7_out.allocator()->free();
        act7_out.allocator()->free();
        fc8_out.allocator()->free();
    }

    /** Runs the model */
    void run()
    {
        // Layer 1
        conv1.run();
        act1.run();
        norm1.run();
        pool1.run();
        // Layer 2
        conv21.run();
        conv22.run();
        act2.run();
        norm2.run();
        pool2.run();
        // Layer 3
        conv3.run();
        act3.run();
        // Layer 4
        conv41.run();
        conv42.run();
        act4.run();
        // Layer 5
        conv51.run();
        conv52.run();
        act5.run();
        pool5.run();
        // Layer 6
        fc6.run();
        act6.run();
        // Layer 7
        fc7.run();
        act7.run();
        // Layer 8
        fc8.run();
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
    struct DirectConv
    {
        template <typename ConvolutionLayerFunction1 = ConvolutionLayerFunction, typename DirectConvolutionLayerFunction1 = DirectConvolutionLayerFunction>
        typename std::enable_if < !std::is_same<ConvolutionLayerFunction1, DirectConvolutionLayerFunction1>::value, void >::type
        configure(ITensorType *input, const ITensorType *weights, const ITensorType *biases, ITensorType *output, const PadStrideInfo &conv_info, const WeightsInfo &weights_info = WeightsInfo())
        {
            _func.configure(input, weights, biases, output, conv_info);
        }

        template <typename ConvolutionLayerFunction1 = ConvolutionLayerFunction, typename DirectConvolutionLayerFunction1 = DirectConvolutionLayerFunction>
        typename std::enable_if<std::is_same<ConvolutionLayerFunction1, DirectConvolutionLayerFunction1>::value, void>::type
        configure(ITensorType *input, const ITensorType *weights, const ITensorType *biases, ITensorType *output, const PadStrideInfo &conv_info, const WeightsInfo &weights_info = WeightsInfo())
        {
            _func.configure(input, weights, biases, output, conv_info, weights_info);
        }

        void run()
        {
            _func.run();
        }

        DirectConvolutionLayerFunction _func{};
    };

    DataType     _data_type{ DataType::UNKNOWN };
    int          _fixed_point_position{ 0 };
    unsigned int _batches{ 0 };
    bool         _reshaped_weights{ false };
    bool         _is_direct_conv{ !std::is_same<ConvolutionLayerFunction, DirectConvolutionLayerFunction>::value };

    ActivationLayerFunction     act1{}, act2{}, act3{}, act4{}, act5{}, act6{}, act7{};
    ConvolutionLayerFunction    conv1{};
    DirectConv                  conv21{}, conv22{}, conv3{}, conv41{}, conv42{}, conv51{}, conv52{};
    FullyConnectedLayerFunction fc6{}, fc7{}, fc8{};
    NormalizationLayerFunction  norm1{}, norm2{};
    PoolingLayerFunction        pool1{}, pool2{}, pool5{};
    SoftmaxLayerFunction        smx{};

    TensorType input{}, output{};
    std::array<TensorType, 8> w{ {} }, b{ {} };
    std::unique_ptr<ITensorType> w11{ nullptr }, w12{ nullptr }, b11{ nullptr }, b12{ nullptr };
    std::unique_ptr<ITensorType> w31{ nullptr }, w32{ nullptr }, b31{ nullptr }, b32{ nullptr };
    std::unique_ptr<ITensorType> w41{ nullptr }, w42{ nullptr }, b41{ nullptr }, b42{ nullptr };

    TensorType conv1_out{}, act1_out{}, norm1_out{}, pool1_out{};
    TensorType conv2_out{}, act2_out{}, pool2_out{}, norm2_out{};
    TensorType conv3_out{}, act3_out{};
    TensorType conv4_out{}, act4_out{};
    TensorType conv5_out{}, act5_out{}, pool5_out{};
    TensorType fc6_out{}, act6_out{};
    TensorType fc7_out{}, act7_out{};
    TensorType fc8_out{};

    std::unique_ptr<SubTensorType> pool11_out{}, pool12_out{};
    std::unique_ptr<SubTensorType> conv21_out{}, conv22_out{};
    std::unique_ptr<SubTensorType> act31_out{}, act32_out{};
    std::unique_ptr<SubTensorType> conv41_out{}, conv42_out{}, act41_out{}, act42_out{};
    std::unique_ptr<SubTensorType> conv51_out{}, conv52_out{};
};
} // namespace networks
} // namespace test
} // namespace arm_compute
#endif //__ARM_COMPUTE_TEST_MODEL_OBJECTS_ALEXNET_H__
