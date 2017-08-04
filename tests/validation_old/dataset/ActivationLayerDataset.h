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
#ifndef __ARM_COMPUTE_TEST_DATASET_ACTIVATION_LAYER_DATASET_H__
#define __ARM_COMPUTE_TEST_DATASET_ACTIVATION_LAYER_DATASET_H__

#include "utils/TypePrinter.h"

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/validation_old/dataset/GenericDataset.h"

#include <sstream>
#include <type_traits>

#ifdef BOOST
#include "tests/validation_old/boost_wrapper.h"
#endif /* BOOST */

namespace arm_compute
{
namespace test
{
class ActivationLayerDataObject
{
public:
    operator std::string() const
    {
        std::stringstream ss;
        ss << "ActivationLayer";
        ss << "_I" << shape;
        ss << "_F_" << info.activation();
        return ss.str();
    }

public:
    TensorShape         shape;
    ActivationLayerInfo info;
};

template <unsigned int Size>
using ActivationLayerDataset = GenericDataset<ActivationLayerDataObject, Size>;

class AlexNetActivationLayerDataset final : public ActivationLayerDataset<5>
{
public:
    AlexNetActivationLayerDataset()
        : GenericDataset
    {
        ActivationLayerDataObject{ TensorShape(55U, 55U, 96U), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) },
        ActivationLayerDataObject{ TensorShape(27U, 27U, 256U), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) },
        ActivationLayerDataObject{ TensorShape(13U, 13U, 384U), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) },
        ActivationLayerDataObject{ TensorShape(13U, 13U, 256U), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) },
        ActivationLayerDataObject{ TensorShape(4096U), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) },
    }
    {
    }

    ~AlexNetActivationLayerDataset() = default;
};

class LeNet5ActivationLayerDataset final : public ActivationLayerDataset<1>
{
public:
    LeNet5ActivationLayerDataset()
        : GenericDataset
    {
        ActivationLayerDataObject{ TensorShape(500U), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) },
    }
    {
    }

    ~LeNet5ActivationLayerDataset() = default;
};

class GoogLeNetActivationLayerDataset final : public ActivationLayerDataset<33>
{
public:
    GoogLeNetActivationLayerDataset()
        : GenericDataset
    {
        // conv1/relu_7x7
        ActivationLayerDataObject{ TensorShape(112U, 112U, 64U), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) },
        // conv2/relu_3x3_reduce
        ActivationLayerDataObject{ TensorShape(56U, 56U, 64U), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) },
        // conv2/relu_3x3
        ActivationLayerDataObject{ TensorShape(56U, 56U, 192U), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) },
        // inception_3a/relu_1x1, inception_3b/relu_pool_proj
        ActivationLayerDataObject{ TensorShape(28U, 28U, 64U), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) },
        // inception_3a/relu_3x3_reduce, inception_3b/relu_5x5
        ActivationLayerDataObject{ TensorShape(28U, 28U, 96U), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) },
        // inception_3a/relu_3x3, inception_3b/relu_1x1, inception_3b/relu_3x3_reduce
        ActivationLayerDataObject{ TensorShape(28U, 28U, 128U), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) },
        // inception_3a/relu_5x5_reduce
        ActivationLayerDataObject{ TensorShape(28U, 28U, 16U), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) },
        // inception_3a/relu_5x5, inception_3a/relu_pool_proj, inception_3b/relu_5x5_reduce
        ActivationLayerDataObject{ TensorShape(28U, 28U, 32U), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) },
        // inception_3b/relu_3x3
        ActivationLayerDataObject{ TensorShape(28U, 28U, 192U), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) },
        // inception_4a/relu_1x1
        ActivationLayerDataObject{ TensorShape(14U, 14U, 192U), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) },
        // inception_4a/relu_3x3_reduce
        ActivationLayerDataObject{ TensorShape(14U, 14U, 96U), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) },
        // inception_4a/relu_3x3
        ActivationLayerDataObject{ TensorShape(14U, 14U, 208U), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) },
        // inception_4a/relu_5x5_reduce
        ActivationLayerDataObject{ TensorShape(14U, 14U, 16U), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) },
        // inception_4a/relu_5x5
        ActivationLayerDataObject{ TensorShape(14U, 14U, 48U), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) },
        // inception_4a/relu_pool_proj, inception_4b/relu_5x5, inception_4b/relu_pool_proj, inception_4c/relu_5x5, inception_4c/relu_pool_proj, inception_4d/relu_5x5, inception_4d/relu_pool_proj
        ActivationLayerDataObject{ TensorShape(14U, 14U, 64U), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) },
        // inception_4b/relu_1x1, inception_4e/relu_3x3_reduce
        ActivationLayerDataObject{ TensorShape(14U, 14U, 160U), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) },
        // inception_4b/relu_3x3_reduce, inception_4d/relu_1x1
        ActivationLayerDataObject{ TensorShape(14U, 14U, 112U), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) },
        // inception_4b/relu_3x3
        ActivationLayerDataObject{ TensorShape(14U, 14U, 224U), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) },
        // inception_4b/relu_5x5_reduce, inception_4c/relu_5x5_reduce
        ActivationLayerDataObject{ TensorShape(14U, 14U, 24U), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) },
        // inception_4c/relu_1x1, inception_4c/relu_3x3_reduce, inception_4e/relu_5x5, inception_4e/relu_pool_proj
        ActivationLayerDataObject{ TensorShape(14U, 14U, 128U), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) },
        // inception_4c/relu_3x3, inception_4e/relu_1x1
        ActivationLayerDataObject{ TensorShape(14U, 14U, 256U), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) },
        // inception_4d/relu_3x3_reduce
        ActivationLayerDataObject{ TensorShape(14U, 14U, 144U), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) },
        // inception_4d/relu_3x3
        ActivationLayerDataObject{ TensorShape(14U, 14U, 288U), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) },
        // inception_4d/relu_5x5_reduce, inception_4e/relu_5x5_reduce
        ActivationLayerDataObject{ TensorShape(14U, 14U, 32U), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) },
        // inception_4e/relu_3x3
        ActivationLayerDataObject{ TensorShape(14U, 14U, 320U), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) },
        // inception_5a/relu_1x1
        ActivationLayerDataObject{ TensorShape(7U, 7U, 256U), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) },
        // inception_5a/relu_3x3_reduce
        ActivationLayerDataObject{ TensorShape(7U, 7U, 160U), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) },
        // inception_5a/relu_3x3
        ActivationLayerDataObject{ TensorShape(7U, 7U, 320U), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) },
        // inception_5a/relu_5x5_reduce
        ActivationLayerDataObject{ TensorShape(7U, 7U, 32U), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) },
        // inception_5a/relu_5x5, inception_5a/relu_pool_proj, inception_5b/relu_5x5, inception_5b/relu_pool_proj
        ActivationLayerDataObject{ TensorShape(7U, 7U, 128U), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) },
        // inception_5b/relu_1x1, inception_5b/relu_3x3
        ActivationLayerDataObject{ TensorShape(7U, 7U, 384U), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) },
        // inception_5b/relu_3x3_reduce
        ActivationLayerDataObject{ TensorShape(7U, 7U, 192U), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) },
        // inception_5b/relu_5x5_reduce
        ActivationLayerDataObject{ TensorShape(7U, 7U, 48U), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU) }
    }
    {
    }

    ~GoogLeNetActivationLayerDataset() = default;
};

} // namespace test
} // namespace arm_compute
#endif //__ARM_COMPUTE_TEST_DATASET_ACTIVATION_LAYER_DATASET_H__
