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
#ifndef __ARM_COMPUTE_TEST_DATASET_FULLY_CONNECTED_LAYER_DATASET_H__
#define __ARM_COMPUTE_TEST_DATASET_FULLY_CONNECTED_LAYER_DATASET_H__

#include "TypePrinter.h"

#include "arm_compute/core/TensorShape.h"
#include "dataset/GenericDataset.h"

#include <sstream>
#include <type_traits>

#ifdef BOOST
#include "boost_wrapper.h"
#endif

namespace arm_compute
{
namespace test
{
class FullyConnectedLayerDataObject
{
public:
    operator std::string() const
    {
        std::stringstream ss;
        ss << "FullyConnectedLayer";
        ss << "_I" << src_shape;
        ss << "_K" << weights_shape;
        return ss.str();
    }

    friend std::ostream &operator<<(std::ostream &os, const FullyConnectedLayerDataObject &obj)
    {
        os << static_cast<std::string>(obj);
        return os;
    }

public:
    TensorShape src_shape;
    TensorShape weights_shape;
    TensorShape bias_shape;
    TensorShape dst_shape;
    bool        transpose_weights;
    bool        are_weights_reshaped;
};

template <unsigned int Size>
using FullyConnectedLayerDataset = GenericDataset<FullyConnectedLayerDataObject, Size>;

class SmallFullyConnectedLayerDataset final : public FullyConnectedLayerDataset<5>
{
public:
    SmallFullyConnectedLayerDataset()
        : GenericDataset
    {
        FullyConnectedLayerDataObject{ TensorShape(9U, 5U, 7U), TensorShape(315U, 271U), TensorShape(271U), TensorShape(271U), true, false },
        FullyConnectedLayerDataObject{ TensorShape(9U, 5U, 7U, 3U), TensorShape(315U, 271U), TensorShape(271U), TensorShape(271U, 3U), true, false },
        FullyConnectedLayerDataObject{ TensorShape(201U), TensorShape(201U, 529U), TensorShape(529U), TensorShape(529U), true, false },
        FullyConnectedLayerDataObject{ TensorShape(9U, 5U, 7U), TensorShape(315U, 271U), TensorShape(271U), TensorShape(271U), true, true },
        FullyConnectedLayerDataObject{ TensorShape(201U), TensorShape(201U, 529U), TensorShape(529U), TensorShape(529U), true, true },
    }
    {
    }

    ~SmallFullyConnectedLayerDataset() = default;
};

class LargeFullyConnectedLayerDataset final : public FullyConnectedLayerDataset<5>
{
public:
    LargeFullyConnectedLayerDataset()
        : GenericDataset
    {
        FullyConnectedLayerDataObject{ TensorShape(9U, 5U, 257U), TensorShape(11565U, 2123U), TensorShape(2123U), TensorShape(2123U), true, false },
        FullyConnectedLayerDataObject{ TensorShape(9U, 5U, 257U, 2U), TensorShape(11565U, 2123U), TensorShape(2123U), TensorShape(2123U, 2U), true, false },
        FullyConnectedLayerDataObject{ TensorShape(3127U), TensorShape(3127U, 989U), TensorShape(989U), TensorShape(989U), true, false },
        FullyConnectedLayerDataObject{ TensorShape(9U, 5U, 257U), TensorShape(11565U, 2123U), TensorShape(2123U), TensorShape(2123U), true, true },
        FullyConnectedLayerDataObject{ TensorShape(3127U), TensorShape(3127U, 989U), TensorShape(989U), TensorShape(989U), true, true },
    }
    {
    }

    ~LargeFullyConnectedLayerDataset() = default;
};

class AlexNetFullyConnectedLayerDataset final : public FullyConnectedLayerDataset<3>
{
public:
    AlexNetFullyConnectedLayerDataset()
        : GenericDataset
    {
        FullyConnectedLayerDataObject{ TensorShape(6U, 6U, 256U), TensorShape(9216U, 4096U), TensorShape(4096U), TensorShape(4096U), true },
        FullyConnectedLayerDataObject{ TensorShape(4096U), TensorShape(4096U, 4096U), TensorShape(4096U), TensorShape(4096U), true },
        FullyConnectedLayerDataObject{ TensorShape(4096U), TensorShape(4096U, 1000U), TensorShape(1000U), TensorShape(1000U), true },
    }
    {
    }

    ~AlexNetFullyConnectedLayerDataset() = default;
};

class LeNet5FullyConnectedLayerDataset final : public FullyConnectedLayerDataset<2>
{
public:
    LeNet5FullyConnectedLayerDataset()
        : GenericDataset
    {
        FullyConnectedLayerDataObject{ TensorShape(4U, 4U, 50U), TensorShape(800U, 500U), TensorShape(500U), TensorShape(500U) },
        FullyConnectedLayerDataObject{ TensorShape(500U), TensorShape(500U, 10U), TensorShape(10U), TensorShape(10U) },
    }
    {
    }

    ~LeNet5FullyConnectedLayerDataset() = default;
};

class GoogLeNetFullyConnectedLayerDataset final : public FullyConnectedLayerDataset<1>
{
public:
    GoogLeNetFullyConnectedLayerDataset()
        : GenericDataset
    {
        FullyConnectedLayerDataObject{ TensorShape(1024U), TensorShape(1024U, 1000U), TensorShape(1000U), TensorShape(1000U), true },
    }
    {
    }

    ~GoogLeNetFullyConnectedLayerDataset() = default;
};
} // namespace test
} // namespace arm_compute
#endif //__ARM_COMPUTE_TEST_DATASET_FULLY_CONNECTED_LAYER_DATASET_H__
