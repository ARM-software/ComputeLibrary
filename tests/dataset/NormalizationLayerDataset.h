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
#ifndef __ARM_COMPUTE_TEST_DATASET_NORMALIZATION_LAYER_DATASET_H__
#define __ARM_COMPUTE_TEST_DATASET_NORMALIZATION_LAYER_DATASET_H__

#include "TypePrinter.h"

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "dataset/GenericDataset.h"

#include <sstream>
#include <type_traits>

#ifdef BOOST
#include "boost_wrapper.h"
#endif /* BOOST */

namespace arm_compute
{
namespace test
{
class NormalizationLayerDataObject
{
public:
    operator std::string() const
    {
        std::stringstream ss;
        ss << "NormalizationLayer";
        ss << "_I" << shape;
        ss << "_F_" << info.type();
        ss << "_S_" << info.norm_size();
        return ss.str();
    }

public:
    TensorShape            shape;
    NormalizationLayerInfo info;
};

template <unsigned int Size>
using NormalizationLayerDataset = GenericDataset<NormalizationLayerDataObject, Size>;

class GoogLeNetNormalizationLayerDataset final : public NormalizationLayerDataset<2>
{
public:
    GoogLeNetNormalizationLayerDataset()
        : GenericDataset
    {
        // conv2/norm2
        NormalizationLayerDataObject{ TensorShape(56U, 56U, 192U), NormalizationLayerInfo(NormType::CROSS_MAP, 5, 0.0001f, 0.75f) },
        // pool1/norm1
        NormalizationLayerDataObject{ TensorShape(56U, 56U, 64U), NormalizationLayerInfo(NormType::CROSS_MAP, 5, 0.0001f, 0.75f) }
    }
    {
    }

    ~GoogLeNetNormalizationLayerDataset() = default;
};

class AlexNetNormalizationLayerDataset final : public NormalizationLayerDataset<2>
{
public:
    AlexNetNormalizationLayerDataset()
        : GenericDataset
    {
        NormalizationLayerDataObject{ TensorShape(55U, 55U, 96U), NormalizationLayerInfo(NormType::CROSS_MAP, 5, 0.0001f, 0.75f) },
        NormalizationLayerDataObject{ TensorShape(27U, 27U, 256U), NormalizationLayerInfo(NormType::CROSS_MAP, 5, 0.0001f, 0.75f) },
    }
    {
    }

    ~AlexNetNormalizationLayerDataset() = default;
};

} // namespace test
} // namespace arm_compute
#endif /* __ARM_COMPUTE_TEST_DATASET_NORMALIZATION_LAYER_DATASET_H__ */
