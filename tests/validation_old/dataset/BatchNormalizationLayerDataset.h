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
#ifndef __ARM_COMPUTE_TEST_DATASET_BATCH_NORMALIZATION_LAYER_DATASET_H__
#define __ARM_COMPUTE_TEST_DATASET_BATCH_NORMALIZATION_LAYER_DATASET_H__

#include "TypePrinter.h"

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/validation_old/dataset/GenericDataset.h"

#include <ostream>
#include <sstream>

#ifdef BOOST
#include "tests/validation_old/boost_wrapper.h"
#endif /* BOOST */

namespace arm_compute
{
namespace test
{
class BatchNormalizationLayerDataObject
{
public:
    operator std::string() const
    {
        std::stringstream ss;
        ss << "BatchNormalizationLayer";
        ss << "_I" << shape0;
        ss << "_I" << shape1;
        ss << "_I" << epsilon;
        return ss.str();
    }

    friend std::ostream &operator<<(std::ostream &s, const BatchNormalizationLayerDataObject &obj)
    {
        s << static_cast<std::string>(obj);
        return s;
    }

public:
    TensorShape shape0;
    TensorShape shape1;
    float       epsilon;
};

template <unsigned int Size>
using BatchNormalizationLayerDataset = GenericDataset<BatchNormalizationLayerDataObject, Size>;

class RandomBatchNormalizationLayerDataset final : public BatchNormalizationLayerDataset<3>
{
public:
    RandomBatchNormalizationLayerDataset()
        : GenericDataset
    {
        BatchNormalizationLayerDataObject{ TensorShape(15U, 16U, 2U, 12U), TensorShape(2U), 0.1f },
        BatchNormalizationLayerDataObject{ TensorShape(21U, 11U, 12U, 7U), TensorShape(12U), 0.1f },
        BatchNormalizationLayerDataObject{ TensorShape(7U, 3U, 6U, 11U), TensorShape(6U), 0.1f },
    }
    {
    }

    ~RandomBatchNormalizationLayerDataset() = default;
};

} // namespace test
} // namespace arm_compute
#endif //__ARM_COMPUTE_TEST_DATASET_BATCH_NORMALIZATION_LAYER_DATASET_H__
