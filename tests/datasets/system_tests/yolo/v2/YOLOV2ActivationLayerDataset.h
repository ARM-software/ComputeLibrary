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
#ifndef ARM_COMPUTE_TEST_YOLOV2_ACTIVATION_LAYER_DATASET
#define ARM_COMPUTE_TEST_YOLOV2_ACTIVATION_LAYER_DATASET

#include "tests/framework/datasets/Datasets.h"

#include "utils/TypePrinter.h"

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class YOLOV2ActivationLayerRELUDataset final : public
    framework::dataset::CartesianProductDataset<framework::dataset::InitializerListDataset<TensorShape>, framework::dataset::SingletonDataset<ActivationLayerInfo>>
{
public:
    YOLOV2ActivationLayerRELUDataset()
        : CartesianProductDataset
    {
        framework::dataset::make("Shape", { // relu1
            TensorShape(416U, 416U, 32U),
            // relu2
            TensorShape(208U, 208U, 64U),
            // relu3, relu5
            TensorShape(104U, 104U, 128U),
            // relu4
            TensorShape(104U, 104U, 64U),
            // relu6, relu8
            TensorShape(52U, 52U, 256U),
            // relu7
            TensorShape(52U, 52U, 128U),
            // relu9, relu11, relu13
            TensorShape(26U, 26U, 512U),
            // relu10, relu12
            TensorShape(26U, 26U, 256U),
            // relu14, relu16, relu18, relu19, relu20, relu21
            TensorShape(13U, 13U, 1024U),
            // relu15, relu17
            TensorShape(13U, 13U, 512U) }),
        framework::dataset::make("Info", ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
    }
    {
    }
    YOLOV2ActivationLayerRELUDataset(YOLOV2ActivationLayerRELUDataset &&) = default;
    ~YOLOV2ActivationLayerRELUDataset()                                   = default;
};

class YOLOV2ActivationLayerLINEARDataset final : public
    framework::dataset::CartesianProductDataset<framework::dataset::InitializerListDataset<TensorShape>, framework::dataset::SingletonDataset<ActivationLayerInfo>>
{
public:
    YOLOV2ActivationLayerLINEARDataset()
        : CartesianProductDataset
    {
        framework::dataset::make("Shape", { // linear22
            TensorShape(15U, 15U, 425U) }),
        framework::dataset::make("Info", ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LINEAR))
    }
    {
    }
    YOLOV2ActivationLayerLINEARDataset(YOLOV2ActivationLayerLINEARDataset &&) = default;
    ~YOLOV2ActivationLayerLINEARDataset()                                     = default;
};

class YOLOV2ActivationLayerDataset final : public framework::dataset::JoinDataset<YOLOV2ActivationLayerRELUDataset, YOLOV2ActivationLayerLINEARDataset>
{
public:
    YOLOV2ActivationLayerDataset()
        : JoinDataset
    {
        YOLOV2ActivationLayerRELUDataset(),
        YOLOV2ActivationLayerLINEARDataset()
    }
    {
    }
    YOLOV2ActivationLayerDataset(YOLOV2ActivationLayerDataset &&) = default;
    ~YOLOV2ActivationLayerDataset()                               = default;
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_YOLOV2_ACTIVATION_LAYER_DATASET */
