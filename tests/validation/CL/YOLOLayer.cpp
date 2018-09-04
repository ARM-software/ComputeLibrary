/*
 * Copyright (c) 2018 ARM Limited.
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
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLYOLOLayer.h"
#include "tests/CL/CLAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ActivationFunctionsDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/YOLOLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
/** Define tolerance of the yolo layer.
 *
 * @param[in] activation The activation function used.
 * @param[in] data_type  Data type.
 *
 * @return Tolerance depending on the activation function.
 */
AbsoluteTolerance<float> tolerance(ActivationLayerInfo::ActivationFunction activation, DataType data_type)
{
    constexpr float epsilon = 1e-6f;

    switch(activation)
    {
        case ActivationLayerInfo::ActivationFunction::LINEAR:
            return AbsoluteTolerance<float>(data_type == DataType::F16 ? 0.2f : epsilon);
        case ActivationLayerInfo::ActivationFunction::SQUARE:
            return AbsoluteTolerance<float>(data_type == DataType::F16 ? 0.1f : epsilon);
        case ActivationLayerInfo::ActivationFunction::LOGISTIC:
            return AbsoluteTolerance<float>(data_type == DataType::F16 ? 0.001f : epsilon);
        case ActivationLayerInfo::ActivationFunction::LEAKY_RELU:
            return AbsoluteTolerance<float>(data_type == DataType::F16 ? 0.00001f : epsilon);
        case ActivationLayerInfo::ActivationFunction::SOFT_RELU:
        case ActivationLayerInfo::ActivationFunction::SQRT:
            return AbsoluteTolerance<float>(data_type == DataType::F16 ? 0.01f : 0.00001f);
        case ActivationLayerInfo::ActivationFunction::TANH:
            return AbsoluteTolerance<float>(data_type == DataType::F16 ? 0.001f : 0.00001f);
        default:
            return AbsoluteTolerance<float>(epsilon);
    }
}

/** Floating point data sets. */
const auto YOLODataset = combine(combine(combine(combine(framework::dataset::make("InPlace", { false, true }), datasets::ActivationFunctions()),
                                                 framework::dataset::make("AlphaBeta", { 0.5f, 1.f })),
                                         framework::dataset::make("Classes", 40)),
                                 framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC }));
} // namespace

TEST_SUITE(CL)
TEST_SUITE(YOLOLayer)

template <typename T>
using CLYOLOLayerFixture = YOLOValidationFixture<CLTensor, CLAccessor, CLYOLOLayer, T>;

TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLYOLOLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallYOLOShapes(), YOLODataset), framework::dataset::make("DataType",
                                                                                                       DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance(_function, _data_type));
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLYOLOLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeYOLOShapes(), YOLODataset), framework::dataset::make("DataType",
                                                                                                     DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance(_function, _data_type));
}
TEST_SUITE_END() // FP32

TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLYOLOLayerFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallYOLOShapes(), YOLODataset), framework::dataset::make("DataType",
                                                                                                      DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance(_function, _data_type));
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLYOLOLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeYOLOShapes(), YOLODataset), framework::dataset::make("DataType",
                                                                                                    DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance(_function, _data_type));
}
TEST_SUITE_END() // FP16
TEST_SUITE_END() // Float

TEST_SUITE_END() // YOLOLayer
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
