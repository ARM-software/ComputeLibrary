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
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCTensor.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCTensorAllocator.h"
#include "arm_compute/runtime/GLES_COMPUTE/functions/GCActivationLayer.h"
#include "tests/GLES_COMPUTE/GCAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ActivationFunctionsDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/ActivationLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
/** Define tolerance of the activation layer.
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
            if(is_data_type_fixed_point(data_type))
            {
                return AbsoluteTolerance<float>(5.f);
            }
            else
            {
                return AbsoluteTolerance<float>(data_type == DataType::F16 ? 0.001f : epsilon);
            }
        case ActivationLayerInfo::ActivationFunction::LEAKY_RELU:
            return AbsoluteTolerance<float>(data_type == DataType::F16 ? 0.00001f : epsilon);
        case ActivationLayerInfo::ActivationFunction::SOFT_RELU:
        case ActivationLayerInfo::ActivationFunction::SQRT:
            if(is_data_type_fixed_point(data_type))
            {
                return AbsoluteTolerance<float>(5.f);
            }
            else
            {
                return AbsoluteTolerance<float>(data_type == DataType::F16 ? 0.01f : 0.00001f);
            }
        case ActivationLayerInfo::ActivationFunction::TANH:
            if(is_data_type_fixed_point(data_type))
            {
                return AbsoluteTolerance<float>(5.f);
            }
            else
            {
                return AbsoluteTolerance<float>(data_type == DataType::F16 ? 0.001f : 0.00001f);
            }
        default:
            return AbsoluteTolerance<float>(epsilon);
    }
}

/** CNN data types */
const auto CNNDataTypes = framework::dataset::make("DataType",
{
    DataType::F16,
    DataType::F32,
});

/** Input data sets. */
const auto ActivationDataset = combine(combine(framework::dataset::make("InPlace", { false, true }), datasets::ActivationFunctions()), framework::dataset::make("AlphaBeta", { 0.5f, 1.f }));
} // namespace

TEST_SUITE(GC)
TEST_SUITE(ActivationLayer)

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(concat(datasets::SmallShapes(), datasets::LargeShapes()), CNNDataTypes), framework::dataset::make("InPlace", { false, true })),
               shape, data_type, in_place)
{
    // Set fixed point position data type allowed
    const int fixed_point_position = 0;

    // Create tensors
    GCTensor src = create_tensor<GCTensor>(shape, data_type, 1, fixed_point_position);
    GCTensor dst = create_tensor<GCTensor>(shape, data_type, 1, fixed_point_position);

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    GCActivationLayer act_layer;

    if(in_place)
    {
        act_layer.configure(&src, nullptr, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::ABS));
    }
    else
    {
        act_layer.configure(&src, &dst, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::ABS));
    }

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(src.info()->valid_region(), valid_region);

    if(!in_place)
    {
        validate(dst.info()->valid_region(), valid_region);
    }

    // Validate padding
    const int         step    = (arm_compute::data_size_from_type(data_type) == 4 ? 1 : 2);
    const PaddingSize padding = PaddingCalculator(shape.x(), step).required_padding();
    validate(src.info()->padding(), padding);

    if(!in_place)
    {
        validate(dst.info()->padding(), padding);
    }
}

template <typename T>
using GCActivationLayerFixture = ActivationValidationFixture<GCTensor, GCAccessor, GCActivationLayer, T>;

TEST_SUITE(Float)
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, GCActivationLayerFixture<half_float::half>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallShapes(), ActivationDataset),
                                                                                                                        framework::dataset::make("DataType",
                                                                                                                                DataType::F16)))
{
    // Validate output
    validate(GCAccessor(_target), _reference, tolerance(_function, _data_type));
}
FIXTURE_DATA_TEST_CASE(RunLarge, GCActivationLayerFixture<half_float::half>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeShapes(), ActivationDataset),
                                                                                                                      framework::dataset::make("DataType",
                                                                                                                              DataType::F16)))
{
    // Validate output
    validate(GCAccessor(_target), _reference, tolerance(_function, _data_type));
}
TEST_SUITE_END()

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, GCActivationLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallShapes(), ActivationDataset), framework::dataset::make("DataType",
                                                                                                             DataType::F32)))
{
    // Validate output
    validate(GCAccessor(_target), _reference, tolerance(_function, _data_type));
}
FIXTURE_DATA_TEST_CASE(RunLarge, GCActivationLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeShapes(), ActivationDataset), framework::dataset::make("DataType",
                                                                                                           DataType::F32)))
{
    // Validate output
    validate(GCAccessor(_target), _reference, tolerance(_function, _data_type));
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
