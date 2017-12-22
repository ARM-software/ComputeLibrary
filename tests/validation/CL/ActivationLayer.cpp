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
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLActivationLayer.h"
#include "tests/CL/CLAccessor.h"
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
    DataType::QS8,
    DataType::QS16,
});

/** Input data sets. */
const auto ActivationDataset = combine(combine(framework::dataset::make("InPlace", { false, true }), datasets::ActivationFunctions()), framework::dataset::make("AlphaBeta", { 0.5f, 1.f }));
} // namespace

TEST_SUITE(CL)
TEST_SUITE(ActivationLayer)

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(concat(datasets::SmallShapes(), datasets::LargeShapes()), CNNDataTypes), framework::dataset::make("InPlace", { false, true })),
               shape, data_type, in_place)
{
    // Set fixed point position data type allowed
    const int fixed_point_position = is_data_type_fixed_point(data_type) ? 3 : 0;

    // Create tensors
    CLTensor src = create_tensor<CLTensor>(shape, data_type, 1, fixed_point_position);
    CLTensor dst = create_tensor<CLTensor>(shape, data_type, 1, fixed_point_position);

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    CLActivationLayer act_layer;

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
    const int         step    = 16 / arm_compute::data_size_from_type(data_type);
    const PaddingSize padding = PaddingCalculator(shape.x(), step).required_padding();
    validate(src.info()->padding(), padding);

    if(!in_place)
    {
        validate(dst.info()->padding(), padding);
    }
}

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
               framework::dataset::make("InputInfo", { TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),     // Mismatching data types
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),     // Window shrink
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QASYMM8),
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::QASYMM8), // Unsupported activation
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),     // Mismatching shapes
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::QS8, 2),  // Mismatching fixed point
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QS8, 2),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QS8, 2),
                                                     }),
               framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QASYMM8),
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::QASYMM8),
                                                       TensorInfo(TensorShape(30U, 11U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::QS8, 3),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QS8, 2),
                                                       TensorInfo(),
                                                     })),
               framework::dataset::make("ActivationInfo", { ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
                                                            ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
                                                            ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
                                                            ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU),
                                                            ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
                                                            ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
                                                            ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
                                                            ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
                                                            ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
                                                          })),
               framework::dataset::make("Expected", { false, false, true, true, false, false, false, true, true })),
               input_info, output_info, act_info, expected)
{
    ARM_COMPUTE_EXPECT(bool(CLActivationLayer::validate(&input_info.clone()->set_is_resizable(false), (output_info.total_size() == 0) ? nullptr : &output_info.clone()->set_is_resizable(false), act_info)) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using CLActivationLayerFixture = ActivationValidationFixture<CLTensor, CLAccessor, CLActivationLayer, T>;

TEST_SUITE(Float)
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLActivationLayerFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallShapes(), ActivationDataset),
                                                                                                            framework::dataset::make("DataType",
                                                                                                                    DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance(_function, _data_type));
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLActivationLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeShapes(), ActivationDataset),
                                                                                                          framework::dataset::make("DataType",
                                                                                                                  DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance(_function, _data_type));
}
TEST_SUITE_END()

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLActivationLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallShapes(), ActivationDataset), framework::dataset::make("DataType",
                                                                                                             DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance(_function, _data_type));
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLActivationLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeShapes(), ActivationDataset), framework::dataset::make("DataType",
                                                                                                           DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance(_function, _data_type));
}
TEST_SUITE_END()
TEST_SUITE_END()

template <typename T>
using CLActivationLayerFixedPointFixture = ActivationValidationFixedPointFixture<CLTensor, CLAccessor, CLActivationLayer, T>;

TEST_SUITE(FixedPoint)
TEST_SUITE(QS8)
// We test for fixed point precision [3,5] because [1,2] and [6,7] ranges cause
// overflowing issues in most of the transcendentals functions.
FIXTURE_DATA_TEST_CASE(RunSmall, CLActivationLayerFixedPointFixture<int8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), ActivationDataset),
                                                                                                                        framework::dataset::make("DataType",
                                                                                                                                DataType::QS8)),
                                                                                                                        framework::dataset::make("FractionalBits", 3, 6)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance(_function, _data_type));
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLActivationLayerFixedPointFixture<int8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), ActivationDataset),
                                                                                                                      framework::dataset::make("DataType",
                                                                                                                              DataType::QS8)),
                                                                                                                      framework::dataset::make("FractionalBits", 3, 6)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance(_function, _data_type));
}
TEST_SUITE_END()

TEST_SUITE(QS16)
// Testing for fixed point position [1,14) as reciprocal limits the maximum fixed point position to 14
FIXTURE_DATA_TEST_CASE(RunSmall, CLActivationLayerFixedPointFixture<int16_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), ActivationDataset),
                       framework::dataset::make("DataType",
                                                DataType::QS16)),
                       framework::dataset::make("FractionalBits", 1, 14)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance(_function, _data_type));
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLActivationLayerFixedPointFixture<int16_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), ActivationDataset),
                                                                                                                       framework::dataset::make("DataType",
                                                                                                                               DataType::QS16)),
                                                                                                                       framework::dataset::make("FractionalBits", 1, 14)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance(_function, _data_type));
}
TEST_SUITE_END()
TEST_SUITE_END()

template <typename T>
using CLActivationLayerQuantizedFixture = ActivationValidationQuantizedFixture<CLTensor, CLAccessor, CLActivationLayer, T>;

/** Input data sets. */
const auto QuantizedActivationDataset = combine(combine(framework::dataset::make("InPlace", { false, true }), framework::dataset::make("ActivationFunction", { ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU })),
                                                framework::dataset::make("AlphaBeta", { 0.5f, 1.f }));

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLActivationLayerQuantizedFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), QuantizedActivationDataset),
                                                                                                                        framework::dataset::make("DataType",
                                                                                                                                DataType::QASYMM8)),
                                                                                                                        framework::dataset::make("QuantizationInfo", { QuantizationInfo(0.1f, 128.0f) })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance(_function, _data_type));
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLActivationLayerQuantizedFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), QuantizedActivationDataset),
                                                                                                                      framework::dataset::make("DataType",
                                                                                                                              DataType::QASYMM8)),
                                                                                                                      framework::dataset::make("QuantizationInfo", { QuantizationInfo(0.1f, 128.0f) })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance(_function, _data_type));
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
