/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#include "arm_compute/runtime/NEON/functions/NEActivationLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
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
/** Define relative tolerance of the activation layer.
 *
 * @param[in] data_type  The data type used.
 * @param[in] activation The activation function used.
 *
 * @return Relative tolerance depending on the activation function.
 */
RelativeTolerance<float> relative_tolerance(DataType data_type, ActivationLayerInfo::ActivationFunction activation)
{
    switch(activation)
    {
        case ActivationLayerInfo::ActivationFunction::LOGISTIC:
        case ActivationLayerInfo::ActivationFunction::SOFT_RELU:
        case ActivationLayerInfo::ActivationFunction::SQRT:
        case ActivationLayerInfo::ActivationFunction::TANH:
            switch(data_type)
            {
                case DataType::F16:
                    return RelativeTolerance<float>(0.1f);
                default:
                    return RelativeTolerance<float>(0.05f);
            }
        default:
            return RelativeTolerance<float>(0.f);
    }
}

/** Define absolute tolerance of the activation layer.
 *
 * @param[in] data_type  The data type used.
 * @param[in] activation The activation function used.
 *
 * @return Absolute tolerance depending on the activation function.
 */
AbsoluteTolerance<float> absolute_tolerance(DataType data_type, ActivationLayerInfo::ActivationFunction activation)
{
    switch(activation)
    {
        case ActivationLayerInfo::ActivationFunction::LOGISTIC:
        case ActivationLayerInfo::ActivationFunction::SOFT_RELU:
        case ActivationLayerInfo::ActivationFunction::SQRT:
        case ActivationLayerInfo::ActivationFunction::TANH:
            switch(data_type)
            {
                case DataType::F16:
                    return AbsoluteTolerance<float>(0.01f);
                default:
                    return AbsoluteTolerance<float>(0.00001f);
            }
        default:
            return AbsoluteTolerance<float>(0.f);
    }
}

/** Tolerance for quantized asymmetric operations */
#if defined(__aarch64__)
constexpr AbsoluteTolerance<uint8_t> tolerance_qasymm8(0);
#else  // defined(__aarch64__)
constexpr AbsoluteTolerance<uint8_t> tolerance_qasymm8(1);
#endif // defined(__aarch64__)

constexpr AbsoluteTolerance<int16_t> tolerance_qsymm16(1);

/** CNN data types */
const auto CNNDataTypes = framework::dataset::make("DataType",
{
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    DataType::F16,
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
    DataType::F32,
});

/** Input data sets. */
const auto ActivationDataset = combine(combine(framework::dataset::make("InPlace", { false, true }), datasets::ActivationFunctions()), framework::dataset::make("AlphaBeta", { 0.5f, 1.f }));
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(ActivationLayer)

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(datasets::SmallShapes(), CNNDataTypes), framework::dataset::make("InPlace", { false, true })),
               shape, data_type, in_place)
{
    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, data_type, 1);
    Tensor dst = create_tensor<Tensor>(shape, data_type, 1);

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    NEActivationLayer act_layer;

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
    validate(src.info()->padding(), PaddingSize());
    if(!in_place)
    {
        validate(dst.info()->padding(), PaddingSize());
    }
}

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
    framework::dataset::make("InputInfo", { TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),     // Mismatching data types
                                            TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                            TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),     // Mismatching shapes
                                          }),
    framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F16),
                                            TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                            TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                          })),
    framework::dataset::make("ActivationInfo", { ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
                                                 ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
                                                 ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
                                               })),
    framework::dataset::make("Expected", { false, true, false})),
    input_info, output_info, act_info, expected)
{
    bool is_valid = bool(NEActivationLayer::validate(&input_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), act_info));
    ARM_COMPUTE_EXPECT(is_valid == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using NEActivationLayerFixture = ActivationValidationFixture<Tensor, Accessor, NEActivationLayer, T>;

TEST_SUITE(Float)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEActivationLayerFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallShapes(), ActivationDataset),
                                                                                                            framework::dataset::make("DataType",
                                                                                                                    DataType::F16)))
{
    // Validate output
    validate(Accessor(_target), _reference, relative_tolerance(_data_type, _function), 0.f, absolute_tolerance(_data_type, _function));
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEActivationLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeShapes(), ActivationDataset),
                                                                                                          framework::dataset::make("DataType",
                                                                                                                  DataType::F16)))
{
    // Validate output
    validate(Accessor(_target), _reference, relative_tolerance(_data_type, _function), 0.f, absolute_tolerance(_data_type, _function));
}
TEST_SUITE_END()
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEActivationLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallShapes(), ActivationDataset), framework::dataset::make("DataType",
                                                                                                             DataType::F32)))

{
    // Validate output
    validate(Accessor(_target), _reference, relative_tolerance(_data_type, _function), 0.f, absolute_tolerance(_data_type, _function));
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEActivationLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeShapes(), ActivationDataset),
                                                                                                           framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(Accessor(_target), _reference, relative_tolerance(_data_type, _function), 0.f, absolute_tolerance(_data_type, _function));
}
TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float

template <typename T>
using NEActivationLayerQuantizedFixture = ActivationValidationQuantizedFixture<Tensor, Accessor, NEActivationLayer, T>;

/** Input data sets. */
const auto QuantizedActivationFunctionsDataset = framework::dataset::make("ActivationFunction", { ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU,
                                                                                                  ActivationLayerInfo::ActivationFunction::RELU,
                                                                                                  ActivationLayerInfo::ActivationFunction::BOUNDED_RELU,
                                                                                                  ActivationLayerInfo::ActivationFunction::LOGISTIC,
                                                                                                  ActivationLayerInfo::ActivationFunction::TANH
                                                                                                });
const auto QuantizedActivationDataset = combine(combine(framework::dataset::make("InPlace", { false }), QuantizedActivationFunctionsDataset),
                                                framework::dataset::make("AlphaBeta", { 0.5f, 1.f }));

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, NEActivationLayerQuantizedFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), QuantizedActivationDataset),
                                                                                                                        framework::dataset::make("DataType",
                                                                                                                                DataType::QASYMM8)),
                                                                                                                        framework::dataset::make("QuantizationInfo", { QuantizationInfo(0.1f, 128.0f) })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEActivationLayerQuantizedFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), QuantizedActivationDataset),
                                                                                                                      framework::dataset::make("DataType",
                                                                                                                              DataType::QASYMM8)),
                                                                                                                      framework::dataset::make("QuantizationInfo", { QuantizationInfo(0.1f, 128.0f) })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // QASYMM8

/** Input data sets. */
const auto Int16QuantizedActivationFunctionsDataset = framework::dataset::make("ActivationFunction", { ActivationLayerInfo::ActivationFunction::LOGISTIC,
                                                                                                       ActivationLayerInfo::ActivationFunction::TANH
                                                                                                     });
const auto Int16QuantizedActivationDataset = combine(combine(framework::dataset::make("InPlace", { false }), Int16QuantizedActivationFunctionsDataset),
                                                     framework::dataset::make("AlphaBeta", { 0.5f, 1.f }));

TEST_SUITE(QSYMM16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEActivationLayerQuantizedFixture<int16_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), Int16QuantizedActivationDataset),
                                                                                                                        framework::dataset::make("DataType",
                                                                                                                                DataType::QSYMM16)),
                                                                                                                        framework::dataset::make("QuantizationInfo", { QuantizationInfo(1.f / 32768.f, 0.f) })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qsymm16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEActivationLayerQuantizedFixture<int16_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), Int16QuantizedActivationDataset),
                                                                                                                      framework::dataset::make("DataType",
                                                                                                                              DataType::QSYMM16)),
                                                                                                                      framework::dataset::make("QuantizationInfo", { QuantizationInfo(1.f / 32768.f, 0.f) })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qsymm16);
}
TEST_SUITE_END() // QSYMM16
TEST_SUITE_END() // Quantized

TEST_SUITE_END() // ActivationLayer
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
