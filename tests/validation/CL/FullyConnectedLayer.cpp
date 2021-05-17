/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "arm_compute/runtime/CL/functions/CLFullyConnectedLayer.h"
#include "tests/CL/CLAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/FullyConnectedLayerDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/FullyConnectedLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
/** Tolerance for float operations */
constexpr RelativeTolerance<float>  rel_tolerance_f32(0.05f);   /**< Relative tolerance value for comparing reference's output against implementation's output for DataType:F32 */
constexpr AbsoluteTolerance<float>  abs_tolerance_f32(0.0001f); /**< Absolute tolerance value for comparing reference's output against implementation's output for DataType::F32 */
RelativeTolerance<half_float::half> tolerance_f16(half(0.2));   /**< Relative tolerance value for comparing reference's output against implementation's output for DataType::F16 */
constexpr float                     tolerance_num = 0.07f;      /**< Tolerance number */

/** Tolerance for quantized asymmetric operations */
constexpr AbsoluteTolerance<uint8_t> tolerance_qasymm8(1);

const auto FullyConnectedParameters = combine(framework::dataset::make("TransposeWeights", { false, true }), framework::dataset::make("ReshapeWeights", { false, true }));

const auto QuantizationData = framework::dataset::make("QuantizationInfo",
{
    QuantizationInfo(1.f / 255.f, 10),
    QuantizationInfo(1.1f, 10),
});

const auto ActivationFunctionsDataset = framework::dataset::make("ActivationInfo",
{
    ActivationLayerInfo(),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 0.5f),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 0.75f, 0.25f),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::TANH)
});

const auto ActivationFunctionsQuantizedDataset = concat(concat(concat(
                                                                   framework::dataset::make("ActivationInfo", ActivationLayerInfo()),
                                                                   framework::dataset::make("ActivationInfo", ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))),
                                                               framework::dataset::make("ActivationInfo", ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 0.5f))),
                                                        framework::dataset::make("ActivationInfo", ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 0.75f, 0.25f)));
} // namespace

TEST_SUITE(CL)
TEST_SUITE(FullyConnectedLayer)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(zip(zip(
    framework::dataset::make("InputInfo", { TensorInfo(TensorShape(9U, 5U, 7U, 3U), 1, DataType::F32),    // Mismatching data types
                                            TensorInfo(TensorShape(8U, 4U, 6U, 4U), 1, DataType::F32),
                                            TensorInfo(TensorShape(8U, 4U, 6U, 4U), 1, DataType::F32),
                                            TensorInfo(TensorShape(9U, 5U, 7U, 3U), 1, DataType::F32),    // Invalid weights dimensions
                                            TensorInfo(TensorShape(9U, 5U, 7U, 3U), 1, DataType::F32),    // Wrongly reshaped weights
                                          }),
    framework::dataset::make("WeightsInfo",{ TensorInfo(TensorShape(315U, 271U), 1, DataType::F16),
                                             TensorInfo(TensorShape(192U, 192U), 1, DataType::F32),
                                             TensorInfo(TensorShape(192U, 192U), 1, DataType::F32),
                                             TensorInfo(TensorShape(217U, 231U), 1, DataType::F32),
                                             TensorInfo(TensorShape(217U, 315U), 1, DataType::F32),
                                          })),
    framework::dataset::make("BiasInfo",{ TensorInfo(TensorShape(271U), 1, DataType::F32),
                                          TensorInfo(TensorShape(192U), 1, DataType::F32),
                                          TensorInfo(TensorShape(192U), 1, DataType::F32),
                                          TensorInfo(TensorShape(271U), 1, DataType::F32),
                                          TensorInfo(TensorShape(271U), 1, DataType::F32),
                                          })),
    framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(271U, 3U), 1, DataType::F32),
                                            TensorInfo(TensorShape(192U, 4U), 1, DataType::F32),
                                            TensorInfo(TensorShape(192U, 4U), 1, DataType::F32),
                                            TensorInfo(TensorShape(271U, 3U), 1, DataType::F32),
                                            TensorInfo(TensorShape(271U, 3U), 1, DataType::F32),
                                           })),
    framework::dataset::make("TransposeWeights",{ true, true, false, true, true })),
    framework::dataset::make("ReshapedWeights",{ false, false, false, false, false})),
    framework::dataset::make("Expected", { false, true, true, false, false })),
    input_info, weights_info, bias_info, output_info, transpose_weights, reshaped_weights, expected)
{
    // Create Fully Connected layer info
    FullyConnectedLayerInfo fc_info;
    fc_info.transpose_weights = transpose_weights;
    fc_info.are_weights_reshaped = reshaped_weights;

    Status status = CLFullyConnectedLayer::validate(&input_info.clone()->set_is_resizable(false),
                                                    &weights_info.clone()->set_is_resizable(false),
                                                    &bias_info.clone()->set_is_resizable(false),
                                                    &output_info.clone()->set_is_resizable(false),
                                                    fc_info);
    ARM_COMPUTE_EXPECT(bool(status) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using CLFullyConnectedLayerFixture = FullyConnectedLayerValidationFixture<CLTensor, CLAccessor, CLFullyConnectedLayer, T>;
template <typename T>
using CLFullyConnectedLayerMixedDataLayoutFixture = FullyConnectedLayerValidationFixture<CLTensor, CLAccessor, CLFullyConnectedLayer, T, true>;
template <typename T>
using CLFullyConnectedLayerDynamicWeightsFixture = FullyConnectedWithDynamicWeightsFixture<CLTensor, CLAccessor, CLFullyConnectedLayer, T>;

TEST_SUITE(Float)
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLFullyConnectedLayerFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallFullyConnectedLayerDataset(),
                                                                                                                        FullyConnectedParameters),
                                                                                                                        framework::dataset::make("DataType", DataType::F16)),
                                                                                                                ActivationFunctionsDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16, tolerance_num);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLFullyConnectedLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeFullyConnectedLayerDataset(),
                                                                                                                      FullyConnectedParameters),
                                                                                                                      framework::dataset::make("DataType", DataType::F16)),
                                                                                                              ActivationFunctionsDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16, tolerance_num);
}
TEST_SUITE_END()

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLFullyConnectedLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallFullyConnectedLayerDataset(), FullyConnectedParameters),
                                                                                                                 framework::dataset::make("DataType", DataType::F32)),
                                                                                                                 ActivationFunctionsDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0, abs_tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunMixedDataLayout, CLFullyConnectedLayerMixedDataLayoutFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(combine(combine(
                           framework::dataset::make("Input", TensorShape(9U, 5U, 7U)),
                           framework::dataset::make("Weights", TensorShape(315U, 271U))),
                       framework::dataset::make("Biases", TensorShape(271U))),
                       framework::dataset::make("Output", TensorShape(271U))),
                       FullyConnectedParameters),
                       framework::dataset::make("DataType", DataType::F32)),
                       framework::dataset::make("ActivationInfo", ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0, abs_tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunDynamicWeights, CLFullyConnectedLayerDynamicWeightsFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallFullyConnectedLayerDataset(),
                       framework::dataset::make("DataType", DataType::F32)),
                       framework::dataset::make("ActivationInfo", ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))))
{
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLFullyConnectedLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeFullyConnectedLayerDataset(), FullyConnectedParameters),
                                                                                                                       framework::dataset::make("DataType", DataType::F32)),
                                                                                                               ActivationFunctionsDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0, abs_tolerance_f32);
}
TEST_SUITE_END()
TEST_SUITE_END()

template <typename T>
using CLFullyConnectedLayerQuantizedFixture = FullyConnectedLayerValidationQuantizedFixture<CLTensor, CLAccessor, CLFullyConnectedLayer, T>;
template <typename T>
using CLFullyConnectedLayerQuantizedMixedDataLayoutFixture = FullyConnectedLayerValidationQuantizedFixture<CLTensor, CLAccessor, CLFullyConnectedLayer, T, true>;

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLFullyConnectedLayerQuantizedFixture<uint8_t>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(datasets::SmallFullyConnectedLayerDataset(), FullyConnectedParameters), framework::dataset::make("DataType", DataType::QASYMM8)), QuantizationData),
                               ActivationFunctionsQuantizedDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunMixedDataLayout, CLFullyConnectedLayerQuantizedMixedDataLayoutFixture<uint8_t>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(combine(combine(combine(
                                                                           framework::dataset::make("Input", TensorShape(9U, 5U, 7U)),
                                                                           framework::dataset::make("Weights", TensorShape(315U, 271U))),
                                                                       framework::dataset::make("Biases", TensorShape(271U))),
                                                               framework::dataset::make("Output", TensorShape(271U))),
                                                       FullyConnectedParameters),
                                               framework::dataset::make("DataType", DataType::QASYMM8)),
                                       QuantizationData),
                               framework::dataset::make("ActivationInfo", ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLFullyConnectedLayerQuantizedFixture<uint8_t>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(datasets::LargeFullyConnectedLayerDataset(), FullyConnectedParameters), framework::dataset::make("DataType", DataType::QASYMM8)), QuantizationData),
                               ActivationFunctionsQuantizedDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() /* QASYMM8 */
TEST_SUITE(QASYMM8_SIGNED)
FIXTURE_DATA_TEST_CASE(RunSmall, CLFullyConnectedLayerQuantizedFixture<int8_t>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(datasets::SmallFullyConnectedLayerDataset(), FullyConnectedParameters), framework::dataset::make("DataType", DataType::QASYMM8_SIGNED)), QuantizationData),
                               ActivationFunctionsQuantizedDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunMixedDataLayout, CLFullyConnectedLayerQuantizedMixedDataLayoutFixture<int8_t>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(combine(combine(combine(
                                                                           framework::dataset::make("Input", TensorShape(9U, 5U, 7U)),
                                                                           framework::dataset::make("Weights", TensorShape(315U, 271U))),
                                                                       framework::dataset::make("Biases", TensorShape(271U))),
                                                               framework::dataset::make("Output", TensorShape(271U))),
                                                       FullyConnectedParameters),
                                               framework::dataset::make("DataType", DataType::QASYMM8_SIGNED)),
                                       QuantizationData),
                               framework::dataset::make("ActivationInfo", ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // QASYMM8_SIGNED
TEST_SUITE_END() // Quantized
TEST_SUITE_END() // FullyConnectedLayer
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
