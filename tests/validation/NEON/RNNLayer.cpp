/*
 * Copyright (c) 2018-2021, 2024-2026 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NERNNLayer.h"

#include "tests/datasets/RNNLayerDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/framework/Macros.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/validation/fixtures/RNNLayerFixture.h"
#include "tests/validation/Validation.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
using framework::dataset::make;

namespace
{
RelativeTolerance<float> tolerance_f32(
    0.001f); /**< Relative tolerance value for comparing reference's output against implementation's output for DataType:F32 */
#ifdef ARM_COMPUTE_ENABLE_FP16
RelativeTolerance<half> tolerance_f16(half(
    0.1)); /**< Relative tolerance value for comparing reference's output against implementation's output for DataType:F16 */
constexpr float         abs_tolerance_f16(
            0.02f); /**< Absolute tolerance value for comparing reference's output against implementation's output for DataType:F16 */
#endif /* ARM_COMPUTE_ENABLE_FP16 */
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(RNNLayer)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(
               make("InputInfo", { TensorInfo(TensorShape(27U, 13U), 1, DataType::U8),      // Wrong data type
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32), // Wrong input size
                                                       TensorInfo(TensorShape(27U, 13U), 1, DataType::F32),     // Wrong weights size
                                                       TensorInfo(TensorShape(27U, 13U), 1, DataType::F32),     // Wrong recurrent weights size
                                                       TensorInfo(TensorShape(27U, 13U), 1, DataType::F32),     // Wrong bias size
                                                       TensorInfo(TensorShape(27U, 13U), 1, DataType::F32),     // Wrong output size
                                                       TensorInfo(TensorShape(27U, 13U), 1, DataType::F32),     // Wrong hidden output size
                                                       TensorInfo(TensorShape(32U, 32U), 1, DataType::F32),
               }),
               make("WeightsInfo", { TensorInfo(TensorShape(27U, 11U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(27U, 11U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(27U, 11U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(27U, 11U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(27U, 11U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(27U, 11U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(27U, 11U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 32U), 1, DataType::F32),
               }),
               make("RecurrentWeightsInfo", { TensorInfo(TensorShape(11U, 11U), 1, DataType::F32),
                                                                  TensorInfo(TensorShape(11U, 11U), 1, DataType::F32),
                                                                  TensorInfo(TensorShape(11U, 11U), 1, DataType::F32),
                                                                  TensorInfo(TensorShape(25U, 11U, 2U), 1, DataType::F32),
                                                                  TensorInfo(TensorShape(11U, 11U), 1, DataType::F32),
                                                                  TensorInfo(TensorShape(11U, 11U), 1, DataType::F32),
                                                                  TensorInfo(TensorShape(11U, 11U), 1, DataType::F32),
                                                                  TensorInfo(TensorShape(32U, 32U), 1, DataType::F32),
               }),
               make("BiasInfo", { TensorInfo(TensorShape(11U), 1, DataType::F32),
                                                      TensorInfo(TensorShape(11U), 1, DataType::F32),
                                                      TensorInfo(TensorShape(11U), 1, DataType::F32),
                                                      TensorInfo(TensorShape(11U), 1, DataType::F32),
                                                      TensorInfo(TensorShape(30U), 1, DataType::F32),
                                                      TensorInfo(TensorShape(11U), 1, DataType::F32),
                                                      TensorInfo(TensorShape(11U), 1, DataType::F32),
                                                      TensorInfo(TensorShape(32U), 1, DataType::F32),
               }),
               make("OutputInfo", { TensorInfo(TensorShape(11U, 13U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(11U, 13U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(11U, 13U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(11U, 13U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(11U, 13U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(11U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(11U, 13U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(32U, 32U), 1, DataType::F32),
               }),
               make("HiddenStateInfo", { TensorInfo(TensorShape(11U, 13U), 1, DataType::F32),
                                                             TensorInfo(TensorShape(11U, 13U), 1, DataType::F32),
                                                             TensorInfo(TensorShape(11U, 13U), 1, DataType::F32),
                                                             TensorInfo(TensorShape(11U, 13U), 1, DataType::F32),
                                                             TensorInfo(TensorShape(11U, 13U), 1, DataType::F32),
                                                             TensorInfo(TensorShape(11U, 13U), 1, DataType::F32),
                                                             TensorInfo(TensorShape(11U, 13U, 2U), 1, DataType::F32),
                                                             TensorInfo(TensorShape(32U, 32U), 1, DataType::F32),
               }),
               make("ActivationInfo", { ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
                                                            ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
                                                            ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
                                                            ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
                                                            ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
                                                            ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
                                                            ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
                                                            ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
               }),
               make("Expected", { false, false, false, false, false, false, false, true })
               ),
               input_info, weights_info, recurrent_weights_info, bias_info, output_info, hidden_output_info, info, expected)
{
    ARM_COMPUTE_EXPECT(bool(NERNNLayer::validate(&input_info.clone()->set_is_resizable(false), &weights_info.clone()->set_is_resizable(false), &recurrent_weights_info.clone()->set_is_resizable(false), &bias_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), &hidden_output_info.clone()->set_is_resizable(false), info)) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using NERNNLayerFixture = RNNLayerValidationFixture<Tensor, Accessor, NERNNLayer, T>;

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NERNNLayerFixture<float>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallRNNLayerDataset(), make("DataType", DataType::F32)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END() // FP32

#ifdef ARM_COMPUTE_ENABLE_FP16
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NERNNLayerFixture<half>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallRNNLayerDataset(), make("DataType", DataType::F16)))
{
    if (CPUInfo::get().has_fp16())
    {
        // Validate output
        validate(Accessor(_target), _reference, tolerance_f16, 0.02f, abs_tolerance_f16);
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}
TEST_SUITE_END() // FP16
#endif           /* ARM_COMPUTE_ENABLE_FP16 */
TEST_SUITE_END() // RNNLayer
TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
