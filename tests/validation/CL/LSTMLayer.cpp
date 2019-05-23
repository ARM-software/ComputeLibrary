/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#include "arm_compute/runtime/CL/functions/CLLSTMLayer.h"
#include "tests/CL/CLAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/LSTMLayerDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/LSTMLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
RelativeTolerance<float> tolerance_f32(0.001f);
RelativeTolerance<half>  tolerance_f16(half(0.1));
} // namespace

TEST_SUITE(CL)
TEST_SUITE(LSTMLayer)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(zip(zip(zip(zip(zip(
               framework::dataset::make("InputInfo", { TensorInfo(TensorShape(8U, 2U), 1, DataType::U8),      // Wrong data type
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32), // Wrong input size
                                                       TensorInfo(TensorShape(8U, 2U), 1, DataType::F32),     // Wrong input weights size
                                                       TensorInfo(TensorShape(8U, 2U), 1, DataType::F32),     // Wrong recurrent weights size
                                                       TensorInfo(TensorShape(8U, 2U), 1, DataType::F32),     // Wrong cell bias size
                                                       TensorInfo(TensorShape(8U, 2U), 1, DataType::F32),     // Wrong cell state size
                                                       TensorInfo(TensorShape(8U, 2U), 1, DataType::F32),     // Wrong output size
                                                       TensorInfo(TensorShape(8U, 2U), 1, DataType::F32),     // Wrong scratch size
               }),
               framework::dataset::make("InputWeightsInfo", { TensorInfo(TensorShape(8U, 16U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(8U, 16U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(27U, 11U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(8U, 16U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(8U, 16U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(8U, 16U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(8U, 16U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(8U, 16U), 1, DataType::F32),
               })),
               framework::dataset::make("RecurrentWeightsInfo", { TensorInfo(TensorShape(16U, 16U), 1, DataType::F32),
                                                                  TensorInfo(TensorShape(16U, 16U), 1, DataType::F32),
                                                                  TensorInfo(TensorShape(16U, 16U), 1, DataType::F32),
                                                                  TensorInfo(TensorShape(25U, 11U, 2U), 1, DataType::F32),
                                                                  TensorInfo(TensorShape(16U, 16U), 1, DataType::F32),
                                                                  TensorInfo(TensorShape(16U, 16U), 1, DataType::F32),
                                                                  TensorInfo(TensorShape(16U, 16U), 1, DataType::F32),
                                                                  TensorInfo(TensorShape(16U, 16U), 1, DataType::F32),
               })),
               framework::dataset::make("CellBiasInfo", { TensorInfo(TensorShape(16U), 1, DataType::F32),
                                                      TensorInfo(TensorShape(16U), 1, DataType::F32),
                                                      TensorInfo(TensorShape(16U), 1, DataType::F32),
                                                      TensorInfo(TensorShape(16U), 1, DataType::F32),
                                                      TensorInfo(TensorShape(30U), 1, DataType::F32),
                                                      TensorInfo(TensorShape(16U), 1, DataType::F32),
                                                      TensorInfo(TensorShape(16U), 1, DataType::F32),
                                                      TensorInfo(TensorShape(16U), 1, DataType::F32),
               })),
               framework::dataset::make("ProjectionBiasInfo", { TensorInfo(TensorShape(16U), 1, DataType::F32),
                                                      TensorInfo(TensorShape(16U), 1, DataType::F32),
                                                      TensorInfo(TensorShape(16U), 1, DataType::F32),
                                                      TensorInfo(TensorShape(16U), 1, DataType::F32),
                                                      TensorInfo(TensorShape(16U), 1, DataType::F32),
                                                      TensorInfo(TensorShape(16U), 1, DataType::F32),
                                                      TensorInfo(TensorShape(16U), 1, DataType::F32),
                                                      TensorInfo(TensorShape(16U), 1, DataType::F32),
               })),
               framework::dataset::make("CellStateInfo", { TensorInfo(TensorShape(16U, 2U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(16U, 2U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(16U, 2U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(16U, 2U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(16U, 2U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(11U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(16U, 2U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(16U, 2U), 1, DataType::F32),
               })),
               framework::dataset::make("OutputInfo", { TensorInfo(TensorShape(16U, 2U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(16U, 2U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(16U, 2U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(16U, 2U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(16U, 2U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(16U, 2U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(11U, 2U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(16U, 2U), 1, DataType::F32),
               })),
               framework::dataset::make("ScratchInfo", { TensorInfo(TensorShape(64U, 2U), 1, DataType::F32),
                                                             TensorInfo(TensorShape(64U, 2U), 1, DataType::F32),
                                                             TensorInfo(TensorShape(64U, 2U), 1, DataType::F32),
                                                             TensorInfo(TensorShape(64U, 2U), 1, DataType::F32),
                                                             TensorInfo(TensorShape(64U, 2U), 1, DataType::F32),
                                                             TensorInfo(TensorShape(64U, 2U), 1, DataType::F32),
                                                             TensorInfo(TensorShape(64U, 2U), 1, DataType::F32),
                                                             TensorInfo(TensorShape(12U, 2U), 1, DataType::F32),
               })),
               framework::dataset::make("ActivationInfo", { ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
                                                            ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
                                                            ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
                                                            ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
                                                            ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
                                                            ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
                                                            ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
                                                            ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
               })),
               framework::dataset::make("Expected", { false, false, false, false, false, false, false, false })),
               input_info, input_weights_info, recurrent_weights_info, cell_bias_info, projection_bias_info, cell_state_info, output_info, scratch_info, info, expected)
{
    LSTMParams<ITensorInfo> lstm_params_info;
    lstm_params_info.set_peephole_params(&cell_bias_info, &cell_bias_info)
                    .set_projection_params(&recurrent_weights_info, &projection_bias_info)
                    .set_cifg_params(&input_weights_info, &recurrent_weights_info, &cell_bias_info, &cell_bias_info);

    ARM_COMPUTE_EXPECT(bool(CLLSTMLayer::validate(&input_info.clone()->set_is_resizable(false), &input_weights_info.clone()->set_is_resizable(false), &input_weights_info.clone()->set_is_resizable(false),
                                                  &input_weights_info.clone()->set_is_resizable(false), &recurrent_weights_info.clone()->set_is_resizable(false), &recurrent_weights_info.clone()->set_is_resizable(false),
                                                  &recurrent_weights_info.clone()->set_is_resizable(false), &cell_bias_info.clone()->set_is_resizable(false), &cell_bias_info.clone()->set_is_resizable(false),
                                                  &cell_bias_info.clone()->set_is_resizable(false),
                                                  &output_info.clone()->set_is_resizable(false), &cell_state_info.clone()->set_is_resizable(false),
                                                  &scratch_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), &cell_state_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false),
                                                  lstm_params_info, info, 0.05, 0.9)) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using CLLSTMLayerFixture = LSTMLayerValidationFixture<CLTensor, CLAccessor, CLLSTMLayer, LSTMParams<ICLTensor>, T>;

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLLSTMLayerFixture<float>, framework::DatasetMode::ALL, combine(combine(combine(datasets::SmallLSTMLayerDataset(), framework::dataset::make("DataType",
                                                                                                                 DataType::F32)),
                                                                                                         framework::dataset::make("ProjectionOpt", { true, false })),
                                                                                                 framework::dataset::make("PeepholeOpt", { true, false })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
    validate(CLAccessor(_target_scratch), _reference_scratch, tolerance_f32);
}
TEST_SUITE_END() // FP32

TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLLSTMLayerFixture<half>, framework::DatasetMode::ALL, combine(combine(combine(datasets::SmallLSTMLayerDataset(), framework::dataset::make("DataType", DataType::F16)),
                                                                                                        framework::dataset::make("ProjectionOpt", { true, false })),
                                                                                                framework::dataset::make("PeepholeOpt", { true, false })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
    validate(CLAccessor(_target_scratch), _reference_scratch, tolerance_f16);
}
TEST_SUITE_END() // FP16
TEST_SUITE_END() // LSTMLayer
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
