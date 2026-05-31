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
#include "arm_compute/runtime/NEON/functions/NELSTMLayer.h"

#include "tests/datasets/LSTMLayerDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/framework/Macros.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/validation/fixtures/LSTMLayerFixture.h"
#include "tests/validation/Validation.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
RelativeTolerance<float> tolerance_f32(0.00001f);
RelativeTolerance<half>  tolerance_f16(half(0.1));
} // namespace

using framework::dataset::make;

TEST_SUITE(NEON)
TEST_SUITE(LSTMLayer)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(
    make("InputInfo", {
        TensorInfo(TensorShape(8U, 2U), 1, DataType::U8),      // Wrong data type
        TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32), // Wrong input size
        TensorInfo(TensorShape(8U, 2U), 1, DataType::F32),     // Wrong input weights size
        TensorInfo(TensorShape(8U, 2U), 1, DataType::F32),     // Wrong recurrent weights size
        TensorInfo(TensorShape(8U, 2U), 1, DataType::F32),     // Wrong cell bias size
        TensorInfo(TensorShape(8U, 2U), 1, DataType::F32),     // Wrong cell state size
        TensorInfo(TensorShape(8U, 2U), 1, DataType::F32),     // Wrong output size
        TensorInfo(TensorShape(8U, 2U), 1, DataType::F32),     // Wrong scratch size
    }),
    make("InputWeightsInfo", {
        TensorInfo(TensorShape(8U, 16U), 1, DataType::F32),
        TensorInfo(TensorShape(8U, 16U), 1, DataType::F32),
        TensorInfo(TensorShape(27U, 11U, 2U), 1, DataType::F32),
        TensorInfo(TensorShape(8U, 16U), 1, DataType::F32),
        TensorInfo(TensorShape(8U, 16U), 1, DataType::F32),
        TensorInfo(TensorShape(8U, 16U), 1, DataType::F32),
        TensorInfo(TensorShape(8U, 16U), 1, DataType::F32),
        TensorInfo(TensorShape(8U, 16U), 1, DataType::F32),
    }),
    make("RecurrentWeightsInfo", {
        TensorInfo(TensorShape(16U, 16U), 1, DataType::F32),
        TensorInfo(TensorShape(16U, 16U), 1, DataType::F32),
        TensorInfo(TensorShape(16U, 16U), 1, DataType::F32),
        TensorInfo(TensorShape(25U, 11U, 2U), 1, DataType::F32),
        TensorInfo(TensorShape(16U, 16U), 1, DataType::F32),
        TensorInfo(TensorShape(16U, 16U), 1, DataType::F32),
        TensorInfo(TensorShape(16U, 16U), 1, DataType::F32),
        TensorInfo(TensorShape(16U, 16U), 1, DataType::F32),
    }),
    make("CellBiasInfo", {
        TensorInfo(TensorShape(16U), 1, DataType::F32),
        TensorInfo(TensorShape(16U), 1, DataType::F32),
        TensorInfo(TensorShape(16U), 1, DataType::F32),
        TensorInfo(TensorShape(16U), 1, DataType::F32),
        TensorInfo(TensorShape(30U), 1, DataType::F32),
        TensorInfo(TensorShape(16U), 1, DataType::F32),
        TensorInfo(TensorShape(16U), 1, DataType::F32),
        TensorInfo(TensorShape(16U), 1, DataType::F32),
    }),
    make("ProjectionBiasInfo", {
        TensorInfo(TensorShape(16U), 1, DataType::F32),
        TensorInfo(TensorShape(16U), 1, DataType::F32),
        TensorInfo(TensorShape(16U), 1, DataType::F32),
        TensorInfo(TensorShape(16U), 1, DataType::F32),
        TensorInfo(TensorShape(16U), 1, DataType::F32),
        TensorInfo(TensorShape(16U), 1, DataType::F32),
        TensorInfo(TensorShape(16U), 1, DataType::F32),
        TensorInfo(TensorShape(16U), 1, DataType::F32),
    }),
    make("CellStateInfo", {
        TensorInfo(TensorShape(16U, 2U), 1, DataType::F32),
        TensorInfo(TensorShape(16U, 2U), 1, DataType::F32),
        TensorInfo(TensorShape(16U, 2U), 1, DataType::F32),
        TensorInfo(TensorShape(16U, 2U), 1, DataType::F32),
        TensorInfo(TensorShape(16U, 2U), 1, DataType::F32),
        TensorInfo(TensorShape(11U), 1, DataType::F32),
        TensorInfo(TensorShape(16U, 2U), 1, DataType::F32),
        TensorInfo(TensorShape(16U, 2U), 1, DataType::F32),
    }),
    make("OutputInfo", {
        TensorInfo(TensorShape(16U, 2U), 1, DataType::F32),
        TensorInfo(TensorShape(16U, 2U), 1, DataType::F32),
        TensorInfo(TensorShape(16U, 2U), 1, DataType::F32),
        TensorInfo(TensorShape(16U, 2U), 1, DataType::F32),
        TensorInfo(TensorShape(16U, 2U), 1, DataType::F32),
        TensorInfo(TensorShape(16U, 2U), 1, DataType::F32),
        TensorInfo(TensorShape(11U, 2U), 1, DataType::F32),
        TensorInfo(TensorShape(16U, 2U), 1, DataType::F32),
    }),
    make("ScratchInfo", {
        TensorInfo(TensorShape(64U, 2U), 1, DataType::F32),
        TensorInfo(TensorShape(64U, 2U), 1, DataType::F32),
        TensorInfo(TensorShape(64U, 2U), 1, DataType::F32),
        TensorInfo(TensorShape(64U, 2U), 1, DataType::F32),
        TensorInfo(TensorShape(64U, 2U), 1, DataType::F32),
        TensorInfo(TensorShape(64U, 2U), 1, DataType::F32),
        TensorInfo(TensorShape(64U, 2U), 1, DataType::F32),
        TensorInfo(TensorShape(12U, 2U), 1, DataType::F32),
    }),
    make("ActivationInfo", {
        ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
        ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
        ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
        ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
        ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
        ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
        ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
        ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
    }),
    make("Expected", { false, false, false, false, false, false, false, false })
    ),
    input_info, input_weights_info, recurrent_weights_info, cell_bias_info,
        projection_bias_info, cell_state_info, output_info, scratch_info, info, expected)
{
    LSTMParams<ITensorInfo> lstm_params_info;
    auto cell_bias_clone = cell_bias_info.clone();
    lstm_params_info.set_peephole_params(cell_bias_clone.get(), cell_bias_clone.get())
                    .set_projection_params(&recurrent_weights_info, &projection_bias_info)
                    .set_cifg_params(&input_weights_info, &recurrent_weights_info, cell_bias_clone.get(), cell_bias_clone.get());

    ARM_COMPUTE_EXPECT(bool(NELSTMLayer::validate(&input_info.clone()->set_is_resizable(false), &input_weights_info.clone()->set_is_resizable(false), &input_weights_info.clone()->set_is_resizable(false),
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
using NELSTMLayerFixture = LSTMLayerValidationFixture<Tensor, Accessor, NELSTMLayer, LSTMParams<ITensor>, T>;

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NELSTMLayerFixture<float>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallLSTMLayerDataset(),
                               make("DataType", DataType::F32),
                               make("ProjectionOpt", {true, false}),
                               make("PeepholeOpt", {true, false}),
                               make("UseLayerNorm", {true, false}),
                               make("UseMemoryManager", {true, false})))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32);
    validate(Accessor(_target_scratch), _reference_scratch, tolerance_f32);
}
TEST_SUITE_END() // FP32

#ifdef ARM_COMPUTE_ENABLE_FP16
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NELSTMLayerFixture<half>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallLSTMLayerDataset(),
                               make("DataType", DataType::F16),
                               make("ProjectionOpt", {true, false}),
                               make("PeepholeOpt", {true, false}),
                               make("UseLayerNorm", {true, false}),
                               make("UseMemoryManager", {true, false})))
{
    if (CPUInfo::get().has_fp16())
    {
        // Validate output
        validate(Accessor(_target), _reference, tolerance_f16);
        validate(Accessor(_target_scratch), _reference_scratch, tolerance_f16);
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}
TEST_SUITE_END() // FP16
#endif           /* ARM_COMPUTE_ENABLE_FP16 */

TEST_SUITE_END() // LSTMLayer
TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
