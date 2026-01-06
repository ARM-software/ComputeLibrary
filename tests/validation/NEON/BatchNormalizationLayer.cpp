/*
 * Copyright (c) 2017-2021, 2024-2026 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEBatchNormalizationLayer.h"
#include "arm_compute/runtime/NEON/functions/NEConvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NEFuseBatchNormalization.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"

#include "tests/datasets/RandomBatchNormalizationLayerDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/datasets/SmallConvolutionLayerDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/framework/Macros.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/validation/fixtures/BatchNormalizationLayerFixture.h"
#include "tests/validation/fixtures/BatchNormalizationLayerFusionFixture.h"
#include "tests/validation/Helpers.h"
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
#ifndef ARM_COMPUTE_ADDRESS_SANITIZER_BUILD
RelativeTolerance<float> rel_tolerance_f32(
    0.05f); /**< Tolerance value for comparing reference's output against implementation's output for DataType::F32 */
#endif      // ARM_COMPUTE_ADDRESS_SANITIZER_BUILD
constexpr AbsoluteTolerance<float> abs_tolerance_f32(
    0.0001f); /**< Tolerance value for comparing reference's output against implementation's output for DataType::F32 */
#ifdef ARM_COMPUTE_ENABLE_FP16
constexpr AbsoluteTolerance<float> abs_tolerance_f16(
    0.015f); /**< Tolerance value for comparing reference's output against implementation's output for DataType::F16 */
#endif       // ARM_COMPUTE_ENABLE_FP16

const auto act_infos             = make("ActivationInfo",
                                        {
                                            ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
                                            ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f),
                                            ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 8.f, 2.f),
                            });
const auto common_fusion_dataset = combine(make("UseBias", {false, true}),
                                           make("UseBeta", {false, true}),
                                           make("UseGamma", {false, true}),
                                           make("Epsilon", {0.001f}));
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(BatchNormalizationLayer)

template <typename T>
using NEBatchNormalizationLayerFixture =
    BatchNormalizationLayerValidationFixture<Tensor, Accessor, NEBatchNormalizationLayer, T>;

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(
               make("InputInfo", { TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),    // Mismatching data types
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),    // Mismatching data types
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),    // Invalid mean/var/beta/gamma shape
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),    // Fused activation's a < b
                                                     }),
               make("OutputInfo",{ TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                     }),
               make("MVBGInfo",{ TensorInfo(TensorShape(2U), 1, DataType::F32),
                                                     TensorInfo(TensorShape(2U), 1, DataType::F16),
                                                     TensorInfo(TensorShape(2U), 1, DataType::F32),
                                                     TensorInfo(TensorShape(5U), 1, DataType::F32),
                                                     TensorInfo(TensorShape(2U), 1, DataType::F32),
                                                   }),
               make("ActivationLayerInfo",{ ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
                                                     ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f),
                                                     ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f),
                                                     ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 6.f),
                                                     ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 2.f, 6.f),
                                                   }),
               make("Expected", { true, false, false, false, false})
               ),
               input_info, output_info, mvbg_info, act_info, expected)
{
    const auto &mean_info = mvbg_info;
    const auto &var_info = mvbg_info;
    const auto &beta_info = mvbg_info;
    const auto &gamma_info = mvbg_info;
    bool has_error = bool(NEBatchNormalizationLayer::validate(
            &input_info.clone()->set_is_resizable(false), output_info.total_size() ? &output_info.clone()->set_is_resizable(false) : nullptr,
            &mean_info.clone()->set_is_resizable(false), &var_info.clone()->set_is_resizable(false),
            &beta_info.clone()->set_is_resizable(false), &gamma_info.clone()->set_is_resizable(false), 1.f, act_info));
    ARM_COMPUTE_EXPECT(has_error == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RandomSmall,
                       NEBatchNormalizationLayerFixture<float>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallRandomBatchNormalizationLayerDataset(),
                               make("UseBeta", {false, true}),
                               make("UseGamma", {false, true}),
                               act_infos,
                               make("DataType", DataType::F32),
                               make("DataLayout", {DataLayout::NCHW, DataLayout::NHWC})))
{
    // Validate output
    validate(Accessor(_target), _reference, abs_tolerance_f32, 0);
}
FIXTURE_DATA_TEST_CASE(RandomLarge,
                       NEBatchNormalizationLayerFixture<float>,
                       framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeRandomBatchNormalizationLayerDataset(),
                               make("UseBeta", {false, true}),
                               make("UseGamma", {false, true}),
                               act_infos,
                               make("DataType", DataType::F32),
                               make("DataLayout", {DataLayout::NCHW, DataLayout::NHWC})))
{
    // Validate output
    validate(Accessor(_target), _reference, abs_tolerance_f32, 0);
}
TEST_SUITE_END() // F32

#ifdef ARM_COMPUTE_ENABLE_FP16
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RandomSmall,
                       NEBatchNormalizationLayerFixture<half>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallRandomBatchNormalizationLayerDataset(),
                               make("UseBeta", {false, true}),
                               make("UseGamma", {false, true}),
                               make("ActivationInfo", ActivationLayerInfo()),
                               make("DataType", DataType::F16),
                               make("DataLayout", {DataLayout::NCHW, DataLayout::NHWC})))
{
    if (CPUInfo::get().has_fp16())
    {
        // Validate output
        validate(Accessor(_target), _reference, abs_tolerance_f16, 0);
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}

FIXTURE_DATA_TEST_CASE(RandomLarge,
                       NEBatchNormalizationLayerFixture<half>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::LargeRandomBatchNormalizationLayerDataset(),
                               make("UseBeta", {false, true}),
                               make("UseGamma", {false, true}),
                               make("ActivationInfo", ActivationLayerInfo()),
                               make("DataType", DataType::F16),
                               make("DataLayout", {DataLayout::NCHW, DataLayout::NHWC})))
{
    if (CPUInfo::get().has_fp16())
    {
        // Validate output
        validate(Accessor(_target), _reference, abs_tolerance_f16, 0);
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}
TEST_SUITE_END() // FP16
#endif           /* ARM_COMPUTE_ENABLE_FP16 */
TEST_SUITE_END() // Float

TEST_SUITE_END() // BatchNormalizationLayer

#ifndef ARM_COMPUTE_ADDRESS_SANITIZER_BUILD
TEST_SUITE(BatchNormalizationLayerFusion)
template <typename T>
using NEBatchNormalizationLayerFusionFixture =
    BatchNormalizationLayerFusionValidationFixture<Tensor, Accessor, NEConvolutionLayer, NEFuseBatchNormalization, T>;

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(
               make("Weights", { TensorInfo(TensorShape(32U, 13U, 2U, 2U), 1, DataType::F32),      // Valid
                                                       TensorInfo(TensorShape(32U, 13U, 2U, 2U), 1, DataType::F32),    // Mismatching data types
                                                       TensorInfo(TensorShape(32U, 13U, 2U, 2U), 1, DataType::F16),    // Mismatching data types
                                                       TensorInfo(TensorShape(32U, 13U, 2U, 1U), 1, DataType::F32),    // Invalid mean/var/beta/gamma shape
                                                     }),
               make("MVBGInfo",{ TensorInfo(TensorShape(2U), 1, DataType::F32),
                                                     TensorInfo(TensorShape(2U), 1, DataType::F16),
                                                     TensorInfo(TensorShape(2U), 1, DataType::F32),
                                                     TensorInfo(TensorShape(5U), 1, DataType::F32),
                                                   }),
               make("Expected", { true, false, false, false})
               ),
               weights_info, mvbg_info, expected)
{
    const auto &weights_in_info = weights_info;
    const auto &mean_info = mvbg_info;
    const auto &var_info = mvbg_info;
    const auto &fused_weights_info = weights_info;
    const auto &fused_bias_info = mvbg_info;
    const auto &conv_bias_info = mvbg_info;
    const auto &beta_info = mvbg_info;
    const auto &gamma_info = mvbg_info;
    bool has_error = bool(NEFuseBatchNormalization::validate(
            &weights_in_info.clone()->set_is_resizable(false), &mean_info.clone()->set_is_resizable(false),
            &var_info.clone()->set_is_resizable(false), &fused_weights_info.clone()->set_is_resizable(false),
            &fused_bias_info.clone()->set_is_resizable(false), &conv_bias_info.clone()->set_is_resizable(false),
            &beta_info.clone()->set_is_resizable(false), &gamma_info.clone()->set_is_resizable(false), 1.f));
    ARM_COMPUTE_EXPECT(has_error == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEBatchNormalizationLayerFusionFixture<float>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallConvolutionLayerDataset(),
                               common_fusion_dataset,
                               make("DataType", DataType::F32),
                               make("DataLayout", {DataLayout::NCHW, DataLayout::NHWC})))
{
    // Validate output
    validate(Accessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
}
TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float

TEST_SUITE_END() // BatchNormalizationLayerFusion
#endif           // ARM_COMPUTE_ADDRESS_SANITIZER_BUILD
TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
