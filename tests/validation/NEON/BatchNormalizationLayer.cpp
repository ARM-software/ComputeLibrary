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
#include "arm_compute/runtime/NEON/functions/NEBatchNormalizationLayer.h"
#include "arm_compute/runtime/NEON/functions/NEConvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NEFuseBatchNormalization.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/RandomBatchNormalizationLayerDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/datasets/SmallConvolutionLayerDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/BatchNormalizationLayerFixture.h"
#include "tests/validation/fixtures/BatchNormalizationLayerFusionFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
RelativeTolerance<float>           rel_tolerance_f32(0.05f);   /**< Tolerance value for comparing reference's output against implementation's output for DataType::F32 */
constexpr AbsoluteTolerance<float> abs_tolerance_f32(0.0001f); /**< Tolerance value for comparing reference's output against implementation's output for DataType::F32 */
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
constexpr AbsoluteTolerance<float> tolerance_f16(0.01f); /**< Tolerance value for comparing reference's output against implementation's output for DataType::F16 */
#endif                                                   // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
const auto act_infos = framework::dataset::make("ActivationInfo",
{
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 8.f, 2.f),
});
const auto common_fusion_dataset = combine(combine(combine(framework::dataset::make("UseBias", { false, true }),
                                                           framework::dataset::make("UseBeta", { false, true })),
                                                   framework::dataset::make("UseGamma", { false, true })),
                                           framework::dataset::make("Epsilon", { 0.001f }));
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(BatchNormalizationLayer)

template <typename T>
using NEBatchNormalizationLayerFixture = BatchNormalizationLayerValidationFixture<Tensor, Accessor, NEBatchNormalizationLayer, T>;

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(combine(datasets::SmallRandomBatchNormalizationLayerDataset(),
                                                                                   combine(framework::dataset::make("UseBeta", { false, true }), framework::dataset::make("UseGamma", { false, true }))),
                                                                           framework::dataset::make("DataType", { DataType::F32 })),
                                                                   framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
               shape0, shape1, epsilon, use_beta, use_gamma, dt, data_layout)
{
    TensorShape src_dst_shapes = shape0;
    if(data_layout == DataLayout::NHWC)
    {
        permute(src_dst_shapes, PermutationVector(2U, 0U, 1U));
    }

    // Create tensors
    Tensor src   = create_tensor<Tensor>(src_dst_shapes, dt, 1, QuantizationInfo(), data_layout);
    Tensor dst   = create_tensor<Tensor>(src_dst_shapes, dt, 1, QuantizationInfo(), data_layout);
    Tensor mean  = create_tensor<Tensor>(shape1, dt, 1);
    Tensor var   = create_tensor<Tensor>(shape1, dt, 1);
    Tensor beta  = create_tensor<Tensor>(shape1, dt, 1);
    Tensor gamma = create_tensor<Tensor>(shape1, dt, 1);

    // Create and Configure function
    NEBatchNormalizationLayer norm;
    Tensor                   *beta_ptr  = use_beta ? &beta : nullptr;
    Tensor                   *gamma_ptr = use_gamma ? &gamma : nullptr;
    norm.configure(&src, &dst, &mean, &var, beta_ptr, gamma_ptr, epsilon);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(src_dst_shapes);
    validate(dst.info()->valid_region(), valid_region);
}

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(
               framework::dataset::make("InputInfo", { TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),    // Window shrink
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),    // Mismatching data types
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),    // Mismatching data types
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),    // Invalid mean/var/beta/gamma shape
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),    // Fused activation's a < b
                                                     }),
               framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                     })),
               framework::dataset::make("MVBGInfo",{ TensorInfo(TensorShape(2U), 1, DataType::F32),
                                                     TensorInfo(TensorShape(2U), 1, DataType::F32),
                                                     TensorInfo(TensorShape(2U), 1, DataType::F16),
                                                     TensorInfo(TensorShape(2U), 1, DataType::F32),
                                                     TensorInfo(TensorShape(5U), 1, DataType::F32),
                                                     TensorInfo(TensorShape(2U), 1, DataType::F32),
                                                   })),
               framework::dataset::make("ActivationLayerInfo",{ ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
                                                    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
                                                     ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f),
                                                     ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f),
                                                     ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 6.f),
                                                     ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 2.f, 6.f),
                                                   })),
               framework::dataset::make("Expected", { true, false, false, false, false, false})),
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
FIXTURE_DATA_TEST_CASE(RandomSmall, NEBatchNormalizationLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::SmallRandomBatchNormalizationLayerDataset(),
                                                                                                                        combine(framework::dataset::make("UseBeta", { false, true }),
                                                                                                                                framework::dataset::make("UseGamma", { false, true }))),
                                                                                                                        act_infos),
                                                                                                                        framework::dataset::make("DataType", DataType::F32)),
                                                                                                                        framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference, abs_tolerance_f32, 0);
}
FIXTURE_DATA_TEST_CASE(RandomLarge, NEBatchNormalizationLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::LargeRandomBatchNormalizationLayerDataset(),
                                                                                                                      combine(framework::dataset::make("UseBeta", { false, true }),
                                                                                                                              framework::dataset::make("UseGamma", { false, true }))),
                                                                                                                      act_infos),
                                                                                                                      framework::dataset::make("DataType", DataType::F32)),
                                                                                                                      framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference, abs_tolerance_f32, 0);
}
TEST_SUITE_END() // F32

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RandomSmall, NEBatchNormalizationLayerFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::SmallRandomBatchNormalizationLayerDataset(),
                                                                                                                       combine(framework::dataset::make("UseBeta", { false, true }),
                                                                                                                               framework::dataset::make("UseGamma", { false, true }))),
                                                                                                                       framework::dataset::make("ActivationInfo", ActivationLayerInfo())),
                                                                                                                       framework::dataset::make("DataType", DataType::F16)),
                                                                                                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f16, 0);
}

FIXTURE_DATA_TEST_CASE(RandomLarge, NEBatchNormalizationLayerFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::LargeRandomBatchNormalizationLayerDataset(),
                                                                                                                       combine(framework::dataset::make("UseBeta", { false, true }),
                                                                                                                               framework::dataset::make("UseGamma", { false, true }))),
                                                                                                                       framework::dataset::make("ActivationInfo", ActivationLayerInfo())),
                                                                                                                       framework::dataset::make("DataType", DataType::F16)),
                                                                                                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f16, 0);
}
TEST_SUITE_END() // FP16
#endif           /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
TEST_SUITE_END() // Float

TEST_SUITE_END() // BatchNormalizationLayer

TEST_SUITE(BatchNormalizationLayerFusion)
template <typename T>
using NEBatchNormalizationLayerFusionFixture = BatchNormalizationLayerFusionValidationFixture<Tensor, Accessor, NEConvolutionLayer, NEFuseBatchNormalization, T>;

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(
               framework::dataset::make("Weights", { TensorInfo(TensorShape(32U, 13U, 2U, 2U), 1, DataType::F32),      // Valid
                                                       TensorInfo(TensorShape(32U, 13U, 2U, 2U), 1, DataType::F32),    // Mismatching data types
                                                       TensorInfo(TensorShape(32U, 13U, 2U, 2U), 1, DataType::F16),    // Mismatching data types
                                                       TensorInfo(TensorShape(32U, 13U, 2U, 1U), 1, DataType::F32),    // Invalid mean/var/beta/gamma shape
                                                     }),
               framework::dataset::make("MVBGInfo",{ TensorInfo(TensorShape(2U), 1, DataType::F32),
                                                     TensorInfo(TensorShape(2U), 1, DataType::F16),
                                                     TensorInfo(TensorShape(2U), 1, DataType::F32),
                                                     TensorInfo(TensorShape(5U), 1, DataType::F32),
                                                   })),
               framework::dataset::make("Expected", { true, false, false, false})),
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
FIXTURE_DATA_TEST_CASE(RunSmall, NEBatchNormalizationLayerFusionFixture<float>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(datasets::SmallConvolutionLayerDataset(), common_fusion_dataset),
                                       framework::dataset::make("DataType", DataType::F32)),
                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
}
TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float

TEST_SUITE_END() // BatchNormalizationLayerFusion
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
