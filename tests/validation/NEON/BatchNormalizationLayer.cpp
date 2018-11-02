/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/RandomBatchNormalizationLayerDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/BatchNormalizationLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
constexpr AbsoluteTolerance<float> tolerance_f32(0.00001f); /**< Tolerance value for comparing reference's output against implementation's output for DataType::F32 */
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
constexpr AbsoluteTolerance<float> tolerance_f16(0.01f); /**< Tolerance value for comparing reference's output against implementation's output for DataType::F16 */
#endif                                                   /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
const auto act_infos = framework::dataset::make("ActivationInfo",
{
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 8.f, 2.f),
});
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(BatchNormalizationLayer)

template <typename T>
using NEBatchNormalizationLayerFixture = BatchNormalizationLayerValidationFixture<Tensor, Accessor, NEBatchNormalizationLayer, T>;

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(combine(datasets::RandomBatchNormalizationLayerDataset(),
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
FIXTURE_DATA_TEST_CASE(Random, NEBatchNormalizationLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::RandomBatchNormalizationLayerDataset(),
                                                                                                                   combine(framework::dataset::make("UseBeta", { false, true }),
                                                                                                                           framework::dataset::make("UseGamma", { false, true }))),
                                                                                                                   act_infos),
                                                                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                                                                                   framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32, 0);
}
TEST_SUITE_END()

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(Random, NEBatchNormalizationLayerFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::RandomBatchNormalizationLayerDataset(),
                                                                                                                  combine(framework::dataset::make("UseBeta", { false, true }),
                                                                                                                          framework::dataset::make("UseGamma", { false, true }))),
                                                                                                                  framework::dataset::make("ActivationInfo", ActivationLayerInfo())),
                                                                                                                  framework::dataset::make("DataType", DataType::F16)),
                                                                                                                  framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f16, 0);
}
TEST_SUITE_END()
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
