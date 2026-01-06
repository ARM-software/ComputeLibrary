/*
 * Copyright (c) 2019-2021, 2024-2026 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEFuseBatchNormalization.h"

#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/framework/Macros.h"
#include "tests/Globals.h"
#include "tests/NEON/Accessor.h"
#include "tests/validation/fixtures/FuseBatchNormalizationFixture.h"
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
AbsoluteTolerance<float> absolute_tolerance_f32(0.001f);
#ifdef ARM_COMPUTE_ENABLE_FP16
AbsoluteTolerance<float> absolute_tolerance_f16(0.2f);
#endif /* ARM_COMPUTE_ENABLE_FP16 */
} // namespace

template <typename T>
using NEFuseBatchNormalizationConvFixture =
    FuseBatchNormalizationFixture<Tensor, Accessor, NEFuseBatchNormalization, 4, T>;
template <typename T>
using NEFuseBatchNormalizationDWCFixture =
    FuseBatchNormalizationFixture<Tensor, Accessor, NEFuseBatchNormalization, 3, T>;

// *INDENT-OFF*
// clang-format off

/** Shapes to test - Precommit */
const auto shape_conv_values_precommit = concat(datasets::Small4DShapes(), datasets::Small3DShapes());

/** Shapes to test - Nightly */
const auto shape_conv_values_nightly = concat(datasets::Large4DShapes(), datasets::Large3DShapes());

/** Data layout to test */
const auto data_layout_values = make("DataLayout", { DataLayout::NHWC, DataLayout::NCHW });

/** In-place flags to test */
const auto in_place_values = make("InPlace", { true, false });

/** With bias flags to test */
const auto with_bias_values = make("WithBias", { true, false });

/** With gamma flags to test */
const auto with_gamma_values = make("WithGamma", { true, false });

/** With beta flags to test */
const auto with_beta_values = make("WithBeta", { true, false });

TEST_SUITE(NEON)
TEST_SUITE(FuseBatchNormalization)
TEST_SUITE(Convolution)
TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEFuseBatchNormalizationConvFixture<float>, framework::DatasetMode::PRECOMMIT,
                                        combine(
                                                        shape_conv_values_precommit,
                                                        make("DataType", { DataType::F32 }),
                                                        data_layout_values,
                                                        in_place_values,
                                                        with_bias_values,
                                                        with_gamma_values,
                                                        with_beta_values))
{
    // Validate outputs
    validate(Accessor(_target_w), _reference_w, absolute_tolerance_f32);
    validate(Accessor(_target_b), _reference_b, absolute_tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEFuseBatchNormalizationConvFixture<float>, framework::DatasetMode::NIGHTLY,
                                        combine(
                                                        shape_conv_values_nightly,
                                                        make("DataType", { DataType::F32 }),
                                                        data_layout_values,
                                                        in_place_values,
                                                        with_bias_values,
                                                        with_gamma_values,
                                                        with_beta_values))
{
    // Validate outputs
    validate(Accessor(_target_w), _reference_w, absolute_tolerance_f32);
    validate(Accessor(_target_b), _reference_b, absolute_tolerance_f32);
}
TEST_SUITE_END() // FP32
#ifdef ARM_COMPUTE_ENABLE_FP16
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEFuseBatchNormalizationConvFixture<half>, framework::DatasetMode::PRECOMMIT,
                                        combine(
                                                        shape_conv_values_precommit,
                                                        make("DataType", { DataType::F16 }),
                                                        data_layout_values,
                                                        in_place_values,
                                                        with_bias_values,
                                                        with_gamma_values,
                                                        with_beta_values))
{
    if(CPUInfo::get().has_fp16())
    {
        // Validate output
        validate(Accessor(_target_w), _reference_w, absolute_tolerance_f16);
        validate(Accessor(_target_b), _reference_b, absolute_tolerance_f16);
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEFuseBatchNormalizationConvFixture<half>, framework::DatasetMode::NIGHTLY,
                                        combine(
                                                        shape_conv_values_nightly,
                                                        make("DataType", { DataType::F16 }),
                                                        data_layout_values,
                                                        in_place_values,
                                                        with_bias_values,
                                                        with_gamma_values,
                                                        with_beta_values))
{
    if(CPUInfo::get().has_fp16())
    {
        // Validate output
        validate(Accessor(_target_w), _reference_w, absolute_tolerance_f16);
        validate(Accessor(_target_b), _reference_b, absolute_tolerance_f16);
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}
TEST_SUITE_END() // FP16
#endif /* ARM_COMPUTE_ENABLE_FP16 */
TEST_SUITE_END() // Float
TEST_SUITE_END() // Convolution
TEST_SUITE(DepthwiseConvolution)
TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEFuseBatchNormalizationDWCFixture<float>, framework::DatasetMode::PRECOMMIT,
                                        combine(
                                                        datasets::Small3DShapes(),
                                                        make("DataType", { DataType::F32 }),
                                                        data_layout_values,
                                                        in_place_values,
                                                        with_bias_values,
                                                        with_gamma_values,
                                                        with_beta_values))
{
    // Validate outputs
    validate(Accessor(_target_w), _reference_w, absolute_tolerance_f32);
    validate(Accessor(_target_b), _reference_b, absolute_tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEFuseBatchNormalizationDWCFixture<float>, framework::DatasetMode::NIGHTLY,
                                        combine(
                                                        datasets::Large3DShapes(),
                                                        make("DataType", { DataType::F32 }),
                                                        data_layout_values,
                                                        in_place_values,
                                                        with_bias_values,
                                                        with_gamma_values,
                                                        with_beta_values))
{
    // Validate outputs
    validate(Accessor(_target_w), _reference_w, absolute_tolerance_f32);
    validate(Accessor(_target_b), _reference_b, absolute_tolerance_f32);
}

TEST_SUITE_END() // FP32
#ifdef ARM_COMPUTE_ENABLE_FP16
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEFuseBatchNormalizationDWCFixture<half>, framework::DatasetMode::PRECOMMIT,
                                        combine(
                                                        datasets::Small3DShapes(),
                                                        make("DataType", { DataType::F16 }),
                                                        data_layout_values,
                                                        in_place_values,
                                                        with_bias_values,
                                                        with_gamma_values,
                                                        with_beta_values))
{
    if(CPUInfo::get().has_fp16())
    {
        // Validate output
        validate(Accessor(_target_w), _reference_w, absolute_tolerance_f16);
        validate(Accessor(_target_b), _reference_b, absolute_tolerance_f16);
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEFuseBatchNormalizationDWCFixture<half>, framework::DatasetMode::NIGHTLY,
                                        combine(
                                                        datasets::Large3DShapes(),
                                                        make("DataType", { DataType::F16 }),
                                                        data_layout_values,
                                                        in_place_values,
                                                        with_bias_values,
                                                        with_gamma_values,
                                                        with_beta_values))
{
    if(CPUInfo::get().has_fp16())
    {
        // Validate output
        validate(Accessor(_target_w), _reference_w, absolute_tolerance_f16);
        validate(Accessor(_target_b), _reference_b, absolute_tolerance_f16);
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}

TEST_SUITE_END() // FP16
#endif /* ARM_COMPUTE_ENABLE_FP16 */
TEST_SUITE_END() // Float
TEST_SUITE_END() // DepthwiseConvolution
TEST_SUITE_END() // FuseBatchNormalization
TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
