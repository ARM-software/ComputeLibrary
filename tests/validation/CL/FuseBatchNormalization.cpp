/*
 * Copyright (c) 2019-2020 Arm Limited.
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
#include "arm_compute/runtime/CL/functions/CLFuseBatchNormalization.h"
#include "tests/CL/CLAccessor.h"
#include "tests/Globals.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/FuseBatchNormalizationFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
AbsoluteTolerance<float> absolute_tolerance_f32(0.001f);
AbsoluteTolerance<float> absolute_tolerance_f16(0.2f);
} // namespace

template <typename T>
using CLFuseBatchNormalizationConvFixture = FuseBatchNormalizationFixture<CLTensor, CLAccessor, CLFuseBatchNormalization, 4, T>;
template <typename T>
using CLFuseBatchNormalizationDWCFixture = FuseBatchNormalizationFixture<CLTensor, CLAccessor, CLFuseBatchNormalization, 3, T>;

// *INDENT-OFF*
// clang-format off

/** Shapes to test - Precommit */
const auto shape_conv_values_precommit = concat(datasets::Small4DShapes(), datasets::Small3DShapes());

/** Shapes to test - Nightly */
const auto shape_conv_values_nightly = concat(datasets::Large4DShapes(), datasets::Large3DShapes());

/** Data layout to test */
const auto data_layout_values = framework::dataset::make("DataLayout", { DataLayout::NHWC, DataLayout::NCHW });

/** In-place flags to test */
const auto in_place_values = framework::dataset::make("InPlace", { true, false });

/** With bias flags to test */
const auto with_bias_values = framework::dataset::make("WithBias", { true, false });

/** With gamma flags to test */
const auto with_gamma_values = framework::dataset::make("WithGamma", { true, false });

/** With beta flags to test */
const auto with_beta_values = framework::dataset::make("WithBeta", { true, false });

TEST_SUITE(CL)
TEST_SUITE(FuseBatchNormalization)
TEST_SUITE(Convolution)
TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLFuseBatchNormalizationConvFixture<float>, framework::DatasetMode::PRECOMMIT,
                                        combine(combine(combine(combine(combine(combine(
                                                        shape_conv_values_precommit,
                                                        framework::dataset::make("DataType", { DataType::F32 })),
                                                        data_layout_values),
                                                        in_place_values),
                                                        with_bias_values),
                                                        with_gamma_values),
                                                        with_beta_values))
{
    // Validate outputs
    validate(CLAccessor(_target_w), _reference_w, absolute_tolerance_f32);
    validate(CLAccessor(_target_b), _reference_b, absolute_tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLFuseBatchNormalizationConvFixture<float>, framework::DatasetMode::NIGHTLY,
                                        combine(combine(combine(combine(combine(combine(
                                                        shape_conv_values_nightly,
                                                        framework::dataset::make("DataType", { DataType::F32 })),
                                                        data_layout_values),
                                                        in_place_values),
                                                        with_bias_values),
                                                        with_gamma_values),
                                                        with_beta_values))
{
    // Validate outputs
    validate(CLAccessor(_target_w), _reference_w, absolute_tolerance_f32);
    validate(CLAccessor(_target_b), _reference_b, absolute_tolerance_f32);
}

TEST_SUITE_END() // FP32

TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLFuseBatchNormalizationConvFixture<half>, framework::DatasetMode::PRECOMMIT,
                                        combine(combine(combine(combine(combine(combine(
                                                        shape_conv_values_precommit,
                                                        framework::dataset::make("DataType", { DataType::F16 })),
                                                        data_layout_values),
                                                        in_place_values),
                                                        with_bias_values),
                                                        with_gamma_values),
                                                        with_beta_values))
{
    // Validate outputs
    validate(CLAccessor(_target_w), _reference_w, absolute_tolerance_f16);
    validate(CLAccessor(_target_b), _reference_b, absolute_tolerance_f16);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLFuseBatchNormalizationConvFixture<half>, framework::DatasetMode::NIGHTLY,
                                        combine(combine(combine(combine(combine(combine(
                                                        shape_conv_values_nightly,
                                                        framework::dataset::make("DataType", { DataType::F16 })),
                                                        data_layout_values),
                                                        in_place_values),
                                                        with_bias_values),
                                                        with_gamma_values),
                                                        with_beta_values))
{
    // Validate outputs
    validate(CLAccessor(_target_w), _reference_w, absolute_tolerance_f16);
    validate(CLAccessor(_target_b), _reference_b, absolute_tolerance_f16);
}

TEST_SUITE_END() // FP16
TEST_SUITE_END() // Float
TEST_SUITE_END() // Convolution

TEST_SUITE(DepthwiseConvolution)
TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLFuseBatchNormalizationDWCFixture<float>, framework::DatasetMode::PRECOMMIT,
                                        combine(combine(combine(combine(combine(combine(
                                                        datasets::Small3DShapes(),
                                                        framework::dataset::make("DataType", { DataType::F32 })),
                                                        data_layout_values),
                                                        in_place_values),
                                                        with_bias_values),
                                                        with_gamma_values),
                                                        with_beta_values))
{
    // Validate outputs
    validate(CLAccessor(_target_w), _reference_w, absolute_tolerance_f32);
    validate(CLAccessor(_target_b), _reference_b, absolute_tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLFuseBatchNormalizationDWCFixture<float>, framework::DatasetMode::NIGHTLY,
                                        combine(combine(combine(combine(combine(combine(
                                                        datasets::Large3DShapes(),
                                                        framework::dataset::make("DataType", { DataType::F32 })),
                                                        data_layout_values),
                                                        in_place_values),
                                                        with_bias_values),
                                                        with_gamma_values),
                                                        with_beta_values))
{
    // Validate outputs
    validate(CLAccessor(_target_w), _reference_w, absolute_tolerance_f32);
    validate(CLAccessor(_target_b), _reference_b, absolute_tolerance_f32);
}

TEST_SUITE_END() // FP32

TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLFuseBatchNormalizationDWCFixture<half>, framework::DatasetMode::PRECOMMIT,
                                        combine(combine(combine(combine(combine(combine(
                                                        datasets::Small3DShapes(),
                                                        framework::dataset::make("DataType", { DataType::F16 })),
                                                        data_layout_values),
                                                        in_place_values),
                                                        with_bias_values),
                                                        with_gamma_values),
                                                        with_beta_values))
{
    // Validate outputs
    validate(CLAccessor(_target_w), _reference_w, absolute_tolerance_f16);
    validate(CLAccessor(_target_b), _reference_b, absolute_tolerance_f16);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLFuseBatchNormalizationDWCFixture<half>, framework::DatasetMode::NIGHTLY,
                                        combine(combine(combine(combine(combine(combine(
                                                        datasets::Large3DShapes(),
                                                        framework::dataset::make("DataType", { DataType::F16 })),
                                                        data_layout_values),
                                                        in_place_values),
                                                        with_bias_values),
                                                        with_gamma_values),
                                                        with_beta_values))
{
    // Validate outputs
    validate(CLAccessor(_target_w), _reference_w, absolute_tolerance_f16);
    validate(CLAccessor(_target_b), _reference_b, absolute_tolerance_f16);
}

TEST_SUITE_END() // FP16
TEST_SUITE_END() // Float
TEST_SUITE_END() // DepthwiseConvolution

TEST_SUITE_END() // FuseBatchNormalization
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute