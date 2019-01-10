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
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/functions/CLScale.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/CL/CLAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/BorderModeDataset.h"
#include "tests/datasets/SamplingPolicyDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/ScaleFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
/** CNN data types */
const auto ScaleDataTypes = framework::dataset::make("DataType",
{
    DataType::U8,
    DataType::S16,
    DataType::F16,
    DataType::F32,
});

/** Tolerance */
constexpr AbsoluteTolerance<uint8_t> tolerance_u8(1);
constexpr AbsoluteTolerance<int16_t> tolerance_s16(1);
constexpr float                      tolerance_f32_absolute(0.001f);

RelativeTolerance<float> tolerance_f32(0.05);
RelativeTolerance<half>  tolerance_f16(half(0.1));

constexpr float tolerance_num_f32(0.01f);
} // namespace

TEST_SUITE(CL)
TEST_SUITE(Scale)

// *INDENT-OFF*
// clang-format off

DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(
                framework::dataset::make("InputInfo",{ TensorInfo(TensorShape(28U, 32U, 2U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(28U, 32U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(36U, 36U, 2U, 4U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(40U, 35U, 2U, 4U), 1, DataType::S16),
                                                       TensorInfo(TensorShape(37U, 37U, 2U), 1, DataType::F32),          // window shrink
                                                       TensorInfo(TensorShape(37U, 37U, 3U, 4U), 1, DataType::F32),      // mismatching datatype
                                                       TensorInfo(TensorShape(28U, 33U, 2U), 1, DataType::F32),          // policy area, scale factor not correct
                                                    }),
                framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(32U, 68U, 2U), 1, DataType::F16),
                                                        TensorInfo(TensorShape(40U, 56U, 2U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(40U, 76U, 2U, 4U), 1, DataType::U8),
                                                        TensorInfo(TensorShape(28U, 32U, 2U, 4U), 1, DataType::S16),
                                                        TensorInfo(TensorShape(39U, 55U, 2U), 1, DataType::F32),           // window shrink
                                                        TensorInfo(TensorShape(39U, 77U, 3U, 4U), 1, DataType::F16),       // mismatching datatype
                                                        TensorInfo(TensorShape(26U, 21U, 2U), 1, DataType::F32),           // policy area, scale factor not correct
                })),
                framework::dataset::make("Policy",{ InterpolationPolicy::BILINEAR,
                                                    InterpolationPolicy::BILINEAR,
                                                    InterpolationPolicy::NEAREST_NEIGHBOR,
                                                    InterpolationPolicy::NEAREST_NEIGHBOR,
                                                    InterpolationPolicy::NEAREST_NEIGHBOR,
                                                    InterpolationPolicy::BILINEAR,
                                                    InterpolationPolicy::AREA,
                })),
                framework::dataset::make("BorderMode",{ BorderMode::UNDEFINED,
                                                        BorderMode::UNDEFINED,
                                                        BorderMode::UNDEFINED,
                                                        BorderMode::UNDEFINED,
                                                        BorderMode::UNDEFINED,
                                                        BorderMode::UNDEFINED,
                                                        BorderMode::UNDEFINED,
                })),
                framework::dataset::make("Expected", { true, true, true, true, false, false, false })),
input_info, output_info, policy, border_mode, expected)
{
    Status status = CLScale::validate(&input_info.clone()->set_is_resizable(false),
                                      &output_info.clone()->set_is_resizable(false), policy, border_mode);
    ARM_COMPUTE_EXPECT(bool(status) == expected, framework::LogLevel::ERRORS);
}

// clang-format on
// *INDENT-ON*

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(combine(combine(datasets::MediumShapes(), ScaleDataTypes),
                                                                                   framework::dataset::make("InterpolationPolicy", { InterpolationPolicy::NEAREST_NEIGHBOR, InterpolationPolicy::BILINEAR })),
                                                                           datasets::BorderModes()),
                                                                   datasets::SamplingPolicies()),
               shape, data_type, policy, border_mode, sampling_policy)
{
    std::mt19937                           generator(library->seed());
    std::uniform_real_distribution<float>  distribution_float(0.25, 2);
    const float                            scale_x = distribution_float(generator);
    const float                            scale_y = distribution_float(generator);
    std::uniform_int_distribution<uint8_t> distribution_u8(0, 255);
    uint8_t                                constant_border_value = distribution_u8(generator);

    // Create tensors
    CLTensor    src = create_tensor<CLTensor>(shape, data_type);
    TensorShape shape_scaled(shape);
    shape_scaled.set(0, shape[0] * scale_x);
    shape_scaled.set(1, shape[1] * scale_y);
    CLTensor dst = create_tensor<CLTensor>(shape_scaled, data_type);

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    CLScale clscale;
    clscale.configure(&src, &dst, policy, border_mode, constant_border_value, sampling_policy);

    // Get border size depending on border mode
    const BorderSize border_size(border_mode == BorderMode::UNDEFINED ? 0 : 1);

    // Validate valid region
    const ValidRegion dst_valid_region = calculate_valid_region_scale(*(src.info()), shape_scaled, policy, sampling_policy, (border_mode == BorderMode::UNDEFINED));
    validate(dst.info()->valid_region(), dst_valid_region);

    // Validate padding
    PaddingCalculator calculator(shape_scaled.x(), 4);
    calculator.set_border_mode(border_mode);

    const PaddingSize read_padding(border_size);
    const PaddingSize write_padding = calculator.required_padding(PaddingCalculator::Option::EXCLUDE_BORDER);
    validate(src.info()->padding(), read_padding);
    validate(dst.info()->padding(), write_padding);
}

template <typename T>
using CLScaleFixture = ScaleValidationFixture<CLTensor, CLAccessor, CLScale, T>;

TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLScaleFixture<float>, framework::DatasetMode::ALL, combine(combine(combine(combine(combine(datasets::Tiny4DShapes(), framework::dataset::make("DataType",
                                                                                                                     DataType::F32)),
                                                                                                                     framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                             framework::dataset::make("InterpolationPolicy", { InterpolationPolicy::NEAREST_NEIGHBOR, InterpolationPolicy::BILINEAR })),
                                                                                                     datasets::BorderModes()),
                                                                                             datasets::SamplingPolicies()))
{
    //Create valid region
    TensorInfo        src_info(_shape, 1, _data_type);
    const ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _policy, _sampling_policy, (_border_mode == BorderMode::UNDEFINED));

    // Validate output
    validate(CLAccessor(_target), _reference, valid_region, tolerance_f32, tolerance_num_f32, tolerance_f32_absolute);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLScaleFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(combine(datasets::LargeShapes(), framework::dataset::make("DataType",
                                                                                                                 DataType::F32)),
                                                                                                                 framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                                 framework::dataset::make("InterpolationPolicy", { InterpolationPolicy::NEAREST_NEIGHBOR, InterpolationPolicy::BILINEAR })),
                                                                                                         datasets::BorderModes()),
                                                                                                 datasets::SamplingPolicies()))
{
    //Create valid region
    TensorInfo        src_info(_shape, 1, _data_type);
    const ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _policy, _sampling_policy, (_border_mode == BorderMode::UNDEFINED));

    // Validate output
    validate(CLAccessor(_target), _reference, valid_region, tolerance_f32, tolerance_num_f32, tolerance_f32_absolute);
}
TEST_SUITE_END() // FP32
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLScaleFixture<half>, framework::DatasetMode::ALL, combine(combine(combine(combine(combine(datasets::Tiny4DShapes(), framework::dataset::make("DataType",
                                                                                                                    DataType::F16)),
                                                                                                                    framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                            framework::dataset::make("InterpolationPolicy", { InterpolationPolicy::NEAREST_NEIGHBOR, InterpolationPolicy::BILINEAR })),
                                                                                                    datasets::BorderModes()),
                                                                                            datasets::SamplingPolicies()))
{
    //Create valid region
    TensorInfo        src_info(_shape, 1, _data_type);
    const ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _policy, _sampling_policy, (_border_mode == BorderMode::UNDEFINED));

    // Validate output
    validate(CLAccessor(_target), _reference, valid_region, tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLScaleFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(combine(datasets::LargeShapes(), framework::dataset::make("DataType",
                                                                                                                        DataType::F16)),
                                                                                                                        framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                                framework::dataset::make("InterpolationPolicy", { InterpolationPolicy::NEAREST_NEIGHBOR, InterpolationPolicy::BILINEAR })),
                                                                                                        datasets::BorderModes()),
                                                                                                datasets::SamplingPolicies()))
{
    //Create valid region
    TensorInfo        src_info(_shape, 1, _data_type);
    const ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _policy, _sampling_policy, (_border_mode == BorderMode::UNDEFINED));

    // Validate output
    validate(CLAccessor(_target), _reference, valid_region, tolerance_f16);
}
TEST_SUITE_END() // FP16
TEST_SUITE_END() // Float

TEST_SUITE(Integer)
TEST_SUITE(U8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLScaleFixture<uint8_t>, framework::DatasetMode::ALL, combine(combine(combine(combine(combine(datasets::Tiny4DShapes(), framework::dataset::make("DataType",
                                                                                                                       DataType::U8)),
                                                                                                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                               framework::dataset::make("InterpolationPolicy", { InterpolationPolicy::NEAREST_NEIGHBOR, InterpolationPolicy::BILINEAR })),
                                                                                                       datasets::BorderModes()),
                                                                                               datasets::SamplingPolicies()))
{
    //Create valid region
    TensorInfo        src_info(_shape, 1, _data_type);
    const ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _policy, _sampling_policy, (_border_mode == BorderMode::UNDEFINED));

    // Validate output
    validate(CLAccessor(_target), _reference, valid_region, tolerance_u8);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLScaleFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(combine(datasets::LargeShapes(), framework::dataset::make("DataType",
                                                                                                                   DataType::U8)),
                                                                                                                   framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                                   framework::dataset::make("InterpolationPolicy", { InterpolationPolicy::NEAREST_NEIGHBOR, InterpolationPolicy::BILINEAR })),
                                                                                                           datasets::BorderModes()),
                                                                                                   datasets::SamplingPolicies()))
{
    //Create valid region
    TensorInfo        src_info(_shape, 1, _data_type);
    const ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _policy, _sampling_policy, (_border_mode == BorderMode::UNDEFINED));

    // Validate output
    validate(CLAccessor(_target), _reference, valid_region, tolerance_u8);
}
TEST_SUITE_END() // U8
TEST_SUITE(S16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLScaleFixture<int16_t>, framework::DatasetMode::ALL, combine(combine(combine(combine(combine(datasets::Tiny4DShapes(), framework::dataset::make("DataType",
                                                                                                                       DataType::S16)),
                                                                                                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                               framework::dataset::make("InterpolationPolicy", { InterpolationPolicy::NEAREST_NEIGHBOR, InterpolationPolicy::BILINEAR })),
                                                                                                       datasets::BorderModes()),
                                                                                               datasets::SamplingPolicies()))
{
    //Create valid region
    TensorInfo        src_info(_shape, 1, _data_type);
    const ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _policy, _sampling_policy, (_border_mode == BorderMode::UNDEFINED));

    // Validate output
    validate(CLAccessor(_target), _reference, valid_region, tolerance_s16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLScaleFixture<int16_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(combine(datasets::LargeShapes(), framework::dataset::make("DataType",
                                                                                                                   DataType::S16)),
                                                                                                                   framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                                   framework::dataset::make("InterpolationPolicy", { InterpolationPolicy::NEAREST_NEIGHBOR, InterpolationPolicy::BILINEAR })),
                                                                                                           datasets::BorderModes()),
                                                                                                   datasets::SamplingPolicies()))
{
    //Create valid region
    TensorInfo        src_info(_shape, 1, _data_type);
    const ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _policy, _sampling_policy, (_border_mode == BorderMode::UNDEFINED));

    // Validate output
    validate(CLAccessor(_target), _reference, valid_region, tolerance_s16);
}
TEST_SUITE_END() // S16
TEST_SUITE_END() // Integer

template <typename T>
using CLScaleQuantizedFixture = ScaleValidationQuantizedFixture<CLTensor, CLAccessor, CLScale, T>;
TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLScaleQuantizedFixture<uint8_t>, framework::DatasetMode::ALL, combine(combine(combine(combine(combine(combine(datasets::Tiny4DShapes(),
                                                                                                                        framework::dataset::make("DataType",
                                                                                                                                DataType::QASYMM8)),
                                                                                                                        framework::dataset::make("QuantizationInfo", { QuantizationInfo(0.5f, -1) })),
                                                                                                                        framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                                        framework::dataset::make("InterpolationPolicy", { InterpolationPolicy::NEAREST_NEIGHBOR, InterpolationPolicy::BILINEAR })),
                                                                                                                datasets::BorderModes()),
                                                                                                        datasets::SamplingPolicies()))
{
    //Create valid region
    TensorInfo        src_info(_shape, 1, _data_type);
    const ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _policy, _sampling_policy, (_border_mode == BorderMode::UNDEFINED));

    // Validate output
    validate(CLAccessor(_target), _reference, valid_region, tolerance_u8);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLScaleQuantizedFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(combine(combine(datasets::LargeShapes(),
                                                                                                                    framework::dataset::make("DataType",
                                                                                                                            DataType::QASYMM8)),
                                                                                                                    framework::dataset::make("QuantizationInfo", { QuantizationInfo(0.5f, -1) })),
                                                                                                                    framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                                    framework::dataset::make("InterpolationPolicy", { InterpolationPolicy::NEAREST_NEIGHBOR, InterpolationPolicy::BILINEAR })),
                                                                                                                    datasets::BorderModes()),
                                                                                                            datasets::SamplingPolicies()))
{
    //Create valid region
    TensorInfo        src_info(_shape, 1, _data_type);
    const ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _policy, _sampling_policy, (_border_mode == BorderMode::UNDEFINED));

    // Validate output
    validate(CLAccessor(_target), _reference, valid_region, tolerance_u8);
}
TEST_SUITE_END() // QASYMM8
TEST_SUITE_END() // Quantized

TEST_SUITE_END() // Scale
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
