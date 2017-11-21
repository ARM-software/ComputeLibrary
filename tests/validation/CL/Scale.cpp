/*
 * Copyright (c) 2017 ARM Limited.
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
RelativeTolerance<float>             tolerance_f32(0.05);
RelativeTolerance<half>              tolerance_f16(half(0.1));

constexpr float tolerance_num_f32(0.01f);
} // namespace

TEST_SUITE(CL)
TEST_SUITE(Scale)

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(combine(combine(concat(datasets::MediumShapes(), datasets::LargeShapes()), ScaleDataTypes),
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
    const ValidRegion dst_valid_region = calculate_valid_region_scale(*(src.info()), shape_scaled, policy, border_size, (border_mode == BorderMode::UNDEFINED));
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
FIXTURE_DATA_TEST_CASE(RunSmall, CLScaleFixture<float>, framework::DatasetMode::ALL, combine(combine(combine(combine(datasets::SmallShapes(), framework::dataset::make("DataType", DataType::F32)),
                                                                                                             framework::dataset::make("InterpolationPolicy", { InterpolationPolicy::NEAREST_NEIGHBOR, InterpolationPolicy::BILINEAR })),
                                                                                                     datasets::BorderModes()),
                                                                                             datasets::SamplingPolicies()))
{
    //Create valid region
    TensorInfo        src_info(_shape, 1, _data_type);
    const ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _policy, BorderSize(1), (_border_mode == BorderMode::UNDEFINED));

    // Validate output
    validate(CLAccessor(_target), _reference, valid_region, tolerance_f32, tolerance_num_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLScaleFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::LargeShapes(), framework::dataset::make("DataType", DataType::F32)),
                                                                                                                 framework::dataset::make("InterpolationPolicy", { InterpolationPolicy::NEAREST_NEIGHBOR, InterpolationPolicy::BILINEAR })),
                                                                                                         datasets::BorderModes()),
                                                                                                 datasets::SamplingPolicies()))
{
    //Create valid region
    TensorInfo        src_info(_shape, 1, _data_type);
    const ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _policy, BorderSize(1), (_border_mode == BorderMode::UNDEFINED));

    // Validate output
    validate(CLAccessor(_target), _reference, valid_region, tolerance_f32, tolerance_num_f32);
}
TEST_SUITE_END()
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLScaleFixture<half>, framework::DatasetMode::ALL, combine(combine(combine(combine(datasets::SmallShapes(), framework::dataset::make("DataType", DataType::F16)),
                                                                                                            framework::dataset::make("InterpolationPolicy", { InterpolationPolicy::NEAREST_NEIGHBOR, InterpolationPolicy::BILINEAR })),
                                                                                                    datasets::BorderModes()),
                                                                                            datasets::SamplingPolicies()))
{
    //Create valid region
    TensorInfo        src_info(_shape, 1, _data_type);
    const ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _policy, BorderSize(1), (_border_mode == BorderMode::UNDEFINED));

    // Validate output
    validate(CLAccessor(_target), _reference, valid_region, tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLScaleFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::LargeShapes(), framework::dataset::make("DataType",
                                                                                                                        DataType::F16)),
                                                                                                                framework::dataset::make("InterpolationPolicy", { InterpolationPolicy::NEAREST_NEIGHBOR, InterpolationPolicy::BILINEAR })),
                                                                                                        datasets::BorderModes()),
                                                                                                datasets::SamplingPolicies()))
{
    //Create valid region
    TensorInfo        src_info(_shape, 1, _data_type);
    const ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _policy, BorderSize(1), (_border_mode == BorderMode::UNDEFINED));

    // Validate output
    validate(CLAccessor(_target), _reference, valid_region, tolerance_f16);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE(Integer)
TEST_SUITE(U8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLScaleFixture<uint8_t>, framework::DatasetMode::ALL, combine(combine(combine(combine(datasets::SmallShapes(), framework::dataset::make("DataType", DataType::U8)),
                                                                                                               framework::dataset::make("InterpolationPolicy", { InterpolationPolicy::NEAREST_NEIGHBOR, InterpolationPolicy::BILINEAR })),
                                                                                                       datasets::BorderModes()),
                                                                                               datasets::SamplingPolicies()))
{
    //Create valid region
    TensorInfo        src_info(_shape, 1, _data_type);
    const ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _policy, BorderSize(1), (_border_mode == BorderMode::UNDEFINED));

    // Validate output
    validate(CLAccessor(_target), _reference, valid_region, tolerance_u8);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLScaleFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::LargeShapes(), framework::dataset::make("DataType", DataType::U8)),
                                                                                                                   framework::dataset::make("InterpolationPolicy", { InterpolationPolicy::NEAREST_NEIGHBOR, InterpolationPolicy::BILINEAR })),
                                                                                                           datasets::BorderModes()),
                                                                                                   datasets::SamplingPolicies()))
{
    //Create valid region
    TensorInfo        src_info(_shape, 1, _data_type);
    const ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _policy, BorderSize(1), (_border_mode == BorderMode::UNDEFINED));

    // Validate output
    validate(CLAccessor(_target), _reference, valid_region, tolerance_u8);
}
TEST_SUITE_END()
TEST_SUITE(S16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLScaleFixture<int16_t>, framework::DatasetMode::ALL, combine(combine(combine(combine(datasets::SmallShapes(), framework::dataset::make("DataType", DataType::S16)),
                                                                                                               framework::dataset::make("InterpolationPolicy", { InterpolationPolicy::NEAREST_NEIGHBOR, InterpolationPolicy::BILINEAR })),
                                                                                                       datasets::BorderModes()),
                                                                                               datasets::SamplingPolicies()))
{
    //Create valid region
    TensorInfo        src_info(_shape, 1, _data_type);
    const ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _policy, BorderSize(1), (_border_mode == BorderMode::UNDEFINED));

    // Validate output
    validate(CLAccessor(_target), _reference, valid_region, tolerance_s16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLScaleFixture<int16_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::LargeShapes(), framework::dataset::make("DataType",
                                                                                                                   DataType::S16)),
                                                                                                                   framework::dataset::make("InterpolationPolicy", { InterpolationPolicy::NEAREST_NEIGHBOR, InterpolationPolicy::BILINEAR })),
                                                                                                           datasets::BorderModes()),
                                                                                                   datasets::SamplingPolicies()))
{
    //Create valid region
    TensorInfo        src_info(_shape, 1, _data_type);
    const ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _policy, BorderSize(1), (_border_mode == BorderMode::UNDEFINED));

    // Validate output
    validate(CLAccessor(_target), _reference, valid_region, tolerance_s16);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
