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
#include "framework/Asserts.h"
#include "framework/Macros.h"
#include "framework/datasets/Datasets.h"
#include "tests/CL/CLAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets_new/BorderModeDataset.h"
#include "tests/datasets_new/ShapeDatasets.h"
#include "tests/validation_new/Helpers.h"
#include "tests/validation_new/Validation.h"
#include "tests/validation_new/fixtures/ScaleFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
/** Tolerance */
constexpr AbsoluteTolerance<uint8_t> tolerance(1);
} // namespace

TEST_SUITE(CL)
TEST_SUITE(Scale)

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(combine(concat(datasets::SmallShapes(), datasets::LargeShapes()), framework::dataset::make("DataType", DataType::U8)),
                                                                           framework::dataset::make("InterpolationPolicy", { InterpolationPolicy::NEAREST_NEIGHBOR, InterpolationPolicy::BILINEAR })),
                                                                   datasets::BorderModes()),
               shape, data_type, policy, border_mode)
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
    clscale.configure(&src, &dst, policy, border_mode, constant_border_value);

    // Validate valid region
    const ValidRegion dst_valid_region = calculate_valid_region_scale(*(src.info()), shape_scaled, policy, BorderSize(1), (border_mode == BorderMode::UNDEFINED));

    validate(dst.info()->valid_region(), dst_valid_region);

    // Validate padding
    PaddingCalculator calculator(shape_scaled.x(), 4);
    calculator.set_border_mode(border_mode);

    const PaddingSize read_padding(1);
    const PaddingSize write_padding = calculator.required_padding(PaddingCalculator::Option::EXCLUDE_BORDER);
    validate(src.info()->padding(), read_padding);
    validate(dst.info()->padding(), write_padding);
}

template <typename T>
using CLScaleFixture = ScaleValidationFixture<CLTensor, CLAccessor, CLScale, T>;

FIXTURE_DATA_TEST_CASE(RunSmall, CLScaleFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), framework::dataset::make("DataType",
                                                                                                                     DataType::U8)),
                                                                                                             framework::dataset::make("InterpolationPolicy",
{ InterpolationPolicy::NEAREST_NEIGHBOR, InterpolationPolicy::BILINEAR })),
datasets::BorderModes()))
{
    //Create valid region
    TensorInfo        src_info(_shape, 1, _data_type);
    const ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _policy, BorderSize(1), (_border_mode == BorderMode::UNDEFINED));

    // Validate output
    validate(CLAccessor(_target), _reference, valid_region, tolerance);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLScaleFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), framework::dataset::make("DataType",
                                                                                                                   DataType::U8)),
                                                                                                           framework::dataset::make("InterpolationPolicy",
{ InterpolationPolicy::NEAREST_NEIGHBOR, InterpolationPolicy::BILINEAR })),
datasets::BorderModes()))
{
    //Create valid region
    TensorInfo        src_info(_shape, 1, _data_type);
    const ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _policy, BorderSize(1), (_border_mode == BorderMode::UNDEFINED));

    // Validate output
    validate(CLAccessor(_target), _reference, valid_region, tolerance);
}
TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
