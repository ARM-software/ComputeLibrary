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
#include "arm_compute/runtime/NEON/functions/NEScale.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/BorderModeDataset.h"
#include "tests/datasets/InterpolationPolicyDataset.h"
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
/** Scale data types */
const auto ScaleDataTypes = framework::dataset::make("DataType",
{
    DataType::U8,
    DataType::S16,
    DataType::F32,
});

/** Scale data types */
const auto ScaleDataLayouts = framework::dataset::make("DataLayout",
{
    DataLayout::NCHW,
    DataLayout::NHWC,
});

/** Tolerance */
constexpr AbsoluteTolerance<uint8_t> tolerance_u8(1);
constexpr AbsoluteTolerance<int16_t> tolerance_s16(1);
RelativeTolerance<float>             tolerance_f32(0.01);
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
RelativeTolerance<half> tolerance_f16(half(0.1));
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

constexpr float tolerance_num_s16 = 0.01f;
constexpr float tolerance_num_f32 = 0.01f;
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(Scale)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(zip(zip(
        framework::dataset::make("InputInfo", { TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::U8),  // Mismatching data type
                                                TensorInfo(TensorShape(4U, 27U, 13U), 1, DataType::F32), // Invalid policy
                                                TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32), // Insufficient padding
                                                TensorInfo(TensorShape(4U, 27U, 13U), 1, DataType::F32),
                                              }),
        framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(132U, 25U, 2U), 1, DataType::F32),
                                                TensorInfo(TensorShape(4U, 132U, 25U), 1, DataType::F32),
                                                TensorInfo(TensorShape(132U, 25U, 2U), 1, DataType::F32),
                                                TensorInfo(TensorShape(4U, 132U, 25U), 1, DataType::F32),
                                              })),
        framework::dataset::make("InterpolationPolicy", { InterpolationPolicy::NEAREST_NEIGHBOR,
                                                          InterpolationPolicy::AREA,
                                                          InterpolationPolicy::AREA,
                                                          InterpolationPolicy::NEAREST_NEIGHBOR,
                                                        })),
        framework::dataset::make("BorderMode",  { BorderMode::UNDEFINED,
                                                  BorderMode::UNDEFINED,
                                                  BorderMode::UNDEFINED,
                                                  BorderMode::REPLICATE,
                                                })),
        framework::dataset::make("SamplingPolicy",  { SamplingPolicy::CENTER,
                                                      SamplingPolicy::CENTER,
                                                      SamplingPolicy::CENTER,
                                                      SamplingPolicy::CENTER,
                                                    })),
        framework::dataset::make("DataLayout",  { DataLayout::NCHW,
                                                  DataLayout::NHWC,
                                                  DataLayout::NCHW,
                                                  DataLayout::NHWC,
                                                })),
        framework::dataset::make("Expected", { false, false, false ,true })),
        input_info, output_info, policy,border_mode, sampling_policy, data_layout, expected)
{
    const PixelValue constant_border(5);
    Status status = NEScale::validate(&input_info.clone()->set_is_resizable(false).set_data_layout(data_layout),
                                           &output_info.clone()->set_is_resizable(false).set_data_layout(data_layout),
                                           policy, border_mode, constant_border, sampling_policy);
    ARM_COMPUTE_EXPECT(bool(status) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(combine(combine(combine(datasets::SmallShapes(), ScaleDataTypes), ScaleDataLayouts),
                                                                                   framework::dataset::make("InterpolationPolicy", { InterpolationPolicy::NEAREST_NEIGHBOR, InterpolationPolicy::BILINEAR })),
                                                                           datasets::BorderModes()),
                                                                   framework::dataset::make("SamplingPolicy", { SamplingPolicy::CENTER })),
               shape, data_type, data_layout, policy, border_mode, sampling_policy)
{
    std::mt19937                          generator(library->seed());
    std::uniform_real_distribution<float> distribution_float(0.25, 2);
    const float                           scale_x               = distribution_float(generator);
    const float                           scale_y               = distribution_float(generator);
    uint8_t                               constant_border_value = 0;
    TensorShape                           src_shape             = shape;
    if(border_mode == BorderMode::CONSTANT)
    {
        std::uniform_int_distribution<uint8_t> distribution_u8(0, 255);
        constant_border_value = distribution_u8(generator);
    }

    // Get width/height indices depending on layout
    const int idx_width  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int idx_height = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);

    // Change shape in case of NHWC.
    if(data_layout == DataLayout::NHWC)
    {
        permute(src_shape, PermutationVector(2U, 0U, 1U));
    }

    // Calculate scaled shape
    TensorShape shape_scaled(src_shape);
    shape_scaled.set(idx_width, src_shape[idx_width] * scale_x);
    shape_scaled.set(idx_height, src_shape[idx_height] * scale_y);

    // Create tensors
    Tensor src = create_tensor<Tensor>(src_shape, data_type, 1, QuantizationInfo(), data_layout);
    Tensor dst = create_tensor<Tensor>(shape_scaled, data_type, 1, QuantizationInfo(), data_layout);

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    NEScale nescale;
    nescale.configure(&src, &dst, policy, border_mode, constant_border_value, sampling_policy);

    // Validate valid region
    const ValidRegion dst_valid_region = calculate_valid_region_scale(*(src.info()), shape_scaled, policy, sampling_policy, (border_mode == BorderMode::UNDEFINED));
    validate(dst.info()->valid_region(), dst_valid_region);

    // Validate padding
    int num_elements_processed_x = 16;
    if(data_layout == DataLayout::NHWC)
    {
        num_elements_processed_x = (policy == InterpolationPolicy::BILINEAR) ? 1 : 16 / src.info()->element_size();
    }
    PaddingCalculator calculator(shape_scaled.x(), num_elements_processed_x);
    calculator.set_border_mode(border_mode);

    PaddingSize read_padding(1);
    if(data_layout == DataLayout::NHWC)
    {
        read_padding = calculator.required_padding(PaddingCalculator::Option::EXCLUDE_BORDER);
        if(border_mode != BorderMode::REPLICATE && policy == InterpolationPolicy::BILINEAR)
        {
            read_padding.top = 1;
        }
    }
    const PaddingSize write_padding = calculator.required_padding(PaddingCalculator::Option::EXCLUDE_BORDER);
    validate(src.info()->padding(), read_padding);
    validate(dst.info()->padding(), write_padding);
}

template <typename T>
using NEScaleFixture = ScaleValidationFixture<Tensor, Accessor, NEScale, T>;
template <typename T>
using NEScaleQuantizedFixture = ScaleValidationQuantizedFixture<Tensor, Accessor, NEScale, T>;

TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEScaleFixture<float>, framework::DatasetMode::ALL, combine(combine(combine(combine(combine(datasets::SmallShapes(), framework::dataset::make("DataType",
                                                                                                                     DataType::F32)),
                                                                                                                     framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                             framework::dataset::make("InterpolationPolicy", { InterpolationPolicy::NEAREST_NEIGHBOR, InterpolationPolicy::BILINEAR })),
                                                                                                     datasets::BorderModes()),
                                                                                             framework::dataset::make("SamplingPolicy", { SamplingPolicy::TOP_LEFT, SamplingPolicy::CENTER })))
{
    //Create valid region
    TensorInfo  src_info(_shape, 1, _data_type);
    ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _policy, _sampling_policy, (_border_mode == BorderMode::UNDEFINED));

    // Validate output
    validate(Accessor(_target), _reference, valid_region, tolerance_f32, tolerance_num_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEScaleFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(combine(datasets::LargeShapes(), framework::dataset::make("DataType",
                                                                                                                 DataType::F32)),
                                                                                                                 framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                                 framework::dataset::make("InterpolationPolicy", { InterpolationPolicy::NEAREST_NEIGHBOR, InterpolationPolicy::BILINEAR })),
                                                                                                         datasets::BorderModes()),
                                                                                                 framework::dataset::make("SamplingPolicy", { SamplingPolicy::TOP_LEFT, SamplingPolicy::CENTER })))
{
    //Create valid region
    TensorInfo  src_info(_shape, 1, _data_type);
    ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _policy, _sampling_policy, (_border_mode == BorderMode::UNDEFINED));

    // Validate output
    validate(Accessor(_target), _reference, valid_region, tolerance_f32, tolerance_num_f32);
}
TEST_SUITE_END() // FP32
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEScaleFixture<half>, framework::DatasetMode::ALL, combine(combine(combine(combine(combine(datasets::SmallShapes(), framework::dataset::make("DataType",
                                                                                                                    DataType::F16)),
                                                                                                                    framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                            framework::dataset::make("InterpolationPolicy", { InterpolationPolicy::NEAREST_NEIGHBOR, InterpolationPolicy::BILINEAR })),
                                                                                                    datasets::BorderModes()),
                                                                                            framework::dataset::make("SamplingPolicy", { SamplingPolicy::TOP_LEFT, SamplingPolicy::CENTER })))
{
    //Create valid region
    TensorInfo        src_info(_shape, 1, _data_type);
    const ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _policy, _sampling_policy, (_border_mode == BorderMode::UNDEFINED));

    // Validate output
    validate(Accessor(_target), _reference, valid_region, tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEScaleFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(combine(datasets::LargeShapes(), framework::dataset::make("DataType",
                                                                                                                        DataType::F16)),
                                                                                                                        framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                                framework::dataset::make("InterpolationPolicy", { InterpolationPolicy::NEAREST_NEIGHBOR, InterpolationPolicy::BILINEAR })),
                                                                                                        datasets::BorderModes()),
                                                                                                framework::dataset::make("SamplingPolicy", { SamplingPolicy::TOP_LEFT, SamplingPolicy::CENTER })))
{
    //Create valid region
    TensorInfo        src_info(_shape, 1, _data_type);
    const ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _policy, _sampling_policy, (_border_mode == BorderMode::UNDEFINED));

    // Validate output
    validate(Accessor(_target), _reference, valid_region, tolerance_f16);
}
TEST_SUITE_END() // FP16
#endif           /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
TEST_SUITE_END() // Float

TEST_SUITE(Integer)
TEST_SUITE(U8)
FIXTURE_DATA_TEST_CASE(RunSmall, NEScaleFixture<uint8_t>, framework::DatasetMode::ALL, combine(combine(combine(combine(combine(datasets::SmallShapes(), framework::dataset::make("DataType",
                                                                                                                       DataType::U8)),
                                                                                                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                               framework::dataset::make("InterpolationPolicy", { InterpolationPolicy::NEAREST_NEIGHBOR, InterpolationPolicy::BILINEAR })),
                                                                                                       datasets::BorderModes()),
                                                                                               framework::dataset::make("SamplingPolicy", { SamplingPolicy::TOP_LEFT, SamplingPolicy::CENTER })))
{
    //Create valid region
    TensorInfo  src_info(_shape, 1, _data_type);
    ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _policy, _sampling_policy, (_border_mode == BorderMode::UNDEFINED));

    // Validate output
    validate(Accessor(_target), _reference, valid_region, tolerance_u8);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEScaleFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(combine(datasets::LargeShapes(), framework::dataset::make("DataType",
                                                                                                                   DataType::U8)),
                                                                                                                   framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                                   framework::dataset::make("InterpolationPolicy", { InterpolationPolicy::NEAREST_NEIGHBOR, InterpolationPolicy::BILINEAR })),
                                                                                                           datasets::BorderModes()),
                                                                                                   framework::dataset::make("SamplingPolicy", { SamplingPolicy::TOP_LEFT, SamplingPolicy::CENTER })))
{
    //Create valid region
    TensorInfo  src_info(_shape, 1, _data_type);
    ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _policy, _sampling_policy, (_border_mode == BorderMode::UNDEFINED));

    // Validate output
    validate(Accessor(_target), _reference, valid_region, tolerance_u8);
}
TEST_SUITE_END() // U8
TEST_SUITE(S16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEScaleFixture<int16_t>, framework::DatasetMode::ALL, combine(combine(combine(combine(combine(datasets::SmallShapes(), framework::dataset::make("DataType",
                                                                                                                       DataType::S16)),
                                                                                                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                               framework::dataset::make("InterpolationPolicy", { InterpolationPolicy::NEAREST_NEIGHBOR, InterpolationPolicy::BILINEAR })),
                                                                                                       datasets::BorderModes()),
                                                                                               framework::dataset::make("SamplingPolicy", { SamplingPolicy::TOP_LEFT, SamplingPolicy::CENTER })))
{
    //Create valid region
    TensorInfo  src_info(_shape, 1, _data_type);
    ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _policy, _sampling_policy, (_border_mode == BorderMode::UNDEFINED));

    // Validate output
    validate(Accessor(_target), _reference, valid_region, tolerance_s16, tolerance_num_s16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEScaleFixture<int16_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(combine(datasets::LargeShapes(), framework::dataset::make("DataType",
                                                                                                                   DataType::S16)),
                                                                                                                   framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                                   framework::dataset::make("InterpolationPolicy", { InterpolationPolicy::NEAREST_NEIGHBOR, InterpolationPolicy::BILINEAR })),
                                                                                                           datasets::BorderModes()),
                                                                                                   framework::dataset::make("SamplingPolicy", { SamplingPolicy::TOP_LEFT, SamplingPolicy::CENTER })))
{
    //Create valid region
    TensorInfo  src_info(_shape, 1, _data_type);
    ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _policy, _sampling_policy, (_border_mode == BorderMode::UNDEFINED));

    // Validate output
    validate(Accessor(_target), _reference, valid_region, tolerance_s16, tolerance_num_s16);
}
TEST_SUITE_END() // S16
TEST_SUITE_END() // Integer

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, NEScaleQuantizedFixture<uint8_t>, framework::DatasetMode::ALL, combine(combine(combine(combine(combine(combine(datasets::SmallShapes(),
                                                                                                                        framework::dataset::make("DataType", DataType::QASYMM8)),
                                                                                                                        framework::dataset::make("QuantizationInfo", { QuantizationInfo(0.5f, -10) })),
                                                                                                                        framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                                        framework::dataset::make("InterpolationPolicy", { InterpolationPolicy::NEAREST_NEIGHBOR, InterpolationPolicy::BILINEAR })),
                                                                                                                datasets::BorderModes()),
                                                                                                        framework::dataset::make("SamplingPolicy", { SamplingPolicy::TOP_LEFT, SamplingPolicy::CENTER })))
{
    //Create valid region
    TensorInfo  src_info(_shape, 1, _data_type);
    ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _policy, _sampling_policy, (_border_mode == BorderMode::UNDEFINED));

    // Validate output
    validate(Accessor(_target), _reference, valid_region, tolerance_u8);
}
TEST_SUITE_END() // QASYMM8
TEST_SUITE_END() // Quantized

TEST_SUITE_END() // Scale
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
