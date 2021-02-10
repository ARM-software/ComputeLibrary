/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "tests/datasets/ScaleValidationDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
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
using datasets::ScaleShapesBaseDataSet;
using datasets::ScaleInterpolationPolicySet;
using datasets::ScaleDataLayouts;
using datasets::ScaleSamplingPolicySet;
using datasets::ScaleAlignCornersSamplingPolicySet;

/** We consider vector size in byte 64 since the maximum size of
 * a vector used by @ref NEScaleKernel is currently 64-byte (float32x4x4).
 * There are possibility to reduce test time further by using
 * smaller vector sizes for different data types where applicable.
 */
constexpr uint32_t vector_byte = 64;

template <typename T>
constexpr uint32_t num_elements_per_vector()
{
    return vector_byte / sizeof(T);
}

/** Scale data types */
const auto ScaleDataTypes = framework::dataset::make("DataType",
{
    DataType::U8,
    DataType::S16,
    DataType::F32,
});

/** Quantization information data set */
const auto QuantizationInfoSet = framework::dataset::make("QuantizationInfo",
{
    QuantizationInfo(0.5f, -10),
});

/** Tolerance */
constexpr AbsoluteTolerance<uint8_t> tolerance_u8(1);
constexpr AbsoluteTolerance<int16_t> tolerance_s16(1);
RelativeTolerance<float>             tolerance_f32(0.05);
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
RelativeTolerance<half> tolerance_f16(half(0.1));
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

constexpr float tolerance_num_s16 = 0.01f;
constexpr float tolerance_num_f32 = 0.01f;
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(Scale)
TEST_SUITE(Validate)

/** Validate test suite is to test ARM_COMPUTE_RETURN_ON_* macros
 * we use to check the validity of given arguments in @ref NEScale
 * and subsequent call to @ref NEScaleKernel.
 * Since this is using validate() of @ref NEScale, which pre-adjust
 * arguments for @ref NEScaleKernel, the following conditions in
 * the kernel are not currently tested.
 * - The same input and output
 * - Data type of offset, dx and dy
 * This suite also tests two different validate() APIs - one is
 * using @ref ScaleKernelInfo and the other one is more verbose
 * one calls the other one - in the same test case. Even though
 * there are possibility that it makes debugging for regression
 * harder, belows are reasons of this test case implementation.
 * - The more verbose one is just a wrapper function calls
 *   the other one without any additional logic. So we are
 *   safe to merge two tests into one.
 * - A large amount of code duplication is test suite can be prevented.
 */

const auto input_shape  = TensorShape{ 2, 3, 3, 2 };
const auto output_shape = TensorShape{ 4, 6, 3, 2 };

constexpr auto default_data_type            = DataType::U8;
constexpr auto default_data_layout          = DataLayout::NHWC;
constexpr auto default_interpolation_policy = InterpolationPolicy::NEAREST_NEIGHBOR;
constexpr auto default_border_mode          = BorderMode::CONSTANT;
constexpr auto default_sampling_policy      = SamplingPolicy::CENTER;

TEST_CASE(NullPtr, framework::DatasetMode::ALL)
{
    const auto input  = TensorInfo{ input_shape, 1, default_data_type, default_data_layout };
    const auto output = TensorInfo{ output_shape, 1, default_data_type, default_data_layout };
    Status     result{};

    // nullptr is given as input
    result = NEScale::validate(nullptr, &output, ScaleKernelInfo{ default_interpolation_policy, default_border_mode, PixelValue(), SamplingPolicy::CENTER, false });
    ARM_COMPUTE_EXPECT(bool(result) == false, framework::LogLevel::ERRORS);

    // nullptr is given as output
    result = NEScale::validate(&input, nullptr, ScaleKernelInfo{ default_interpolation_policy, default_border_mode, PixelValue(), SamplingPolicy::CENTER, false });
    ARM_COMPUTE_EXPECT(bool(result) == false, framework::LogLevel::ERRORS);
}

TEST_CASE(SupportDataType, framework::DatasetMode::ALL)
{
    const std::map<DataType, bool> supported_data_types =
    {
        { DataType::U8, true },
        { DataType::S8, false },
        { DataType::QSYMM8, false },
        { DataType::QASYMM8, true },
        { DataType::QASYMM8_SIGNED, true },
        { DataType::QSYMM8_PER_CHANNEL, false },
        { DataType::U16, false },
        { DataType::S16, true },
        { DataType::QSYMM16, false },
        { DataType::QASYMM16, false },
        { DataType::U32, false },
        { DataType::S32, false },
        { DataType::U64, false },
        { DataType::S64, false },
        { DataType::BFLOAT16, false },
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        { DataType::F16, true },
#else  // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        { DataType::F16, false },
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        { DataType::F32, true },
        { DataType::F64, false },
        { DataType::SIZET, false },
    };
    Status result{};
    for(auto &kv : supported_data_types)
    {
        const auto input  = TensorInfo{ input_shape, 1, kv.first, default_data_layout };
        const auto output = TensorInfo{ output_shape, 1, kv.first, default_data_layout };

        result = NEScale::validate(&input, &output, ScaleKernelInfo{ default_interpolation_policy, default_border_mode, PixelValue(), SamplingPolicy::CENTER, false });
        ARM_COMPUTE_EXPECT(bool(result) == kv.second, framework::LogLevel::ERRORS);
    }
}

TEST_CASE(MissmatchingDataType, framework::DatasetMode::ALL)
{
    constexpr auto non_default_data_type = DataType::F32;

    const auto input  = TensorInfo{ input_shape, 1, default_data_type, default_data_layout };
    const auto output = TensorInfo{ output_shape, 1, non_default_data_type, default_data_layout };
    Status     result{};

    result = NEScale::validate(&input, &output, ScaleKernelInfo{ default_interpolation_policy, default_border_mode, PixelValue(), SamplingPolicy::CENTER, false });
    ARM_COMPUTE_EXPECT(bool(result) == false, framework::LogLevel::ERRORS);
}

TEST_CASE(UsePadding, framework::DatasetMode::ALL)
{
    const auto input  = TensorInfo{ input_shape, 1, default_data_type, default_data_layout };
    const auto output = TensorInfo{ output_shape, 1, default_data_type, default_data_layout };
    Status     result{};

    // Padding is not supported anymore
    constexpr auto border_mode = BorderMode::CONSTANT;
    constexpr bool use_padding = true;

    result = NEScale::validate(&input, &output, ScaleKernelInfo{ default_interpolation_policy, border_mode, PixelValue(), default_sampling_policy, use_padding });
    ARM_COMPUTE_EXPECT(bool(result) == false, framework::LogLevel::ERRORS);
}

TEST_CASE(AreaWithNHWC, framework::DatasetMode::ALL)
{
    // InterpolationPolicy::AREA is not supported for NHWC
    constexpr auto interpolation_policy = InterpolationPolicy::AREA;
    constexpr auto data_layout          = DataLayout::NHWC;

    const auto input  = TensorInfo{ input_shape, 1, default_data_type, data_layout };
    const auto output = TensorInfo{ output_shape, 1, default_data_type, data_layout };
    Status     result{};

    result = NEScale::validate(&input, &output, ScaleKernelInfo{ interpolation_policy, default_border_mode, PixelValue(), SamplingPolicy::CENTER, false });
    ARM_COMPUTE_EXPECT(bool(result) == false, framework::LogLevel::ERRORS);
}

TEST_CASE(AreaWithNonU8, framework::DatasetMode::ALL)
{
    // InterpolationPolicy::AREA only supports U8
    constexpr auto interpolation_policy = InterpolationPolicy::AREA;
    constexpr auto data_type            = DataType::F32;
    constexpr auto data_layout          = DataLayout::NCHW;

    const auto input  = TensorInfo{ input_shape, 1, data_type, data_layout };
    const auto output = TensorInfo{ output_shape, 1, data_type, data_layout };
    Status     result{};

    result = NEScale::validate(&input, &output, ScaleKernelInfo{ interpolation_policy, default_border_mode, PixelValue(), SamplingPolicy::CENTER, false });
    ARM_COMPUTE_EXPECT(bool(result) == false, framework::LogLevel::ERRORS);
}

TEST_CASE(AlignedCornerNotSupported, framework::DatasetMode::ALL)
{
    // Aligned corners require sampling policy to be TOP_LEFT.
    constexpr auto interpolation_policy = InterpolationPolicy::BILINEAR;
    constexpr bool align_corners        = true;
    constexpr auto sampling_policy      = SamplingPolicy::CENTER;

    const auto input  = TensorInfo{ input_shape, 1, default_data_type, default_data_layout };
    const auto output = TensorInfo{ output_shape, 1, default_data_type, default_data_layout };
    Status     result{};

    result = NEScale::validate(&input, &output, ScaleKernelInfo{ interpolation_policy, default_border_mode, PixelValue(), sampling_policy, false, align_corners });
    ARM_COMPUTE_EXPECT(bool(result) == false, framework::LogLevel::ERRORS);
}
TEST_SUITE_END() // Validate

DATA_TEST_CASE(CheckNoPadding, framework::DatasetMode::ALL, combine(combine(combine(combine(datasets::Medium4DShapes(),
                                                                                            framework::dataset::make("DataType", { DataType::F32, DataType::QASYMM8 })),
                                                                                    framework::dataset::make("InterpolationPolicy", { InterpolationPolicy::BILINEAR, InterpolationPolicy::NEAREST_NEIGHBOR })),
                                                                            framework::dataset::make("SamplingPolicy", { SamplingPolicy::CENTER, SamplingPolicy::TOP_LEFT })),
                                                                    framework::dataset::make("DataLayout", { DataLayout::NHWC, DataLayout::NCHW })),
               shape, data_type, interpolation_policy, sampling_policy, data_layout)
{
    constexpr auto  default_border_mode = BorderMode::CONSTANT;
    ScaleKernelInfo info(interpolation_policy, default_border_mode, PixelValue(), sampling_policy, false);

    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, data_type);
    src.info()->set_data_layout(data_layout);

    const float scale_x = 0.5f;
    const float scale_y = 0.5f;
    TensorShape shape_scaled(shape);
    const int   idx_width  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int   idx_height = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    shape_scaled.set(idx_width, shape[idx_width] * scale_x, /* apply_dim_correction = */ false);
    shape_scaled.set(idx_height, shape[idx_height] * scale_y, /* apply_dim_correction = */ false);
    Tensor dst = create_tensor<Tensor>(shape_scaled, data_type);

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    NEScale scale;
    scale.configure(&src, &dst, info);

    validate(src.info()->padding(), PaddingSize(0, 0, 0, 0));
    validate(dst.info()->padding(), PaddingSize(0, 0, 0, 0));
}

DATA_TEST_CASE(CheckNoPaddingInterpAREA, framework::DatasetMode::ALL, combine(combine(combine(combine(datasets::Medium4DShapes(),
                                                                                                      framework::dataset::make("DataType", { DataType::U8 })),
                                                                                              framework::dataset::make("InterpolationPolicy", { InterpolationPolicy::AREA })),
                                                                                      framework::dataset::make("SamplingPolicy", { SamplingPolicy::CENTER, SamplingPolicy::TOP_LEFT })),
                                                                              framework::dataset::make("DataLayout", { DataLayout::NCHW })),
               shape, data_type, interpolation_policy, sampling_policy, data_layout)
{
    constexpr auto  default_border_mode = BorderMode::CONSTANT;
    ScaleKernelInfo info(interpolation_policy, default_border_mode, PixelValue(), sampling_policy, false);

    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, data_type);
    src.info()->set_data_layout(data_layout);

    const float scale_x = 0.5f;
    const float scale_y = 0.5f;
    TensorShape shape_scaled(shape);
    const int   idx_width  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int   idx_height = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    shape_scaled.set(idx_width, shape[idx_width] * scale_x, /* apply_dim_correction = */ false);
    shape_scaled.set(idx_height, shape[idx_height] * scale_y, /* apply_dim_correction = */ false);

    Tensor dst = create_tensor<Tensor>(shape, data_type);

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    NEScale scale;
    scale.configure(&src, &dst, info);

    validate(src.info()->padding(), PaddingSize(0, 0, 0, 0));
    validate(dst.info()->padding(), PaddingSize(0, 0, 0, 0));
}

template <typename T>
using NEScaleFixture = ScaleValidationFixture<Tensor, Accessor, NEScale, T>;
template <typename T>
using NEScaleQuantizedFixture = ScaleValidationQuantizedFixture<Tensor, Accessor, NEScale, T>;

TEST_SUITE(Float)
TEST_SUITE(FP32)
const auto f32_shape = combine((SCALE_SHAPE_DATASET(num_elements_per_vector<float>())), framework::dataset::make("DataType", DataType::F32));
FIXTURE_DATA_TEST_CASE(RunSmall, NEScaleFixture<float>, framework::DatasetMode::ALL, ASSEMBLE_DATASET(f32_shape, ScaleSamplingPolicySet))
{
    //Create valid region
    TensorInfo  src_info(_shape, 1, _data_type);
    ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _policy, _sampling_policy, (_border_mode == BorderMode::UNDEFINED));

    // Validate output
    validate(Accessor(_target), _reference, valid_region, tolerance_f32, tolerance_num_f32);
}
FIXTURE_DATA_TEST_CASE(RunSmallAlignCorners, NEScaleFixture<float>, framework::DatasetMode::ALL, ASSEMBLE_DATASET(f32_shape, ScaleAlignCornersSamplingPolicySet))
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
const auto f16_shape = combine((SCALE_SHAPE_DATASET(num_elements_per_vector<half>())), framework::dataset::make("DataType", DataType::F16));
FIXTURE_DATA_TEST_CASE(RunSmall, NEScaleFixture<half>, framework::DatasetMode::ALL, ASSEMBLE_DATASET(f16_shape, ScaleSamplingPolicySet))
{
    //Create valid region
    TensorInfo        src_info(_shape, 1, _data_type);
    const ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _policy, _sampling_policy, (_border_mode == BorderMode::UNDEFINED));

    // Validate output
    validate(Accessor(_target), _reference, valid_region, tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunSmallAlignCorners, NEScaleFixture<half>, framework::DatasetMode::ALL, ASSEMBLE_DATASET(f16_shape, ScaleAlignCornersSamplingPolicySet))
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
const auto u8_shape = combine((SCALE_SHAPE_DATASET(num_elements_per_vector<uint8_t>())), framework::dataset::make("DataType", DataType::U8));
FIXTURE_DATA_TEST_CASE(RunSmall, NEScaleFixture<uint8_t>, framework::DatasetMode::ALL, ASSEMBLE_DATASET(u8_shape, ScaleSamplingPolicySet))
{
    //Create valid region
    TensorInfo  src_info(_shape, 1, _data_type);
    ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _policy, _sampling_policy, (_border_mode == BorderMode::UNDEFINED));

    // Validate output
    validate(Accessor(_target), _reference, valid_region, tolerance_u8);
}
FIXTURE_DATA_TEST_CASE(RunSmallAlignCorners, NEScaleFixture<uint8_t>, framework::DatasetMode::ALL, ASSEMBLE_DATASET(u8_shape, ScaleAlignCornersSamplingPolicySet))
{
    //Create valid region
    TensorInfo  src_info(_shape, 1, _data_type);
    ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _policy, _sampling_policy, (_border_mode == BorderMode::UNDEFINED));

    // Validate output
    validate(Accessor(_target), _reference, valid_region, tolerance_u8);
}
TEST_SUITE_END() // U8
TEST_SUITE(S16)
const auto s16_shape = combine((SCALE_SHAPE_DATASET(num_elements_per_vector<int16_t>())), framework::dataset::make("DataType", DataType::S16));
FIXTURE_DATA_TEST_CASE(RunSmall, NEScaleFixture<int16_t>, framework::DatasetMode::ALL, ASSEMBLE_DATASET(s16_shape, ScaleSamplingPolicySet))
{
    //Create valid region
    TensorInfo  src_info(_shape, 1, _data_type);
    ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _policy, _sampling_policy, (_border_mode == BorderMode::UNDEFINED));

    // Validate output
    validate(Accessor(_target), _reference, valid_region, tolerance_s16, tolerance_num_s16);
}
FIXTURE_DATA_TEST_CASE(RunSmallAlignCorners, NEScaleFixture<int16_t>, framework::DatasetMode::ALL, ASSEMBLE_DATASET(s16_shape, ScaleAlignCornersSamplingPolicySet))
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
const auto qasymm8_shape = combine((SCALE_SHAPE_DATASET(num_elements_per_vector<uint8_t>())), framework::dataset::make("DataType", DataType::QASYMM8));
FIXTURE_DATA_TEST_CASE(RunSmall, NEScaleQuantizedFixture<uint8_t>, framework::DatasetMode::ALL, ASSEMBLE_QUANTIZED_DATASET(qasymm8_shape, ScaleSamplingPolicySet, QuantizationInfoSet))
{
    //Create valid region
    TensorInfo  src_info(_shape, 1, _data_type);
    ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _policy, _sampling_policy, (_border_mode == BorderMode::UNDEFINED));

    // Validate output
    validate(Accessor(_target), _reference, valid_region, tolerance_u8);
}
FIXTURE_DATA_TEST_CASE(RunSmallAlignCorners, NEScaleQuantizedFixture<uint8_t>, framework::DatasetMode::ALL, ASSEMBLE_QUANTIZED_DATASET(qasymm8_shape, ScaleAlignCornersSamplingPolicySet,
                       QuantizationInfoSet))
{
    //Create valid region
    TensorInfo  src_info(_shape, 1, _data_type);
    ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _policy, _sampling_policy, (_border_mode == BorderMode::UNDEFINED));

    // Validate output
    validate(Accessor(_target), _reference, valid_region, tolerance_u8);
}
TEST_SUITE_END() // QASYMM8
TEST_SUITE(QASYMM8_SIGNED)
const auto                          qasymm8_signed_shape = combine((SCALE_SHAPE_DATASET(num_elements_per_vector<int8_t>())), framework::dataset::make("DataType", DataType::QASYMM8_SIGNED));
constexpr AbsoluteTolerance<int8_t> tolerance_qasymm8_signed{ 1 };
FIXTURE_DATA_TEST_CASE(RunSmall, NEScaleQuantizedFixture<int8_t>, framework::DatasetMode::ALL, ASSEMBLE_QUANTIZED_DATASET(qasymm8_signed_shape, ScaleSamplingPolicySet, QuantizationInfoSet))
{
    //Create valid region
    TensorInfo  src_info(_shape, 1, _data_type);
    ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _policy, _sampling_policy, (_border_mode == BorderMode::UNDEFINED));

    // Validate output
    validate(Accessor(_target), _reference, valid_region, tolerance_qasymm8_signed);
}
FIXTURE_DATA_TEST_CASE(RunSmallAlignCorners, NEScaleQuantizedFixture<int8_t>, framework::DatasetMode::ALL, ASSEMBLE_QUANTIZED_DATASET(qasymm8_signed_shape, ScaleAlignCornersSamplingPolicySet,
                       QuantizationInfoSet))
{
    //Create valid region
    TensorInfo  src_info(_shape, 1, _data_type);
    ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _policy, _sampling_policy, (_border_mode == BorderMode::UNDEFINED));

    // Validate output
    validate(Accessor(_target), _reference, valid_region, tolerance_qasymm8_signed);
}
TEST_SUITE_END() // QASYMM8_SIGNED
TEST_SUITE_END() // Quantized

TEST_SUITE_END() // Scale
TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
