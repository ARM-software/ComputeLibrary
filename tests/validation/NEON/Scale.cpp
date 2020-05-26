/*
 * Copyright (c) 2017-2020 ARM Limited.
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
using test::datasets::ShapeDataset;

/** Class to generate boundary values for the given template parameters
 * including shapes with large differences between width and height
 */
template <uint32_t channel, uint32_t batch, uint32_t element_per_vector, uint32_t vector_size>
class ScaleShapesBaseDataSet : public ShapeDataset
{
    static constexpr auto boundary_minus_one = element_per_vector * vector_size - 1;
    static constexpr auto boundary_plus_one  = element_per_vector * vector_size + 1;
    static constexpr auto small_size         = 3;

public:
    // These tensor shapes are NCHW layout, fixture will convert to NHWC.
    ScaleShapesBaseDataSet()
        : ShapeDataset("Shape",
    {
        TensorShape{ small_size, boundary_minus_one, channel, batch },
                     TensorShape{ small_size, boundary_plus_one, channel, batch },
                     TensorShape{ boundary_minus_one, small_size, channel, batch },
                     TensorShape{ boundary_plus_one, small_size, channel, batch },
                     TensorShape{ boundary_minus_one, boundary_plus_one, channel, batch },
                     TensorShape{ boundary_plus_one, boundary_minus_one, channel, batch },
    })
    {
    }
};

/** For the single vector, only larger value (+1) than boundary
 * since smaller value (-1) could cause some invalid shapes like
 * - invalid zero size
 * - size 1 which isn't compatible with scale with aligned corners.
 */
template <uint32_t channel, uint32_t batch, uint32_t element_per_vector>
class ScaleShapesBaseDataSet<channel, batch, element_per_vector, 1> : public ShapeDataset
{
    static constexpr auto small_size        = 3;
    static constexpr auto boundary_plus_one = element_per_vector + 1;

public:
    // These tensor shapes are NCHW layout, fixture will convert to NHWC.
    ScaleShapesBaseDataSet()
        : ShapeDataset("Shape",
    {
        TensorShape{ small_size, boundary_plus_one, channel, batch },
                     TensorShape{ boundary_plus_one, small_size, channel, batch },
    })
    {
    }
};

/** For the shapes smaller than one vector, only pre-defined tiny shapes
 * are tested (3x2, 2x3) as smaller shapes are more likely to cause
 * issues and easier to debug.
 */
template <uint32_t channel, uint32_t batch, uint32_t element_per_vector>
class ScaleShapesBaseDataSet<channel, batch, element_per_vector, 0> : public ShapeDataset
{
    static constexpr auto small_size                 = 3;
    static constexpr auto zero_vector_boundary_value = 2;

public:
    // These tensor shapes are NCHW layout, fixture will convert to NHWC.
    ScaleShapesBaseDataSet()
        : ShapeDataset("Shape",
    {
        TensorShape{ small_size, zero_vector_boundary_value, channel, batch },
                     TensorShape{ zero_vector_boundary_value, small_size, channel, batch },
    })
    {
    }
};

/** Generated shaeps
 * - 2D shapes with 0, 1, 2 vector iterations
 * - 3D shapes with 0, 1 vector iterations
 * - 4D shapes with 0 vector iterations
 */
#define SCALE_SHAPE_DATASET(element_per_vector)                                                  \
    concat(concat(concat(concat(concat(ScaleShapesBaseDataSet<1, 1, (element_per_vector), 0>(),  \
                                       ScaleShapesBaseDataSet<1, 1, (element_per_vector), 1>()), \
                                ScaleShapesBaseDataSet<1, 1, (element_per_vector), 2>()),        \
                         ScaleShapesBaseDataSet<3, 3, (element_per_vector), 0>()),               \
                  ScaleShapesBaseDataSet<3, 3, (element_per_vector), 1>()),                      \
           ScaleShapesBaseDataSet<3, 7, (element_per_vector), 0>())

/** We consider vector size in byte 64 since the maximum size of
 * a vector used by @ref ScaleKernelInfo is currently 64-byte (float32x4x4).
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

/** Interpolation policy test set */
const auto InterpolationPolicySet = framework::dataset::make("InterpolationPolicy",
{
    InterpolationPolicy::NEAREST_NEIGHBOR,
    InterpolationPolicy::BILINEAR,
});

/** Scale data types */
const auto ScaleDataLayouts = framework::dataset::make("DataLayout",
{
    DataLayout::NCHW,
    DataLayout::NHWC,
});

/** Sampling policy data set */
const auto SamplingPolicySet = combine(datasets::SamplingPolicies(),
                                       framework::dataset::make("AlignCorners", { false }));

/** Sampling policy data set for Aligned Corners which only allows TOP_LEFT poicy.*/
const auto AlignCornersSamplingPolicySet = combine(framework::dataset::make("SamplingPolicy",
{
    SamplingPolicy::TOP_LEFT,
}),
framework::dataset::make("AlignCorners", { true }));

/** Generating dataset for non-quantized data tyeps with the given shapes */
#define ASSEMBLE_DATASET(shape, samping_policy_set)             \
    combine(combine(combine(combine((shape), ScaleDataLayouts), \
                            InterpolationPolicySet),            \
                    datasets::BorderModes()),                   \
            samping_policy_set)

/** Quantization information data set */
const auto QuantizationInfoSet = framework::dataset::make("QuantizationInfo",
{
    QuantizationInfo(0.5f, -10),
});

/** Generating dataset for quantized data tyeps with the given shapes */
#define ASSEMBLE_QUANTIZED_DATASET(shape, sampling_policy_set)    \
    combine(combine(combine(combine(combine(shape,                \
                                            QuantizationInfoSet), \
                                    ScaleDataLayouts),            \
                            InterpolationPolicySet),              \
                    datasets::BorderModes()),                     \
            sampling_policy_set)

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
constexpr auto default_border_mode          = BorderMode::UNDEFINED;
constexpr auto default_sampling_policy      = SamplingPolicy::CENTER;
constexpr bool default_use_padding          = false;

TEST_CASE(NullPtr, framework::DatasetMode::ALL)
{
    const auto input  = TensorInfo{ input_shape, 1, default_data_type, default_data_layout };
    const auto output = TensorInfo{ output_shape, 1, default_data_type, default_data_layout };
    Status     result{};

    // nullptr is given as input
    result = NEScale::validate(nullptr, &output, default_interpolation_policy, default_border_mode);
    ARM_COMPUTE_EXPECT(bool(result) == false, framework::LogLevel::ERRORS);

    result = NEScale::validate(nullptr, &output, ScaleKernelInfo{ default_interpolation_policy, default_border_mode });
    ARM_COMPUTE_EXPECT(bool(result) == false, framework::LogLevel::ERRORS);

    // nullptr is given as output
    result = NEScale::validate(&input, nullptr, default_interpolation_policy, default_border_mode);
    ARM_COMPUTE_EXPECT(bool(result) == false, framework::LogLevel::ERRORS);

    result = NEScale::validate(&input, nullptr, ScaleKernelInfo{ default_interpolation_policy, default_border_mode });
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

        result = NEScale::validate(&input, &output, default_interpolation_policy, default_border_mode);
        ARM_COMPUTE_EXPECT(bool(result) == kv.second, framework::LogLevel::ERRORS);

        result = NEScale::validate(&input, &output, ScaleKernelInfo{ default_interpolation_policy, default_border_mode });
        ARM_COMPUTE_EXPECT(bool(result) == kv.second, framework::LogLevel::ERRORS);
    }
}

TEST_CASE(MissmatchingDataType, framework::DatasetMode::ALL)
{
    constexpr auto non_default_data_type = DataType::F32;

    const auto input  = TensorInfo{ input_shape, 1, default_data_type, default_data_layout };
    const auto output = TensorInfo{ output_shape, 1, non_default_data_type, default_data_layout };
    Status     result{};

    result = NEScale::validate(&input, &output, default_interpolation_policy, default_border_mode);
    ARM_COMPUTE_EXPECT(bool(result) == false, framework::LogLevel::ERRORS);

    result = NEScale::validate(&input, &output, ScaleKernelInfo{ default_interpolation_policy, default_border_mode });
    ARM_COMPUTE_EXPECT(bool(result) == false, framework::LogLevel::ERRORS);
}

TEST_CASE(UsePadding, framework::DatasetMode::ALL)
{
    const auto input  = TensorInfo{ input_shape, 1, default_data_type, default_data_layout };
    const auto output = TensorInfo{ output_shape, 1, default_data_type, default_data_layout };
    Status     result{};

    // When use padding is false, border mode should be constant
    constexpr auto border_mode = BorderMode::UNDEFINED;
    constexpr bool use_padding = false;

    result = NEScale::validate(&input, &output, default_interpolation_policy, border_mode, PixelValue(), default_sampling_policy, use_padding);
    ARM_COMPUTE_EXPECT(bool(result) == false, framework::LogLevel::ERRORS);

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

    result = NEScale::validate(&input, &output, interpolation_policy, default_border_mode);
    ARM_COMPUTE_EXPECT(bool(result) == false, framework::LogLevel::ERRORS);

    result = NEScale::validate(&input, &output, ScaleKernelInfo{ interpolation_policy, default_border_mode });
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

    result = NEScale::validate(&input, &output, interpolation_policy, default_border_mode);
    ARM_COMPUTE_EXPECT(bool(result) == false, framework::LogLevel::ERRORS);

    result = NEScale::validate(&input, &output, ScaleKernelInfo{ interpolation_policy, default_border_mode });
    ARM_COMPUTE_EXPECT(bool(result) == false, framework::LogLevel::ERRORS);
}

TEST_CASE(InvalidAlignedCornerOutput, framework::DatasetMode::ALL)
{
    // Bilinear with aligned corners require at least 2x2 output to prevent overflow.
    // Also, aligned corners require sampling policy to be TOP_LEFT.
    constexpr auto interpolation_policy = InterpolationPolicy::BILINEAR;
    constexpr bool align_corners        = true;
    constexpr auto sampling_policy      = SamplingPolicy::TOP_LEFT;
    const auto     invalid_output_shape = TensorShape{ 1, 1, 3, 2 };

    const auto input  = TensorInfo{ input_shape, 1, default_data_type, default_data_layout };
    const auto output = TensorInfo{ invalid_output_shape, 1, default_data_type, default_data_layout };
    Status     result{};

    result = NEScale::validate(&input, &output, interpolation_policy, default_border_mode, PixelValue(), sampling_policy, default_use_padding, align_corners);
    ARM_COMPUTE_EXPECT(bool(result) == false, framework::LogLevel::ERRORS);

    result = NEScale::validate(&input, &output, ScaleKernelInfo{ interpolation_policy, default_border_mode, PixelValue(), sampling_policy, default_use_padding, align_corners });
    ARM_COMPUTE_EXPECT(bool(result) == false, framework::LogLevel::ERRORS);
}
TEST_SUITE_END() // Validate

template <typename T>
using NEScaleFixture = ScaleValidationFixture<Tensor, Accessor, NEScale, T>;
template <typename T>
using NEScaleQuantizedFixture = ScaleValidationQuantizedFixture<Tensor, Accessor, NEScale, T>;

TEST_SUITE(Float)
TEST_SUITE(FP32)
const auto f32_shape = combine((SCALE_SHAPE_DATASET(num_elements_per_vector<float>())), framework::dataset::make("DataType", DataType::F32));
FIXTURE_DATA_TEST_CASE(RunSmall, NEScaleFixture<float>, framework::DatasetMode::ALL, ASSEMBLE_DATASET(f32_shape, SamplingPolicySet))
{
    //Create valid region
    TensorInfo  src_info(_shape, 1, _data_type);
    ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _policy, _sampling_policy, (_border_mode == BorderMode::UNDEFINED));

    // Validate output
    validate(Accessor(_target), _reference, valid_region, tolerance_f32, tolerance_num_f32);
}
FIXTURE_DATA_TEST_CASE(RunSmallAlignCorners, NEScaleFixture<float>, framework::DatasetMode::ALL, ASSEMBLE_DATASET(f32_shape, AlignCornersSamplingPolicySet))
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
FIXTURE_DATA_TEST_CASE(RunSmall, NEScaleFixture<half>, framework::DatasetMode::ALL, ASSEMBLE_DATASET(f16_shape, SamplingPolicySet))
{
    //Create valid region
    TensorInfo        src_info(_shape, 1, _data_type);
    const ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _policy, _sampling_policy, (_border_mode == BorderMode::UNDEFINED));

    // Validate output
    validate(Accessor(_target), _reference, valid_region, tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunSmallAlignCorners, NEScaleFixture<half>, framework::DatasetMode::ALL, ASSEMBLE_DATASET(f16_shape, AlignCornersSamplingPolicySet))
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
FIXTURE_DATA_TEST_CASE(RunSmall, NEScaleFixture<uint8_t>, framework::DatasetMode::ALL, ASSEMBLE_DATASET(u8_shape, SamplingPolicySet))
{
    //Create valid region
    TensorInfo  src_info(_shape, 1, _data_type);
    ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _policy, _sampling_policy, (_border_mode == BorderMode::UNDEFINED));

    // Validate output
    validate(Accessor(_target), _reference, valid_region, tolerance_u8);
}
FIXTURE_DATA_TEST_CASE(RunSmallAlignCorners, NEScaleFixture<uint8_t>, framework::DatasetMode::ALL, ASSEMBLE_DATASET(u8_shape, AlignCornersSamplingPolicySet))
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
FIXTURE_DATA_TEST_CASE(RunSmall, NEScaleFixture<int16_t>, framework::DatasetMode::ALL, ASSEMBLE_DATASET(s16_shape, SamplingPolicySet))
{
    //Create valid region
    TensorInfo  src_info(_shape, 1, _data_type);
    ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _policy, _sampling_policy, (_border_mode == BorderMode::UNDEFINED));

    // Validate output
    validate(Accessor(_target), _reference, valid_region, tolerance_s16, tolerance_num_s16);
}
FIXTURE_DATA_TEST_CASE(RunSmallAlignCorners, NEScaleFixture<int16_t>, framework::DatasetMode::ALL, ASSEMBLE_DATASET(s16_shape, AlignCornersSamplingPolicySet))
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
FIXTURE_DATA_TEST_CASE(RunSmall, NEScaleQuantizedFixture<uint8_t>, framework::DatasetMode::ALL, ASSEMBLE_QUANTIZED_DATASET(qasymm8_shape, SamplingPolicySet))
{
    //Create valid region
    TensorInfo  src_info(_shape, 1, _data_type);
    ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _policy, _sampling_policy, (_border_mode == BorderMode::UNDEFINED));

    // Validate output
    validate(Accessor(_target), _reference, valid_region, tolerance_u8);
}
FIXTURE_DATA_TEST_CASE(RunSmallAlignCorners, NEScaleQuantizedFixture<uint8_t>, framework::DatasetMode::ALL, ASSEMBLE_QUANTIZED_DATASET(qasymm8_shape, AlignCornersSamplingPolicySet))
{
    //Create valid region
    TensorInfo  src_info(_shape, 1, _data_type);
    ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _policy, _sampling_policy, (_border_mode == BorderMode::UNDEFINED));

    // Validate output
    validate(Accessor(_target), _reference, valid_region, tolerance_u8);
}
TEST_SUITE_END() // QASYMM8
TEST_SUITE(QASYMM8_SIGNED)
const auto qasymm8_signed_shape = combine((SCALE_SHAPE_DATASET(num_elements_per_vector<int8_t>())), framework::dataset::make("DataType", DataType::QASYMM8_SIGNED));
FIXTURE_DATA_TEST_CASE(RunSmall, NEScaleQuantizedFixture<int8_t>, framework::DatasetMode::ALL, ASSEMBLE_QUANTIZED_DATASET(qasymm8_signed_shape, SamplingPolicySet))
{
    //Create valid region
    TensorInfo  src_info(_shape, 1, _data_type);
    ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _policy, _sampling_policy, (_border_mode == BorderMode::UNDEFINED));

    // Validate output
    validate(Accessor(_target), _reference, valid_region, tolerance_u8);
}
FIXTURE_DATA_TEST_CASE(RunSmallAlignCorners, NEScaleQuantizedFixture<int8_t>, framework::DatasetMode::ALL, ASSEMBLE_QUANTIZED_DATASET(qasymm8_signed_shape, AlignCornersSamplingPolicySet))
{
    //Create valid region
    TensorInfo  src_info(_shape, 1, _data_type);
    ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _policy, _sampling_policy, (_border_mode == BorderMode::UNDEFINED));

    // Validate output
    validate(Accessor(_target), _reference, valid_region, tolerance_u8);
}
TEST_SUITE_END() // QASYMM8_SIGNED
TEST_SUITE_END() // Quantized

TEST_SUITE_END() // Scale
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
