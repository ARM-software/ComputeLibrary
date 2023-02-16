/*
* Copyright (c) 2022-2023 Arm Limited.
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

#include "arm_compute/dynamic_fusion/sketch/gpu/operators/GpuResize.h"

#include "tests/CL/CLAccessor.h"
#include "tests/datasets/ScaleValidationDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/framework/Macros.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/dynamic_fusion/operators/ResizeFixture.h"

using namespace arm_compute::experimental::dynamic_fusion;
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
using datasets::ScaleSamplingPolicySet;
using datasets::ScaleAlignCornersSamplingPolicySet;

/** We consider vector size in byte 16 since the maximum size of
 * a vector used by @ref CLScaleKernel is currently 16-byte (float4).
 */
constexpr uint32_t vector_byte = 16;

template <typename T>
constexpr uint32_t num_elements_per_vector()
{
    return vector_byte / sizeof(T);
}

/** Quantization information data set */
const auto QuantizationInfoSet = framework::dataset::make("QuantizationInfo",
{
    QuantizationInfo(0.5f, -1),
});

/** Tolerance */
constexpr AbsoluteTolerance<uint8_t> tolerance_q8(1);
constexpr AbsoluteTolerance<int8_t>  tolerance_qs8(1);
constexpr AbsoluteTolerance<int16_t> tolerance_s16(1);
constexpr float                      tolerance_f32_absolute(0.001f);

RelativeTolerance<float> tolerance_f32(0.05);
constexpr float          abs_tolerance_f16(0.1f);
RelativeTolerance<half>  tolerance_f16(half(0.1));

constexpr float tolerance_num_f32(0.01f);

} // namespace

TEST_SUITE(CL)
TEST_SUITE(DYNAMIC_FUSION)
TEST_SUITE(RESIZE)

TEST_SUITE(Validate)

const auto default_input_shape  = TensorShape{ 2, 3, 3, 2 };
const auto default_output_shape = TensorShape{ 4, 6, 3, 2 };

constexpr auto default_data_type   = DataType::U8;
constexpr auto default_data_layout = DataLayout::NHWC;

TEST_CASE(NullPtr, framework::DatasetMode::ALL)
{
    const TensorInfo input_info  = TensorInfo{ default_input_shape, 1, default_data_type, default_data_layout };
    const TensorInfo output_info = TensorInfo{ default_output_shape, 1, default_data_type, default_data_layout };

    CLCompileContext   cl_compile_ctx = CLKernelLibrary::get().get_compile_context();
    GpuWorkloadContext gpu_ctx        = GpuWorkloadContext{ &cl_compile_ctx };
    GpuWorkloadSketch  sketch{ &gpu_ctx };

    const TensorInfo sketch_input_info = sketch.create_tensor_info(input_info);

    // nullptr is given as input
    Status status = GpuResize::validate_op(sketch, nullptr, ResizeAttributes());
    ARM_COMPUTE_EXPECT(bool(status) == false, framework::LogLevel::ERRORS);
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
        { DataType::F16, true },
        { DataType::F32, true },
        { DataType::F64, false },
        { DataType::SIZET, false },
    };

    for(auto &kv : supported_data_types)
    {
        const TensorInfo input_info = TensorInfo{ default_input_shape, 1, kv.first, default_data_layout };

        CLCompileContext   cl_compile_ctx = CLKernelLibrary::get().get_compile_context();
        GpuWorkloadContext gpu_ctx        = GpuWorkloadContext{ &cl_compile_ctx };
        GpuWorkloadSketch  sketch{ &gpu_ctx };

        const TensorInfo sketch_input_info = sketch.create_tensor_info(input_info);

        ResizeAttributes attributes;
        attributes.output_width(default_output_shape[0]); // shape is not important unless it's empty
        attributes.output_height(default_output_shape[1]);

        Status status = GpuResize::validate_op(sketch, &sketch_input_info, attributes);
        ARM_COMPUTE_EXPECT(bool(status) == kv.second, framework::LogLevel::ERRORS);
    }
}

TEST_CASE(MismatchingDataType, framework::DatasetMode::ALL)
{
    constexpr DataType non_default_data_type = DataType::F32;

    const TensorInfo input_info  = TensorInfo{ default_input_shape, 1, default_data_type, default_data_layout };
    const TensorInfo output_info = TensorInfo{ default_output_shape, 1, non_default_data_type, default_data_layout };

    CLCompileContext   cl_compile_ctx = CLKernelLibrary::get().get_compile_context();
    GpuWorkloadContext gpu_ctx        = GpuWorkloadContext{ &cl_compile_ctx };
    GpuWorkloadSketch  sketch{ &gpu_ctx };

    const TensorInfo sketch_input_info = sketch.create_tensor_info(input_info);

    Status status = GpuResize::validate_op(sketch, &sketch_input_info, ResizeAttributes());
    ARM_COMPUTE_EXPECT(bool(status) == false, framework::LogLevel::ERRORS);
}

TEST_CASE(AlignedCornerNotSupported, framework::DatasetMode::ALL)
{
    // Aligned corners require sampling policy to be TOP_LEFT.
    constexpr InterpolationPolicy interpolation_policy = InterpolationPolicy::BILINEAR;
    constexpr bool                align_corners        = true;
    constexpr SamplingPolicy      sampling_policy      = SamplingPolicy::CENTER;

    const TensorInfo input_info  = TensorInfo{ default_input_shape, 1, default_data_type, default_data_layout };
    const TensorInfo output_info = TensorInfo{ default_output_shape, 1, default_data_type, default_data_layout };

    CLCompileContext   cl_compile_ctx = CLKernelLibrary::get().get_compile_context();
    GpuWorkloadContext gpu_ctx        = GpuWorkloadContext{ &cl_compile_ctx };
    GpuWorkloadSketch  sketch{ &gpu_ctx };

    const TensorInfo sketch_input_info = sketch.create_tensor_info(input_info);

    ResizeAttributes attributes{};
    attributes.interpolation_policy(interpolation_policy)
    .sampling_policy(sampling_policy)
    .align_corners(align_corners);

    Status status = GpuResize::validate_op(sketch, &sketch_input_info, attributes);
    ARM_COMPUTE_EXPECT(bool(status) == false, framework::LogLevel::ERRORS);
}

TEST_CASE(UnsupportedInterpolationPolicy, framework::DatasetMode::ALL)
{
    const TensorInfo input_info           = TensorInfo{ TensorShape(28U, 33U, 2U), 1, DataType::F32, default_data_layout };
    const TensorInfo output_info          = TensorInfo{ TensorShape(26U, 21U, 2U), 1, DataType::F32, default_data_layout };
    constexpr auto   interpolation_policy = InterpolationPolicy::AREA;

    CLCompileContext   cl_compile_ctx = CLKernelLibrary::get().get_compile_context();
    GpuWorkloadContext gpu_ctx        = GpuWorkloadContext{ &cl_compile_ctx };
    GpuWorkloadSketch  sketch{ &gpu_ctx };

    const TensorInfo sketch_input_info = sketch.create_tensor_info(input_info);

    ResizeAttributes attributes{};
    attributes.interpolation_policy(interpolation_policy);

    Status status = GpuResize::validate_op(sketch, &sketch_input_info, attributes);
    ARM_COMPUTE_EXPECT(bool(status) == false, framework::LogLevel::ERRORS);
}

TEST_CASE(UnsupportedLayout, framework::DatasetMode::ALL)
{
    const TensorInfo input_info           = TensorInfo{ default_input_shape, 1, default_data_type, DataLayout::NCHW };
    const TensorInfo output_info          = TensorInfo{ default_output_shape, 1, default_data_type, DataLayout::NCHW };
    constexpr auto   interpolation_policy = InterpolationPolicy::BILINEAR;

    CLCompileContext   cl_compile_ctx = CLKernelLibrary::get().get_compile_context();
    GpuWorkloadContext gpu_ctx        = GpuWorkloadContext{ &cl_compile_ctx };
    GpuWorkloadSketch  sketch{ &gpu_ctx };

    const TensorInfo sketch_input_info = sketch.create_tensor_info(input_info);

    ResizeAttributes attributes{};
    attributes.interpolation_policy(interpolation_policy);

    Status status = GpuResize::validate_op(sketch, &sketch_input_info, attributes);
    ARM_COMPUTE_EXPECT(bool(status) == false, framework::LogLevel::ERRORS);
}

TEST_SUITE_END() // Validate

template <typename T>
using DynamicFusionResizeFixture = DynamicFusionResizeValidationFixture<CLTensor, CLAccessor, GpuResize, T>;

TEST_SUITE(Float)
TEST_SUITE(FP32)

const auto f32_shape = combine((SCALE_PRECOMMIT_SHAPE_DATASET(num_elements_per_vector<float>())), framework::dataset::make("DataType", DataType::F32));

FIXTURE_DATA_TEST_CASE(Run, DynamicFusionResizeFixture<float>, framework::DatasetMode::ALL, ASSEMBLE_DATASET_DYNAMIC_FUSION(f32_shape, ScaleSamplingPolicySet))
{
    //Create valid region
    TensorInfo        src_info(_shape, 1, _data_type);
    const ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _interpolation_policy, _sampling_policy, false);

    // Validate output
    validate(CLAccessor(_target), _reference, valid_region, tolerance_f32, tolerance_num_f32, tolerance_f32_absolute);
}

FIXTURE_DATA_TEST_CASE(RunAlignCorners, DynamicFusionResizeFixture<float>, framework::DatasetMode::ALL, ASSEMBLE_DATASET_DYNAMIC_FUSION(f32_shape, ScaleAlignCornersSamplingPolicySet))
{
    //Create valid region
    TensorInfo        src_info(_shape, 1, _data_type);
    const ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _interpolation_policy, _sampling_policy, false);

    // Validate output
    validate(CLAccessor(_target), _reference, valid_region, tolerance_f32, tolerance_num_f32, tolerance_f32_absolute);
}
const auto f32_nightly_shape = combine((SCALE_NIGHTLY_SHAPE_DATASET(num_elements_per_vector<float>())), framework::dataset::make("DataType", DataType::F32));
FIXTURE_DATA_TEST_CASE(RunNightly, DynamicFusionResizeFixture<float>, framework::DatasetMode::NIGHTLY, ASSEMBLE_DATASET_DYNAMIC_FUSION(f32_nightly_shape, ScaleSamplingPolicySet))
{
    //Create valid region
    TensorInfo        src_info(_shape, 1, _data_type);
    const ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _interpolation_policy, _sampling_policy, false);

    // Validate output
    validate(CLAccessor(_target), _reference, valid_region, tolerance_f32, tolerance_num_f32, tolerance_f32_absolute);
}
FIXTURE_DATA_TEST_CASE(RunNightlyAlignCorners, DynamicFusionResizeFixture<float>, framework::DatasetMode::NIGHTLY, ASSEMBLE_DATASET_DYNAMIC_FUSION(f32_nightly_shape,
                       ScaleAlignCornersSamplingPolicySet))
{
    //Create valid region
    TensorInfo        src_info(_shape, 1, _data_type);
    const ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _interpolation_policy, _sampling_policy, false);

    // Validate output
    validate(CLAccessor(_target), _reference, valid_region, tolerance_f32, tolerance_num_f32, tolerance_f32_absolute);
}
TEST_SUITE_END() // FP32

TEST_SUITE(FP16)
const auto f16_shape = combine((SCALE_PRECOMMIT_SHAPE_DATASET(num_elements_per_vector<half>())), framework::dataset::make("DataType", DataType::F16));
FIXTURE_DATA_TEST_CASE(Run, DynamicFusionResizeFixture<half>, framework::DatasetMode::ALL, ASSEMBLE_DATASET_DYNAMIC_FUSION(f16_shape, ScaleSamplingPolicySet))
{
    //Create valid region
    TensorInfo        src_info(_shape, 1, _data_type);
    const ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _interpolation_policy, _sampling_policy, false);

    // Validate output
    validate(CLAccessor(_target), _reference, valid_region, tolerance_f16, 0.0f, abs_tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunAlignCorners, DynamicFusionResizeFixture<half>, framework::DatasetMode::ALL, ASSEMBLE_DATASET_DYNAMIC_FUSION(f16_shape, ScaleAlignCornersSamplingPolicySet))
{
    //Create valid region
    TensorInfo        src_info(_shape, 1, _data_type);
    const ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _interpolation_policy, _sampling_policy, false);

    // Validate output
    validate(CLAccessor(_target), _reference, valid_region, tolerance_f16, 0.0f, abs_tolerance_f16);
}
const auto f16_nightly_shape = combine((SCALE_NIGHTLY_SHAPE_DATASET(num_elements_per_vector<half>())), framework::dataset::make("DataType", DataType::F16));
FIXTURE_DATA_TEST_CASE(RunNightly, DynamicFusionResizeFixture<half>, framework::DatasetMode::NIGHTLY, ASSEMBLE_DATASET_DYNAMIC_FUSION(f16_nightly_shape, ScaleSamplingPolicySet))
{
    //Create valid region
    TensorInfo        src_info(_shape, 1, _data_type);
    const ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _interpolation_policy, _sampling_policy, false);

    // Validate output
    validate(CLAccessor(_target), _reference, valid_region, tolerance_f16, 0.0f, abs_tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunNightlyAlignCorners, DynamicFusionResizeFixture<half>, framework::DatasetMode::NIGHTLY, ASSEMBLE_DATASET_DYNAMIC_FUSION(f16_nightly_shape,
                       ScaleAlignCornersSamplingPolicySet))
{
    //Create valid region
    TensorInfo        src_info(_shape, 1, _data_type);
    const ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _interpolation_policy, _sampling_policy, false);

    // Validate output
    validate(CLAccessor(_target), _reference, valid_region, tolerance_f16, 0.0f, abs_tolerance_f16);
}
TEST_SUITE_END() // FP16
TEST_SUITE_END() // Float

TEST_SUITE(Integer)
TEST_SUITE(U8)
const auto u8_shape = combine((SCALE_PRECOMMIT_SHAPE_DATASET(num_elements_per_vector<uint8_t>())), framework::dataset::make("DataType", DataType::U8));
FIXTURE_DATA_TEST_CASE(Run, DynamicFusionResizeFixture<uint8_t>, framework::DatasetMode::ALL, ASSEMBLE_DATASET_DYNAMIC_FUSION(u8_shape, ScaleSamplingPolicySet))
{
    //Create valid region
    TensorInfo        src_info(_shape, 1, _data_type);
    const ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _interpolation_policy, _sampling_policy, false);

    // Validate output
    validate(CLAccessor(_target), _reference, valid_region, tolerance_q8);
}
FIXTURE_DATA_TEST_CASE(RunAlignCorners, DynamicFusionResizeFixture<uint8_t>, framework::DatasetMode::ALL, ASSEMBLE_DATASET_DYNAMIC_FUSION(u8_shape, ScaleAlignCornersSamplingPolicySet))
{
    //Create valid region
    TensorInfo        src_info(_shape, 1, _data_type);
    const ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _interpolation_policy, _sampling_policy, false);

    // Validate output
    validate(CLAccessor(_target), _reference, valid_region, tolerance_q8);
}
const auto u8_nightly_shape = combine((SCALE_NIGHTLY_SHAPE_DATASET(num_elements_per_vector<uint8_t>())), framework::dataset::make("DataType", DataType::U8));
FIXTURE_DATA_TEST_CASE(RunNightly, DynamicFusionResizeFixture<uint8_t>, framework::DatasetMode::NIGHTLY, ASSEMBLE_DATASET_DYNAMIC_FUSION(u8_nightly_shape, ScaleSamplingPolicySet))
{
    //Create valid region
    TensorInfo        src_info(_shape, 1, _data_type);
    const ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _interpolation_policy, _sampling_policy, false);

    // Validate output
    validate(CLAccessor(_target), _reference, valid_region, tolerance_q8);
}
FIXTURE_DATA_TEST_CASE(RunNightlyAlignCorners, DynamicFusionResizeFixture<uint8_t>, framework::DatasetMode::NIGHTLY, ASSEMBLE_DATASET_DYNAMIC_FUSION(u8_nightly_shape,
                       ScaleAlignCornersSamplingPolicySet))
{
    //Create valid region
    TensorInfo        src_info(_shape, 1, _data_type);
    const ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _interpolation_policy, _sampling_policy, false);

    // Validate output
    validate(CLAccessor(_target), _reference, valid_region, tolerance_q8);
}
TEST_SUITE_END() // U8

TEST_SUITE(S16)
const auto s16_shape = combine((SCALE_PRECOMMIT_SHAPE_DATASET(num_elements_per_vector<int16_t>())), framework::dataset::make("DataType", DataType::S16));
FIXTURE_DATA_TEST_CASE(Run, DynamicFusionResizeFixture<int16_t>, framework::DatasetMode::ALL, ASSEMBLE_DATASET_DYNAMIC_FUSION(s16_shape, ScaleSamplingPolicySet))
{
    //Create valid region
    TensorInfo        src_info(_shape, 1, _data_type);
    const ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _interpolation_policy, _sampling_policy, false);

    // Validate output
    validate(CLAccessor(_target), _reference, valid_region, tolerance_s16);
}
FIXTURE_DATA_TEST_CASE(RunAlignCorners, DynamicFusionResizeFixture<int16_t>, framework::DatasetMode::ALL, ASSEMBLE_DATASET_DYNAMIC_FUSION(s16_shape, ScaleAlignCornersSamplingPolicySet))
{
    //Create valid region
    TensorInfo        src_info(_shape, 1, _data_type);
    const ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _interpolation_policy, _sampling_policy, false);

    // Validate output
    validate(CLAccessor(_target), _reference, valid_region, tolerance_s16);
}
const auto s16_nightly_shape = combine((SCALE_NIGHTLY_SHAPE_DATASET(num_elements_per_vector<int16_t>())), framework::dataset::make("DataType", DataType::S16));
FIXTURE_DATA_TEST_CASE(RunNightly, DynamicFusionResizeFixture<int16_t>, framework::DatasetMode::NIGHTLY, ASSEMBLE_DATASET_DYNAMIC_FUSION(s16_nightly_shape, ScaleSamplingPolicySet))
{
    //Create valid region
    TensorInfo        src_info(_shape, 1, _data_type);
    const ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _interpolation_policy, _sampling_policy, false);

    // Validate output
    validate(CLAccessor(_target), _reference, valid_region, tolerance_s16);
}
FIXTURE_DATA_TEST_CASE(RunNightlyAlignCorners, DynamicFusionResizeFixture<int16_t>, framework::DatasetMode::NIGHTLY, ASSEMBLE_DATASET_DYNAMIC_FUSION(s16_nightly_shape,
                       ScaleAlignCornersSamplingPolicySet))
{
    //Create valid region
    TensorInfo        src_info(_shape, 1, _data_type);
    const ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _interpolation_policy, _sampling_policy, false);

    // Validate output
    validate(CLAccessor(_target), _reference, valid_region, tolerance_s16);
}
TEST_SUITE_END() // S16
TEST_SUITE_END() // Integer

template <typename T>
using DynamicFusionResizeQuantizedFixture = DynamicFusionResizeQuantizedValidationFixture<CLTensor, CLAccessor, GpuResize, T>;
TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
const auto qasymm8_shape = combine((SCALE_PRECOMMIT_SHAPE_DATASET(num_elements_per_vector<uint8_t>())), framework::dataset::make("DataType", DataType::QASYMM8));
FIXTURE_DATA_TEST_CASE(Run, DynamicFusionResizeQuantizedFixture<uint8_t>, framework::DatasetMode::ALL, ASSEMBLE_QUANTIZED_DATASET_DYNAMIC_FUSION(qasymm8_shape, ScaleSamplingPolicySet,
                       QuantizationInfoSet))
{
    //Create valid region
    TensorInfo        src_info(_shape, 1, _data_type);
    const ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _interpolation_policy, _sampling_policy, false);

    // Validate output
    validate(CLAccessor(_target), _reference, valid_region, tolerance_q8);
}
FIXTURE_DATA_TEST_CASE(RunAlignCorners, DynamicFusionResizeQuantizedFixture<uint8_t>, framework::DatasetMode::ALL, ASSEMBLE_QUANTIZED_DATASET_DYNAMIC_FUSION(qasymm8_shape,
                       ScaleAlignCornersSamplingPolicySet,
                       QuantizationInfoSet))
{
    //Create valid region
    TensorInfo        src_info(_shape, 1, _data_type);
    const ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _interpolation_policy, _sampling_policy, false);

    // Validate output
    validate(CLAccessor(_target), _reference, valid_region, tolerance_q8);
}
const auto qasymm8_nightly_shape = combine((SCALE_NIGHTLY_SHAPE_DATASET(num_elements_per_vector<uint8_t>())), framework::dataset::make("DataType", DataType::QASYMM8));
FIXTURE_DATA_TEST_CASE(RunNightly, DynamicFusionResizeQuantizedFixture<uint8_t>, framework::DatasetMode::NIGHTLY, ASSEMBLE_QUANTIZED_DATASET_DYNAMIC_FUSION(qasymm8_nightly_shape,
                       ScaleSamplingPolicySet,
                       QuantizationInfoSet))
{
    //Create valid region
    TensorInfo        src_info(_shape, 1, _data_type);
    const ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _interpolation_policy, _sampling_policy, false);

    // Validate output
    validate(CLAccessor(_target), _reference, valid_region, tolerance_q8);
}
FIXTURE_DATA_TEST_CASE(RunNightlyAlignCorners, DynamicFusionResizeQuantizedFixture<uint8_t>, framework::DatasetMode::NIGHTLY, ASSEMBLE_QUANTIZED_DATASET_DYNAMIC_FUSION(qasymm8_nightly_shape,
                       ScaleAlignCornersSamplingPolicySet,
                       QuantizationInfoSet))
{
    //Create valid region
    TensorInfo        src_info(_shape, 1, _data_type);
    const ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _interpolation_policy, _sampling_policy, false);

    // Validate output
    validate(CLAccessor(_target), _reference, valid_region, tolerance_q8);
}
TEST_SUITE_END() // QASYMM8

TEST_SUITE(QASYMM8_SIGNED)
const auto qasymm8_signed_shape = combine((SCALE_PRECOMMIT_SHAPE_DATASET(num_elements_per_vector<int8_t>())), framework::dataset::make("DataType", DataType::QASYMM8_SIGNED));
FIXTURE_DATA_TEST_CASE(Run, DynamicFusionResizeQuantizedFixture<int8_t>, framework::DatasetMode::ALL, ASSEMBLE_QUANTIZED_DATASET_DYNAMIC_FUSION(qasymm8_signed_shape, ScaleSamplingPolicySet,
                       QuantizationInfoSet))
{
    //Create valid region
    TensorInfo        src_info(_shape, 1, _data_type);
    const ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _interpolation_policy, _sampling_policy, false);

    // Validate output
    validate(CLAccessor(_target), _reference, valid_region, tolerance_qs8);
}
FIXTURE_DATA_TEST_CASE(RunAlignCorners, DynamicFusionResizeQuantizedFixture<int8_t>, framework::DatasetMode::ALL, ASSEMBLE_QUANTIZED_DATASET_DYNAMIC_FUSION(qasymm8_signed_shape,
                       ScaleAlignCornersSamplingPolicySet,
                       QuantizationInfoSet))
{
    //Create valid region
    TensorInfo        src_info(_shape, 1, _data_type);
    const ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _interpolation_policy, _sampling_policy, false);

    // Validate output
    validate(CLAccessor(_target), _reference, valid_region, tolerance_qs8);
}
const auto qasymm8_signed_nightly_shape = combine((SCALE_NIGHTLY_SHAPE_DATASET(num_elements_per_vector<int8_t>())), framework::dataset::make("DataType", DataType::QASYMM8_SIGNED));
FIXTURE_DATA_TEST_CASE(RunNightly, DynamicFusionResizeQuantizedFixture<int8_t>, framework::DatasetMode::NIGHTLY, ASSEMBLE_QUANTIZED_DATASET_DYNAMIC_FUSION(qasymm8_signed_nightly_shape,
                       ScaleSamplingPolicySet,
                       QuantizationInfoSet))
{
    //Create valid region
    TensorInfo        src_info(_shape, 1, _data_type);
    const ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _interpolation_policy, _sampling_policy, false);

    // Validate output
    validate(CLAccessor(_target), _reference, valid_region, tolerance_qs8);
}
FIXTURE_DATA_TEST_CASE(RunNightlyAlignCorners, DynamicFusionResizeQuantizedFixture<int8_t>, framework::DatasetMode::NIGHTLY, ASSEMBLE_QUANTIZED_DATASET_DYNAMIC_FUSION(qasymm8_signed_nightly_shape,
                       ScaleAlignCornersSamplingPolicySet,
                       QuantizationInfoSet))
{
    //Create valid region
    TensorInfo        src_info(_shape, 1, _data_type);
    const ValidRegion valid_region = calculate_valid_region_scale(src_info, _reference.shape(), _interpolation_policy, _sampling_policy, false);

    // Validate output
    validate(CLAccessor(_target), _reference, valid_region, tolerance_qs8);
}
TEST_SUITE_END() // QASYMM8_SIGNED

TEST_SUITE_END() // Quantized

TEST_SUITE_END() // RESIZE
TEST_SUITE_END() // DYNAMIC_FUSION
TEST_SUITE_END() // CL

} // namespace validation
} // namespace test
} // namespace arm_compute
