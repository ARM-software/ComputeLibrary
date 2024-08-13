/*
 * Copyright (c) 2017-2024 Arm Limited.
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
#ifdef __aarch64__

#include "arm_compute/runtime/experimental/operators/CpuDepthwiseConv2d.h"

#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"

#include "src/core/helpers/MemoryHelpers.h"
#include "tests/datasets/DepthwiseConvolutionLayerDataset.h"
#include "tests/datasets/DilatedDepthwiseConvolutionLayerDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/framework/Macros.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/validation/fixtures/CpuDepthwiseConv2dFixture.h"
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
constexpr RelativeTolerance<float> tolerance_f32(
    0.01f); /**< Tolerance value for comparing reference's output against implementation's output for DataType::F32 */
constexpr AbsoluteTolerance<uint8_t> tolerance_qasymm8(
    1); /**< Tolerance value for comparing reference's output against implementation's output for DataType::QASYMM8 */
constexpr AbsoluteTolerance<int8_t> tolerance_qasymm8_signed(
    1); /**< Tolerance value for comparing reference's output against implementation's output for DataType::QASYMM8_SIGNED */

const auto depth_multipliers       = make("DepthMultiplier", {1, 2, 8});
const auto large_depth_multipliers = make("DepthMultiplier", {5, 32});

// Activation Functions
const auto NoActivation = make("ActivationInfo", ActivationLayerInfo());

const auto ActivationFunctionsDataset =
    make("ActivationInfo", {ActivationLayerInfo(), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)});

const auto ActivationFunctionsDatasetNightly =
    make("ActivationInfo",
         {
             ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 0.5f),
             ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 0.5f, -0.5f),
             ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LEAKY_RELU, 0.1f),
             ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::SOFT_RELU),
             ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::ELU),
             ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::ABS),
             ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC),
             ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::TANH),
             ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::SQUARE),
             ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::SWISH),
             ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::HARD_SWISH),
             ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LINEAR, 2.f, 1.f),
#ifdef __aarch64__
             ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::GELU),
#endif // __aarch64__
         });

const auto ActivationFunctionsQuantizedSmallDataset =
    make("ActivationInfo", {ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)});

const auto ActivationFunctionsQuantizedDataset =
    make("ActivationInfo",
         {
             ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 0.5f),
             ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 0.5f, -0.5f),
         });

// This is only used when there is fused activation
const auto input_qinfo_dataset = make("InputQInfo",
                                      {
                                          QuantizationInfo(0.3f, 10),
                                          QuantizationInfo(2.2f, 10),
                                      });

const auto IgnoredQuantizationInfo = make("IgnoredQuantizationInfo", QuantizationInfo());

} // namespace

TEST_SUITE(NEON)
TEST_SUITE(OPERATORS)
TEST_SUITE(CpuDepthwiseConv2d)

TEST_CASE(OpCpuDepthwiseConv2dMemoryInjection, framework::DatasetMode::ALL)
{
    auto conv = std::make_unique<arm_compute::experimental::op::CpuDepthwiseConv2d>();

    auto src_shape     = TensorShape(7U, 7U);
    auto weights_shape = TensorShape(1U, 1U);
    auto bias_shape    = TensorShape(1U);
    auto output_shape  = TensorShape(7U, 7U);

    auto       src_info     = TensorInfo(src_shape, 1, DataType::F32, DataLayout::NHWC);
    const auto weights_info = TensorInfo(weights_shape, 1, DataType::F32, DataLayout::NHWC);
    const auto biases_info  = TensorInfo(bias_shape, 1, DataType::F32, DataLayout::NHWC);
    auto       dst_info     = TensorInfo(output_shape, 1, DataType::F32, DataLayout::NHWC);

    conv->configure(&src_info, &weights_info, &biases_info, &dst_info, PadStrideInfo(1, 1, 0, 0));
    auto const status = conv->validate(&src_info, &weights_info, &biases_info, &dst_info, PadStrideInfo(1, 1, 0, 0));
    ARM_COMPUTE_ASSERT(status);

    // tensors are newly created every call of this lambda function
    auto src     = create_tensor<Tensor>(src_info);
    auto weights = create_tensor<Tensor>(weights_info);
    auto biases  = create_tensor<Tensor>(biases_info);

    src.allocator()->allocate();
    weights.allocator()->allocate();
    biases.allocator()->allocate();

    ITensorPack run_pack{
        {TensorType::ACL_SRC_0, &src}, {TensorType::ACL_SRC_1, &weights}, {TensorType::ACL_SRC_2, &biases}};

    auto mg = MemoryGroup{};
    auto ws = manage_workspace<Tensor>(conv->workspace(), mg, run_pack, run_pack);

    auto run_conv = [&]() -> Tensor
    {
        auto dst = create_tensor<Tensor>(dst_info);
        dst.allocator()->allocate();
        run_pack.add_tensor(TensorType::ACL_DST, &dst);

        library->fill_tensor_value(Accessor(src), 1.f);
        library->fill_tensor_value(Accessor(weights), 2.f);
        library->fill_tensor_value(Accessor(biases), 3.f);
        // This operator is configured once and captured by this lambda.
        conv->prepare(run_pack);
        conv->run(run_pack);
        return dst;
    };
    auto result_0 = run_conv();
    auto result_1 = run_conv();
    for (size_t i = 0; i < result_0.info()->tensor_shape().total_size(); ++i)
    {
        ARM_COMPUTE_EXPECT((reinterpret_cast<float *>(result_0.buffer()))[i] ==
                               (reinterpret_cast<float *>(result_1.buffer()))[i],
                           framework::LogLevel::ERRORS);
    }
}

DATA_TEST_CASE(
    Validate3x3,
    framework::DatasetMode::ALL,
    zip(make("InputInfo",
             {
                 TensorInfo(TensorShape(32U, 18U, 2U), 1, DataType::F32),     // Mismatching data type input/weights
                 TensorInfo(TensorShape(32U, 18U, 3U), 1, DataType::F32),     // Mismatching input feature maps
                 TensorInfo(TensorShape(32U, 18U, 2U), 1, DataType::F32),     // Unsupported weights dimensions
                 TensorInfo(TensorShape(32U, 18U, 2U), 1, DataType::F32),     // Mismatching depth multiplier
                 TensorInfo(TensorShape(32U, 18U, 2U), 1, DataType::QASYMM8), // Invalid stride
                 TensorInfo(TensorShape(32U, 18U, 2U), 1, DataType::F32),     // Invalid biases size
                 TensorInfo(TensorShape(32U, 18U, 2U), 1, DataType::F32),     // Invalid biases dimensions
                 TensorInfo(TensorShape(32U, 18U, 2U), 1, DataType::F32),     // Invalid output size
                 TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),     // patch size bigger than input width
                 TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),     // dilation < 1
             }),
        make("WeightsInfo",
             {
                 TensorInfo(TensorShape(3U, 3U, 2U, 2U), 1, DataType::F16),
                 TensorInfo(TensorShape(3U, 3U, 2U, 2U), 1, DataType::F32),
                 TensorInfo(TensorShape(5U, 5U, 2U, 2U), 1, DataType::F32),
                 TensorInfo(TensorShape(3U, 3U, 2U, 2U), 1, DataType::F32),
                 TensorInfo(TensorShape(3U, 3U, 2U, 2U), 1, DataType::QASYMM8),
                 TensorInfo(TensorShape(3U, 3U, 2U, 2U), 1, DataType::F32),
                 TensorInfo(TensorShape(3U, 3U, 2U, 2U), 1, DataType::F32),
                 TensorInfo(TensorShape(3U, 3U, 2U, 2U), 1, DataType::F32),
                 TensorInfo(TensorShape(3U, 3U, 2U, 2U), 1, DataType::F32),
                 TensorInfo(TensorShape(3U, 3U, 2U, 2U), 1, DataType::F32),
             }),
        make("BiasesInfo",
             {
                 TensorInfo(TensorShape(2U), 1, DataType::F32),
                 TensorInfo(TensorShape(2U), 1, DataType::F32),
                 TensorInfo(TensorShape(2U), 1, DataType::F32),
                 TensorInfo(TensorShape(2U), 1, DataType::F32),
                 TensorInfo(TensorShape(2U), 1, DataType::S32),
                 TensorInfo(TensorShape(4U), 1, DataType::F32),
                 TensorInfo(TensorShape(2U, 2U), 1, DataType::F32),
                 TensorInfo(TensorShape(2U), 1, DataType::F32),
                 TensorInfo(TensorShape(2U), 1, DataType::F32),
                 TensorInfo(TensorShape(2U), 1, DataType::F32),
             }),
        make("OutputInfo",
             {
                 TensorInfo(TensorShape(30U, 16U, 2U), 1, DataType::F32),
                 TensorInfo(TensorShape(30U, 16U, 2U), 1, DataType::F32),
                 TensorInfo(TensorShape(30U, 16U, 2U), 1, DataType::F32),
                 TensorInfo(TensorShape(30U, 16U, 2U), 1, DataType::F32),
                 TensorInfo(TensorShape(30U, 16U, 2U), 1, DataType::QASYMM8),
                 TensorInfo(TensorShape(30U, 16U, 2U), 1, DataType::F32),
                 TensorInfo(TensorShape(30U, 16U, 2U), 1, DataType::F32),
                 TensorInfo(TensorShape(32U, 18U, 2U), 1, DataType::F32),
                 TensorInfo(TensorShape(25U, 11U, 2U), 1, DataType::F32),
                 TensorInfo(TensorShape(25U, 11U, 2U), 1, DataType::F32),
             }),
        make("ConvInfo",
             {
                 PadStrideInfo(1, 1, 0, 0),
                 PadStrideInfo(1, 1, 0, 0),
                 PadStrideInfo(1, 1, 0, 0),
                 PadStrideInfo(1, 1, 0, 0),
                 PadStrideInfo(4, 1, 0, 0),
                 PadStrideInfo(1, 1, 0, 0),
                 PadStrideInfo(1, 1, 0, 0),
                 PadStrideInfo(1, 1, 0, 0),
                 PadStrideInfo(1, 1, 0, 0),
                 PadStrideInfo(1, 1, 0, 0),
             }),
        make("DepthMultiplier",
             {
                 1,
                 1,
                 1,
                 3,
                 1,
                 1,
                 1,
                 1,
                 1,
                 1,
             }),
        make("Dilation",
             {
                 Size2D(1U, 1U),
                 Size2D(1U, 1U),
                 Size2D(1U, 1U),
                 Size2D(1U, 1U),
                 Size2D(1U, 1U),
                 Size2D(1U, 1U),
                 Size2D(1U, 1U),
                 Size2D(1U, 1U),
                 Size2D(25U, 1U),
                 Size2D(0U, 1U),
             }),
        make("Expected", {false, false, false, false, false, false, false, false, false, false})),
    input_info,
    weights_info,
    biases_info,
    output_info,
    conv_info,
    depth_multiplier,
    dilation,
    expected)
{
    bool is_valid = bool(experimental::op::CpuDepthwiseConv2d::validate(
        &input_info.clone()->set_is_resizable(false), &weights_info.clone()->set_is_resizable(false),
        &biases_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), conv_info,
        depth_multiplier, ActivationLayerInfo(), dilation));
    ARM_COMPUTE_EXPECT(is_valid == expected, framework::LogLevel::ERRORS);
}

DATA_TEST_CASE(
    ValidateGeneric,
    framework::DatasetMode::ALL,
    zip(make("InputInfo",
             {
                 TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32), // Mismatching data type input/weights
                 TensorInfo(TensorShape(27U, 13U, 3U), 1, DataType::F32), // Mismatching input feature maps
                 TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32), // Mismatching depth multiplier
                 TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32), // Invalid biases size
                 TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32), // Invalid biases dimensions
                 TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32), // Invalid output size
                 TensorInfo(TensorShape(27U, 13U, 8U), 1, DataType::F32), // Patch size bigger than input width
                 TensorInfo(TensorShape(27U, 13U, 8U), 1, DataType::F32), // Dilation < 1
             }),
        make("WeightsInfo",
             {
                 TensorInfo(TensorShape(3U, 3U, 2U), 1, DataType::F16),
                 TensorInfo(TensorShape(3U, 3U, 2U), 1, DataType::F32),
                 TensorInfo(TensorShape(3U, 3U, 2U), 1, DataType::F32),
                 TensorInfo(TensorShape(3U, 3U, 2U), 1, DataType::F32),
                 TensorInfo(TensorShape(3U, 3U, 2U), 1, DataType::F32),
                 TensorInfo(TensorShape(3U, 3U, 2U), 1, DataType::F32),
                 TensorInfo(TensorShape(3U, 3U, 16U), 1, DataType::F32),
                 TensorInfo(TensorShape(3U, 3U, 16U), 1, DataType::F32),
             }),
        make("BiasesInfo",
             {
                 TensorInfo(TensorShape(2U), 1, DataType::F32),
                 TensorInfo(TensorShape(2U), 1, DataType::F32),
                 TensorInfo(TensorShape(2U), 1, DataType::F32),
                 TensorInfo(TensorShape(4U), 1, DataType::F32),
                 TensorInfo(TensorShape(2U, 2U), 1, DataType::F32),
                 TensorInfo(TensorShape(2U), 1, DataType::F32),
                 TensorInfo(TensorShape(16U), 1, DataType::F32),
                 TensorInfo(TensorShape(16U), 1, DataType::F32),
             }),
        make("OutputInfo",
             {
                 TensorInfo(TensorShape(25U, 11U, 2U), 1, DataType::F32),
                 TensorInfo(TensorShape(25U, 11U, 2U), 1, DataType::F32),
                 TensorInfo(TensorShape(25U, 11U, 2U), 1, DataType::F32),
                 TensorInfo(TensorShape(25U, 11U, 2U), 1, DataType::F32),
                 TensorInfo(TensorShape(25U, 11U, 2U), 1, DataType::F32),
                 TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),
                 TensorInfo(TensorShape(25U, 11U, 16U), 1, DataType::F32),
                 TensorInfo(TensorShape(25U, 11U, 16U), 1, DataType::F32),
             }),
        make("ConvInfo",
             {
                 PadStrideInfo(1, 1, 0, 0),
                 PadStrideInfo(1, 1, 0, 0),
                 PadStrideInfo(1, 1, 0, 0),
                 PadStrideInfo(1, 1, 0, 0),
                 PadStrideInfo(1, 1, 0, 0),
                 PadStrideInfo(1, 1, 0, 0),
                 PadStrideInfo(1, 1, 0, 0),
                 PadStrideInfo(1, 1, 0, 0),
             }),
        make("DepthMultiplier",
             {
                 1,
                 1,
                 3,
                 1,
                 1,
                 1,
                 2,
                 2,
             }),
        make("Dilation",
             {
                 Size2D(1U, 1U),
                 Size2D(1U, 1U),
                 Size2D(1U, 1U),
                 Size2D(1U, 1U),
                 Size2D(1U, 1U),
                 Size2D(1U, 1U),
                 Size2D(25U, 1U),
                 Size2D(0U, 1U),
             }),
        make("Expected", {false, false, false, false, false, false, false, false})),
    input_info,
    weights_info,
    biases_info,
    output_info,
    conv_info,
    depth_multiplier,
    dilation,
    expected)
{
    bool is_valid = bool(experimental::op::CpuDepthwiseConv2d::validate(
        &input_info.clone()->set_is_resizable(false), &weights_info.clone()->set_is_resizable(false),
        &biases_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), conv_info,
        depth_multiplier, ActivationLayerInfo(), dilation));
    ARM_COMPUTE_EXPECT(is_valid == expected, framework::LogLevel::ERRORS);
}

template <typename T>
using CpuDepthwiseConv2dFixture =
    CpuDepthwiseConv2dValidationFixture<Tensor, Accessor, experimental::op::CpuDepthwiseConv2d, T>;
template <typename T>
using CpuDepthwiseConv2dMixedDataLayoutFixture =
    CpuDepthwiseConv2dValidationFixture<Tensor, Accessor, experimental::op::CpuDepthwiseConv2d, T, true>;
template <typename T>
using CpuDepthwiseConv2dVariableWeightsFixture =
    CpuDepthwiseConv2dValidationFixture<Tensor, Accessor, experimental::op::CpuDepthwiseConv2d, T, false, false, true>;

TEST_SUITE(Float)
TEST_SUITE(F32)

FIXTURE_DATA_TEST_CASE_NEW(RunActivations,
                           CpuDepthwiseConv2dFixture<float>,
                           framework::DatasetMode::NIGHTLY,
                           combine(make("In", TensorShape(33U, 27U, 11U, 3U)),
                                   make("Weights", Size2D(3U, 4U)),
                                   make("Info", PadStrideInfo(1, 2, 0, 1)),
                                   make("Dilation", Size2D(2U, 2U)),
                                   make("DepthMultiplier", {5}),
                                   make("DataType", DataType::F32),
                                   make("DataLayout", {DataLayout::NHWC}),
                                   ActivationFunctionsDatasetNightly))
{
    validate(Accessor(_target), _reference, tolerance_f32);
}

TEST_SUITE(Generic)
FIXTURE_DATA_TEST_CASE_NEW(RunSmall,
                           CpuDepthwiseConv2dFixture<float>,
                           framework::DatasetMode::PRECOMMIT,
                           combine(datasets::SmallDepthwiseConvolutionLayerDataset(),
                                   depth_multipliers,
                                   make("DataType", DataType::F32),
                                   make("DataLayout", {DataLayout::NHWC}),
                                   ActivationFunctionsDataset))
{
    validate(Accessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE_NEW(RunMixedDataLayout,
                           CpuDepthwiseConv2dMixedDataLayoutFixture<float>,
                           framework::DatasetMode::PRECOMMIT,
                           combine(datasets::SmallDepthwiseConvolutionLayerDataset(),
                                   make("DepthMultiplier", {2}),
                                   make("DataType", DataType::F32),
                                   make("DataLayout", {DataLayout::NHWC}),
                                   make("ActivationInfo", ActivationLayerInfo())))
{
    validate(Accessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE_NEW(RunLarge,
                           CpuDepthwiseConv2dFixture<float>,
                           framework::DatasetMode::NIGHTLY,
                           combine(datasets::LargeCpuDepthwiseConv2dDataset(),
                                   large_depth_multipliers,
                                   make("DataType", DataType::F32),
                                   make("DataLayout", {DataLayout::NHWC}),
                                   make("ActivationInfo", {ActivationLayerInfo()})))
{
    validate(Accessor(_target), _reference, tolerance_f32);
}
TEST_SUITE(Dilation)
FIXTURE_DATA_TEST_CASE_NEW(RunSmall,
                           CpuDepthwiseConv2dFixture<float>,
                           framework::DatasetMode::PRECOMMIT,
                           combine(datasets::SmallDepthwiseDilatedConvolutionLayerDataset(),
                                   depth_multipliers,
                                   make("DataType", DataType::F32),
                                   make("DataLayout", {DataLayout::NHWC}),
                                   ActivationFunctionsDataset))
{
    validate(Accessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE_NEW(RunLarge,
                           CpuDepthwiseConv2dFixture<float>,
                           framework::DatasetMode::NIGHTLY,
                           combine(datasets::LargeDepthwiseDilatedConvolutionLayerDataset(),
                                   large_depth_multipliers,
                                   make("DataType", DataType::F32),
                                   make("DataLayout", {DataLayout::NHWC}),
                                   make("ActivationInfo", {ActivationLayerInfo()})))
{
    validate(Accessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END() // Dilation
TEST_SUITE_END() // Generic

TEST_SUITE(W3x3)
FIXTURE_DATA_TEST_CASE_NEW(RunSmall,
                           CpuDepthwiseConv2dFixture<float>,
                           framework::DatasetMode::PRECOMMIT,
                           combine(datasets::SmallDepthwiseConvolutionLayerDataset3x3(),
                                   depth_multipliers,
                                   make("DataType", DataType::F32),
                                   make("DataLayout", {DataLayout::NHWC}),
                                   ActivationFunctionsDataset))
{
    validate(Accessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE_NEW(RunLarge,
                           CpuDepthwiseConv2dFixture<float>,
                           framework::DatasetMode::NIGHTLY,
                           combine(datasets::LargeDepthwiseConvolutionLayerDataset3x3(),
                                   large_depth_multipliers,
                                   make("DataType", DataType::F32),
                                   make("DataLayout", {DataLayout::NHWC}),
                                   make("ActivationInfo", {ActivationLayerInfo()})))
{
    validate(Accessor(_target), _reference, tolerance_f32);
}
TEST_SUITE(Dilation)
FIXTURE_DATA_TEST_CASE_NEW(RunSmall,
                           CpuDepthwiseConv2dFixture<float>,
                           framework::DatasetMode::PRECOMMIT,
                           combine(datasets::SmallDepthwiseDilatedConvolutionLayerDataset3x3(),
                                   depth_multipliers,
                                   make("DataType", DataType::F32),
                                   make("DataLayout", {DataLayout::NHWC}),
                                   ActivationFunctionsDataset))
{
    validate(Accessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE_NEW(RunLarge,
                           CpuDepthwiseConv2dFixture<float>,
                           framework::DatasetMode::NIGHTLY,
                           combine(datasets::LargeDepthwiseDilatedConvolutionLayerDataset3x3(),
                                   large_depth_multipliers,
                                   make("DataType", DataType::F32),
                                   make("DataLayout", {DataLayout::NHWC}),
                                   make("ActivationInfo", {ActivationLayerInfo()})))
{
    validate(Accessor(_target), _reference, tolerance_f32);
}

TEST_SUITE_END() // Dilation
TEST_SUITE_END() // W3x3

TEST_SUITE(Optimized)
FIXTURE_DATA_TEST_CASE_NEW(RunSmall3x3,
                           CpuDepthwiseConv2dFixture<float>,
                           framework::DatasetMode::PRECOMMIT,
                           combine(datasets::SmallOptimizedDepthwiseConvolutionLayerDataset3x3(),
                                   make("DepthMultiplier", 1),
                                   make("DataType", DataType::F32),
                                   make("DataLayout", {DataLayout::NHWC}),
                                   ActivationFunctionsDataset))
{
    validate(Accessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE_NEW(RunVariableWeightsSmall3x3,
                           CpuDepthwiseConv2dVariableWeightsFixture<float>,
                           framework::DatasetMode::PRECOMMIT,
                           combine(datasets::SmallOptimizedDepthwiseConvolutionLayerDataset3x3(),
                                   make("DepthMultiplier", 1),
                                   make("DataType", DataType::F32),
                                   make("DataLayout", {DataLayout::NHWC}),
                                   ActivationFunctionsDataset))
{
    validate(Accessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE_NEW(RunMixedDataLayout3x3,
                           CpuDepthwiseConv2dMixedDataLayoutFixture<float>,
                           framework::DatasetMode::PRECOMMIT,
                           combine(datasets::SmallOptimizedDepthwiseConvolutionLayerDataset3x3(),
                                   make("DepthMultiplier", 1),
                                   make("DataType", DataType::F32),
                                   make("DataLayout", {DataLayout::NHWC}),
                                   make("ActivationInfo", ActivationLayerInfo())))
{
    validate(Accessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE_NEW(RunSmall5x5,
                           CpuDepthwiseConv2dFixture<float>,
                           framework::DatasetMode::PRECOMMIT,
                           combine(datasets::SmallOptimizedDepthwiseConvolutionLayerDataset5x5(),
                                   make("DepthMultiplier", 1),
                                   make("DataType", DataType::F32),
                                   make("DataLayout", {DataLayout::NHWC}),
                                   ActivationFunctionsDataset))
{
    validate(Accessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE_NEW(RunVariableWeightsSmall5x5,
                           CpuDepthwiseConv2dVariableWeightsFixture<float>,
                           framework::DatasetMode::PRECOMMIT,
                           combine(datasets::SmallOptimizedDepthwiseConvolutionLayerDataset5x5(),
                                   make("DepthMultiplier", 1),
                                   make("DataType", DataType::F32),
                                   make("DataLayout", {DataLayout::NHWC}),
                                   ActivationFunctionsDataset))
{
    validate(Accessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE_NEW(RunLarge3x3,
                           CpuDepthwiseConv2dFixture<float>,
                           framework::DatasetMode::NIGHTLY,
                           combine(datasets::LargeOptimizedDepthwiseConvolutionLayerDataset3x3(),
                                   make("DepthMultiplier", 1),
                                   make("DataType", DataType::F32),
                                   make("DataLayout", {DataLayout::NHWC}),
                                   make("ActivationInfo", {ActivationLayerInfo()})))
{
    validate(Accessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE_NEW(RunVariableWeightsLarge3x3,
                           CpuDepthwiseConv2dVariableWeightsFixture<float>,
                           framework::DatasetMode::NIGHTLY,
                           combine(datasets::LargeOptimizedDepthwiseConvolutionLayerDataset3x3(),
                                   make("DepthMultiplier", 1),
                                   make("DataType", DataType::F32),
                                   make("DataLayout", {DataLayout::NHWC}),
                                   make("ActivationInfo", {ActivationLayerInfo()})))
{
    validate(Accessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END() // Optimized
TEST_SUITE_END() // F32
TEST_SUITE_END() // Float

template <typename T>
using CpuDepthwiseConv2dQuantizedFixture =
    CpuDepthwiseConv2dValidationQuantizedFixture<Tensor, Accessor, experimental::op::CpuDepthwiseConv2d, T>;
template <typename T>
using CpuDepthwiseConv2dQuantizedMixedDataLayoutFixture =
    CpuDepthwiseConv2dValidationQuantizedFixture<Tensor, Accessor, experimental::op::CpuDepthwiseConv2d, T, true>;
using CpuDepthwiseConv2dQuantizedSymmetricPerChannelFixture =
    CpuDepthwiseConv2dValidationQuantizedPerChannelFixture<Tensor,
                                                           Accessor,
                                                           experimental::op::CpuDepthwiseConv2d,
                                                           uint8_t,
                                                           int8_t>;

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)

FIXTURE_DATA_TEST_CASE_NEW(RunActivations,
                           CpuDepthwiseConv2dQuantizedFixture<uint8_t>,
                           framework::DatasetMode::NIGHTLY,
                           combine(make("In", TensorShape(33U, 27U, 11U, 3U)),
                                   make("Weights", Size2D(3U, 4U)),
                                   make("Info", PadStrideInfo(1, 2, 0, 1)),
                                   make("Dilation", Size2D(2U, 2U)),
                                   make("DepthMultiplier", {5}),
                                   make("DataType", DataType::QASYMM8),
                                   make("SrcQuantizationInfo", {QuantizationInfo(0.3f, 10)}),
                                   make("DstQuantizationInfo", {QuantizationInfo(0.05f, 4)}),
                                   make("DataLayout", {DataLayout::NHWC}),
                                   ActivationFunctionsQuantizedDataset))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}

TEST_SUITE(Generic)
FIXTURE_DATA_TEST_CASE_NEW(RunSmall,
                           CpuDepthwiseConv2dQuantizedFixture<uint8_t>,
                           framework::DatasetMode::PRECOMMIT,
                           combine(datasets::SmallDepthwiseConvolutionLayerDataset(),
                                   depth_multipliers,
                                   make("DataType", DataType::QASYMM8),
                                   IgnoredQuantizationInfo,
                                   IgnoredQuantizationInfo,
                                   make("DataLayout", {DataLayout::NHWC}),
                                   NoActivation))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE_NEW(RunSmallWithActivation,
                           CpuDepthwiseConv2dQuantizedFixture<uint8_t>,
                           framework::DatasetMode::PRECOMMIT,
                           combine(datasets::SmallDepthwiseConvolutionLayerDataset(),
                                   depth_multipliers,
                                   make("DataType", DataType::QASYMM8),
                                   input_qinfo_dataset,
                                   make("DstQuantizationInfo", {QuantizationInfo(0.5f, 4)}),
                                   make("DataLayout", {DataLayout::NHWC}),
                                   ActivationFunctionsQuantizedSmallDataset))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE_NEW(RunMixedDataLayout,
                           CpuDepthwiseConv2dQuantizedMixedDataLayoutFixture<uint8_t>,
                           framework::DatasetMode::PRECOMMIT,
                           combine(datasets::SmallDepthwiseConvolutionLayerDataset(),
                                   make("DepthMultiplier", {2}),
                                   make("DataType", DataType::QASYMM8),
                                   IgnoredQuantizationInfo,
                                   IgnoredQuantizationInfo,
                                   make("DataLayout", {DataLayout::NHWC}),
                                   NoActivation))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE(Dilation)
FIXTURE_DATA_TEST_CASE_NEW(RunSmall,
                           CpuDepthwiseConv2dQuantizedFixture<uint8_t>,
                           framework::DatasetMode::PRECOMMIT,
                           combine(datasets::SmallDepthwiseDilatedConvolutionLayerDataset(),
                                   depth_multipliers,
                                   make("DataType", DataType::QASYMM8),
                                   IgnoredQuantizationInfo,
                                   IgnoredQuantizationInfo,
                                   make("DataLayout", {DataLayout::NHWC}),
                                   NoActivation))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE_NEW(RunSmallWithActivation,
                           CpuDepthwiseConv2dQuantizedFixture<uint8_t>,
                           framework::DatasetMode::PRECOMMIT,
                           combine(datasets::SmallDepthwiseDilatedConvolutionLayerDataset(),
                                   depth_multipliers,
                                   make("DataType", DataType::QASYMM8),
                                   input_qinfo_dataset,
                                   make("DstQuantizationInfo", {QuantizationInfo(0.8f, 1)}),
                                   make("DataLayout", {DataLayout::NHWC}),
                                   ActivationFunctionsDataset))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE_NEW(RunLarge,
                           CpuDepthwiseConv2dQuantizedFixture<uint8_t>,
                           framework::DatasetMode::NIGHTLY,
                           combine(datasets::LargeDepthwiseDilatedConvolutionLayerDataset(),
                                   large_depth_multipliers,
                                   make("DataType", DataType::QASYMM8),
                                   IgnoredQuantizationInfo,
                                   IgnoredQuantizationInfo,
                                   make("DataLayout", {DataLayout::NHWC}),
                                   NoActivation))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // Dilation
TEST_SUITE_END() // Generic
TEST_SUITE(W3x3)
FIXTURE_DATA_TEST_CASE_NEW(RunSmall,
                           CpuDepthwiseConv2dQuantizedFixture<uint8_t>,
                           framework::DatasetMode::PRECOMMIT,
                           combine(datasets::SmallDepthwiseConvolutionLayerDataset3x3(),
                                   depth_multipliers,
                                   make("DataType", DataType::QASYMM8),
                                   IgnoredQuantizationInfo,
                                   IgnoredQuantizationInfo,
                                   make("DataLayout", {DataLayout::NHWC}),
                                   NoActivation))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE_NEW(RunSmallWithActivation,
                           CpuDepthwiseConv2dQuantizedFixture<uint8_t>,
                           framework::DatasetMode::PRECOMMIT,
                           combine(datasets::SmallDepthwiseConvolutionLayerDataset3x3(),
                                   depth_multipliers,
                                   make("DataType", DataType::QASYMM8),
                                   input_qinfo_dataset,
                                   make("DstQuantizationInfo", {QuantizationInfo(0.5f, 10)}),
                                   make("DataLayout", {DataLayout::NHWC}),
                                   ActivationFunctionsQuantizedSmallDataset))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE_NEW(RunLarge,
                           CpuDepthwiseConv2dQuantizedFixture<uint8_t>,
                           framework::DatasetMode::NIGHTLY,
                           combine(datasets::LargeDepthwiseConvolutionLayerDataset3x3(),
                                   large_depth_multipliers,
                                   make("DataType", DataType::QASYMM8),
                                   IgnoredQuantizationInfo,
                                   IgnoredQuantizationInfo,
                                   make("DataLayout", {DataLayout::NHWC}),
                                   NoActivation))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}

TEST_SUITE(Dilation)
FIXTURE_DATA_TEST_CASE_NEW(RunSmall,
                           CpuDepthwiseConv2dQuantizedFixture<uint8_t>,
                           framework::DatasetMode::PRECOMMIT,
                           combine(datasets::SmallDepthwiseDilatedConvolutionLayerDataset3x3(),
                                   depth_multipliers,
                                   make("DataType", DataType::QASYMM8),
                                   IgnoredQuantizationInfo,
                                   IgnoredQuantizationInfo,
                                   make("DataLayout", {DataLayout::NHWC}),
                                   NoActivation))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE_NEW(RunSmallWithActivation,
                           CpuDepthwiseConv2dQuantizedFixture<uint8_t>,
                           framework::DatasetMode::PRECOMMIT,
                           combine(datasets::SmallDepthwiseDilatedConvolutionLayerDataset3x3(),
                                   depth_multipliers,
                                   make("DataType", DataType::QASYMM8),
                                   input_qinfo_dataset,
                                   make("DstQuantizationInfo", {QuantizationInfo(0.7f, 10)}),
                                   make("DataLayout", {DataLayout::NHWC}),
                                   ActivationFunctionsQuantizedSmallDataset))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE_NEW(RunLarge,
                           CpuDepthwiseConv2dQuantizedFixture<uint8_t>,
                           framework::DatasetMode::NIGHTLY,
                           combine(datasets::LargeDepthwiseDilatedConvolutionLayerDataset3x3(),
                                   large_depth_multipliers,
                                   make("DataType", DataType::QASYMM8),
                                   IgnoredQuantizationInfo,
                                   IgnoredQuantizationInfo,
                                   make("DataLayout", {DataLayout::NHWC}),
                                   NoActivation))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // Dilation
TEST_SUITE_END() // W3x3

TEST_SUITE(Optimized)
FIXTURE_DATA_TEST_CASE_NEW(RunSmall3x3,
                           CpuDepthwiseConv2dQuantizedFixture<uint8_t>,
                           framework::DatasetMode::PRECOMMIT,
                           combine(datasets::SmallOptimizedDepthwiseConvolutionLayerDataset3x3(),
                                   make("DepthMultiplier", 1),
                                   make("DataType", DataType::QASYMM8),
                                   IgnoredQuantizationInfo,
                                   IgnoredQuantizationInfo,
                                   make("DataLayout", {DataLayout::NHWC}),
                                   NoActivation))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE_NEW(RunSmall3x3WithActivation,
                           CpuDepthwiseConv2dQuantizedFixture<uint8_t>,
                           framework::DatasetMode::PRECOMMIT,
                           combine(datasets::SmallOptimizedDepthwiseConvolutionLayerDataset3x3(),
                                   make("DepthMultiplier", 1),
                                   make("DataType", DataType::QASYMM8),
                                   input_qinfo_dataset,
                                   make("DstQuantizationInfo", {QuantizationInfo(0.5f, 10)}),
                                   make("DataLayout", {DataLayout::NHWC}),
                                   ActivationFunctionsQuantizedSmallDataset))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE_NEW(RunMixedDataLayout3x3,
                           CpuDepthwiseConv2dQuantizedMixedDataLayoutFixture<uint8_t>,
                           framework::DatasetMode::PRECOMMIT,
                           combine(datasets::SmallOptimizedDepthwiseConvolutionLayerDataset3x3(),
                                   make("DepthMultiplier", 1),
                                   make("DataType", DataType::QASYMM8),
                                   IgnoredQuantizationInfo,
                                   IgnoredQuantizationInfo,
                                   make("DataLayout", {DataLayout::NHWC}),
                                   NoActivation))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE_NEW(RunSmall5x5,
                           CpuDepthwiseConv2dQuantizedFixture<uint8_t>,
                           framework::DatasetMode::PRECOMMIT,
                           combine(datasets::SmallOptimizedDepthwiseConvolutionLayerDataset5x5(),
                                   make("DepthMultiplier", 1),
                                   make("DataType", DataType::QASYMM8),
                                   IgnoredQuantizationInfo,
                                   IgnoredQuantizationInfo,
                                   make("DataLayout", {DataLayout::NHWC}),
                                   NoActivation))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE_NEW(RunSmall5x5WithActivation,
                           CpuDepthwiseConv2dQuantizedFixture<uint8_t>,
                           framework::DatasetMode::PRECOMMIT,
                           combine(datasets::SmallOptimizedDepthwiseConvolutionLayerDataset5x5(),
                                   make("DepthMultiplier", 1),
                                   make("DataType", DataType::QASYMM8),
                                   input_qinfo_dataset,
                                   make("DstQuantizationInfo", {QuantizationInfo(0.5f, 10)}),
                                   make("DataLayout", {DataLayout::NHWC}),
                                   ActivationFunctionsQuantizedSmallDataset))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE_NEW(RunLarge3x3,
                           CpuDepthwiseConv2dQuantizedFixture<uint8_t>,
                           framework::DatasetMode::NIGHTLY,
                           combine(datasets::LargeOptimizedDepthwiseConvolutionLayerDataset3x3(),
                                   make("DepthMultiplier", 1),
                                   make("DataType", DataType::QASYMM8),
                                   IgnoredQuantizationInfo,
                                   IgnoredQuantizationInfo,
                                   make("DataLayout", {DataLayout::NHWC}),
                                   NoActivation))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // Optimized
TEST_SUITE_END() // QASYMM8

TEST_SUITE(QASYMM8_SIGNED)

FIXTURE_DATA_TEST_CASE_NEW(RunActivations,
                           CpuDepthwiseConv2dQuantizedFixture<int8_t>,
                           framework::DatasetMode::NIGHTLY,
                           combine(make("In", TensorShape(33U, 27U, 11U, 3U)),
                                   make("Weights", Size2D(3U, 4U)),
                                   make("Info", PadStrideInfo(1, 2, 0, 1)),
                                   make("Dilation", Size2D(2U, 2U)),
                                   make("DepthMultiplier", {5}),
                                   make("DataType", DataType::QASYMM8_SIGNED),
                                   make("SrcQuantizationInfo", {QuantizationInfo(0.3f, 10)}),
                                   make("DstQuantizationInfo", {QuantizationInfo(0.05f, 4)}),
                                   make("DataLayout", {DataLayout::NHWC}),
                                   ActivationFunctionsQuantizedDataset))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}

TEST_SUITE(Generic)
FIXTURE_DATA_TEST_CASE_NEW(RunSmall,
                           CpuDepthwiseConv2dQuantizedFixture<int8_t>,
                           framework::DatasetMode::PRECOMMIT,
                           combine(datasets::SmallDepthwiseConvolutionLayerDataset(),
                                   depth_multipliers,
                                   make("DataType", DataType::QASYMM8_SIGNED),
                                   IgnoredQuantizationInfo,
                                   IgnoredQuantizationInfo,
                                   make("DataLayout", {DataLayout::NHWC}),
                                   NoActivation))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8_signed);
}
FIXTURE_DATA_TEST_CASE_NEW(RunSmallWithActivation,
                           CpuDepthwiseConv2dQuantizedFixture<int8_t>,
                           framework::DatasetMode::PRECOMMIT,
                           combine(datasets::SmallDepthwiseConvolutionLayerDataset(),
                                   depth_multipliers,
                                   make("DataType", DataType::QASYMM8_SIGNED),
                                   input_qinfo_dataset,
                                   make("DstQuantizationInfo", {QuantizationInfo(0.5f, 4)}),
                                   make("DataLayout", {DataLayout::NHWC}),
                                   ActivationFunctionsQuantizedSmallDataset))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8_signed);
}
TEST_SUITE(Dilation)
FIXTURE_DATA_TEST_CASE_NEW(RunSmall,
                           CpuDepthwiseConv2dQuantizedFixture<int8_t>,
                           framework::DatasetMode::PRECOMMIT,
                           combine(datasets::SmallDepthwiseDilatedConvolutionLayerDataset(),
                                   depth_multipliers,
                                   make("DataType", DataType::QASYMM8_SIGNED),
                                   IgnoredQuantizationInfo,
                                   IgnoredQuantizationInfo,
                                   make("DataLayout", {DataLayout::NHWC}),
                                   NoActivation))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8_signed);
}
FIXTURE_DATA_TEST_CASE_NEW(RunSmallWithActivation,
                           CpuDepthwiseConv2dQuantizedFixture<int8_t>,
                           framework::DatasetMode::PRECOMMIT,
                           combine(datasets::SmallDepthwiseDilatedConvolutionLayerDataset(),
                                   depth_multipliers,
                                   make("DataType", DataType::QASYMM8_SIGNED),
                                   input_qinfo_dataset,
                                   make("DstQuantizationInfo", {QuantizationInfo(0.8f, 1)}),
                                   make("DataLayout", {DataLayout::NHWC}),
                                   ActivationFunctionsQuantizedSmallDataset))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8_signed);
}
FIXTURE_DATA_TEST_CASE_NEW(RunLarge,
                           CpuDepthwiseConv2dQuantizedFixture<int8_t>,
                           framework::DatasetMode::NIGHTLY,
                           combine(datasets::LargeDepthwiseDilatedConvolutionLayerDataset(),
                                   large_depth_multipliers,
                                   make("DataType", DataType::QASYMM8_SIGNED),
                                   IgnoredQuantizationInfo,
                                   IgnoredQuantizationInfo,
                                   make("DataLayout", {DataLayout::NHWC}),
                                   NoActivation))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8_signed);
}
TEST_SUITE_END() // Dilation
TEST_SUITE_END() // Generic

TEST_SUITE(W3x3)
FIXTURE_DATA_TEST_CASE_NEW(RunSmall,
                           CpuDepthwiseConv2dQuantizedFixture<int8_t>,
                           framework::DatasetMode::PRECOMMIT,
                           combine(datasets::SmallDepthwiseConvolutionLayerDataset3x3(),
                                   depth_multipliers,
                                   make("DataType", DataType::QASYMM8_SIGNED),
                                   IgnoredQuantizationInfo,
                                   IgnoredQuantizationInfo,
                                   make("DataLayout", {DataLayout::NHWC}),
                                   NoActivation))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8_signed);
}
FIXTURE_DATA_TEST_CASE_NEW(RunSmallWithActivation,
                           CpuDepthwiseConv2dQuantizedFixture<int8_t>,
                           framework::DatasetMode::PRECOMMIT,
                           combine(datasets::SmallDepthwiseConvolutionLayerDataset3x3(),
                                   depth_multipliers,
                                   make("DataType", DataType::QASYMM8_SIGNED),
                                   input_qinfo_dataset,
                                   make("DstQuantizationInfo", {QuantizationInfo(0.5f, 10)}),
                                   make("DataLayout", {DataLayout::NHWC}),
                                   ActivationFunctionsQuantizedSmallDataset))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8_signed);
}
FIXTURE_DATA_TEST_CASE_NEW(RunLarge,
                           CpuDepthwiseConv2dQuantizedFixture<int8_t>,
                           framework::DatasetMode::NIGHTLY,
                           combine(datasets::LargeDepthwiseConvolutionLayerDataset3x3(),
                                   large_depth_multipliers,
                                   make("DataType", DataType::QASYMM8_SIGNED),
                                   IgnoredQuantizationInfo,
                                   IgnoredQuantizationInfo,
                                   make("DataLayout", {DataLayout::NHWC}),
                                   NoActivation))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8_signed);
}

TEST_SUITE(Dilation)
FIXTURE_DATA_TEST_CASE_NEW(RunSmall,
                           CpuDepthwiseConv2dQuantizedFixture<int8_t>,
                           framework::DatasetMode::PRECOMMIT,
                           combine(datasets::SmallDepthwiseDilatedConvolutionLayerDataset3x3(),
                                   depth_multipliers,
                                   make("DataType", DataType::QASYMM8_SIGNED),
                                   IgnoredQuantizationInfo,
                                   IgnoredQuantizationInfo,
                                   make("DataLayout", {DataLayout::NHWC}),
                                   NoActivation))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8_signed);
}
FIXTURE_DATA_TEST_CASE_NEW(RunSmallWithActivation,
                           CpuDepthwiseConv2dQuantizedFixture<int8_t>,
                           framework::DatasetMode::PRECOMMIT,
                           combine(datasets::SmallDepthwiseDilatedConvolutionLayerDataset3x3(),
                                   depth_multipliers,
                                   make("DataType", DataType::QASYMM8_SIGNED),
                                   input_qinfo_dataset,
                                   make("DstQuantizationInfo", {QuantizationInfo(0.7f, 10)}),
                                   make("DataLayout", {DataLayout::NHWC}),
                                   ActivationFunctionsQuantizedSmallDataset))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8_signed);
}
FIXTURE_DATA_TEST_CASE_NEW(RunLarge,
                           CpuDepthwiseConv2dQuantizedFixture<int8_t>,
                           framework::DatasetMode::NIGHTLY,
                           combine(datasets::LargeDepthwiseDilatedConvolutionLayerDataset3x3(),
                                   large_depth_multipliers,
                                   make("DataType", DataType::QASYMM8_SIGNED),
                                   IgnoredQuantizationInfo,
                                   IgnoredQuantizationInfo,
                                   make("DataLayout", {DataLayout::NHWC}),
                                   NoActivation))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8_signed);
}
TEST_SUITE_END() // Dilation
TEST_SUITE_END() // W3x3

TEST_SUITE(Optimized)
FIXTURE_DATA_TEST_CASE_NEW(RunSmall3x3,
                           CpuDepthwiseConv2dQuantizedFixture<int8_t>,
                           framework::DatasetMode::PRECOMMIT,
                           combine(datasets::SmallOptimizedDepthwiseConvolutionLayerDataset3x3(),
                                   make("DepthMultiplier", 1),
                                   make("DataType", DataType::QASYMM8_SIGNED),
                                   IgnoredQuantizationInfo,
                                   IgnoredQuantizationInfo,
                                   make("DataLayout", {DataLayout::NHWC}),
                                   NoActivation))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8_signed);
}
FIXTURE_DATA_TEST_CASE_NEW(RunSmall3x3WithActivation,
                           CpuDepthwiseConv2dQuantizedFixture<int8_t>,
                           framework::DatasetMode::PRECOMMIT,
                           combine(datasets::SmallOptimizedDepthwiseConvolutionLayerDataset3x3(),
                                   make("DepthMultiplier", 1),
                                   make("DataType", DataType::QASYMM8_SIGNED),
                                   input_qinfo_dataset,
                                   make("DstQuantizationInfo", {QuantizationInfo(0.5f, 10)}),
                                   make("DataLayout", {DataLayout::NHWC}),
                                   ActivationFunctionsQuantizedSmallDataset))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8_signed);
}
FIXTURE_DATA_TEST_CASE_NEW(RunSmall5x5,
                           CpuDepthwiseConv2dQuantizedFixture<int8_t>,
                           framework::DatasetMode::PRECOMMIT,
                           combine(datasets::SmallOptimizedDepthwiseConvolutionLayerDataset5x5(),
                                   make("DepthMultiplier", 1),
                                   make("DataType", DataType::QASYMM8_SIGNED),
                                   IgnoredQuantizationInfo,
                                   IgnoredQuantizationInfo,
                                   make("DataLayout", {DataLayout::NHWC}),
                                   NoActivation))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8_signed);
}
FIXTURE_DATA_TEST_CASE_NEW(RunSmall5x5WithActivation,
                           CpuDepthwiseConv2dQuantizedFixture<int8_t>,
                           framework::DatasetMode::PRECOMMIT,
                           combine(datasets::SmallOptimizedDepthwiseConvolutionLayerDataset5x5(),
                                   make("DepthMultiplier", 1),
                                   make("DataType", DataType::QASYMM8_SIGNED),
                                   input_qinfo_dataset,
                                   make("DstQuantizationInfo", {QuantizationInfo(0.5f, 10)}),
                                   make("DataLayout", {DataLayout::NHWC}),
                                   ActivationFunctionsQuantizedSmallDataset))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8_signed);
}
FIXTURE_DATA_TEST_CASE_NEW(RunLarge3x3,
                           CpuDepthwiseConv2dQuantizedFixture<int8_t>,
                           framework::DatasetMode::NIGHTLY,
                           combine(datasets::LargeOptimizedDepthwiseConvolutionLayerDataset3x3(),
                                   make("DepthMultiplier", 1),
                                   make("DataType", DataType::QASYMM8_SIGNED),
                                   IgnoredQuantizationInfo,
                                   IgnoredQuantizationInfo,
                                   make("DataLayout", {DataLayout::NHWC}),
                                   NoActivation))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8_signed);
}
TEST_SUITE_END() // Optimized
TEST_SUITE_END() // QASYMM8_SIGNED

TEST_SUITE(QSYMM8_PER_CHANNEL)

FIXTURE_DATA_TEST_CASE_NEW(RunActivations,
                           CpuDepthwiseConv2dQuantizedSymmetricPerChannelFixture,
                           framework::DatasetMode::NIGHTLY,
                           combine(make("In", TensorShape(33U, 27U, 11U, 3U)),
                                   make("Weights", Size2D(3U, 4U)),
                                   make("Info", PadStrideInfo(1, 2, 0, 1)),
                                   make("Dilation", Size2D(2U, 2U)),
                                   make("DepthMultiplier", {5}),
                                   make("InputDataType", DataType::QASYMM8),
                                   make("WeightsDataType", DataType::QSYMM8_PER_CHANNEL),
                                   make("SrcQuantizationInfo", {QuantizationInfo(0.3f, 10)}),
                                   make("DstQuantizationInfo", {QuantizationInfo(0.05f, 4)}),
                                   make("DataLayout", {DataLayout::NHWC}),
                                   ActivationFunctionsQuantizedDataset))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}

TEST_SUITE(Generic)
FIXTURE_DATA_TEST_CASE_NEW(RunSmall,
                           CpuDepthwiseConv2dQuantizedSymmetricPerChannelFixture,
                           framework::DatasetMode::PRECOMMIT,
                           combine(datasets::SmallDepthwiseConvolutionLayerDataset(),
                                   depth_multipliers,
                                   make("InputDataType", DataType::QASYMM8),
                                   make("WeightsDataType", DataType::QSYMM8_PER_CHANNEL),
                                   input_qinfo_dataset,
                                   make("DstQuantizationInfo", {QuantizationInfo(0.5f, 4)}),
                                   make("DataLayout", {DataLayout::NHWC}),
                                   ActivationFunctionsDataset))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}

TEST_SUITE(Dilation)
FIXTURE_DATA_TEST_CASE_NEW(RunSmall,
                           CpuDepthwiseConv2dQuantizedSymmetricPerChannelFixture,
                           framework::DatasetMode::PRECOMMIT,
                           combine(datasets::SmallDepthwiseDilatedConvolutionLayerDataset(),
                                   depth_multipliers,
                                   make("InputDataType", DataType::QASYMM8),
                                   make("WeightsDataType", DataType::QSYMM8_PER_CHANNEL),
                                   input_qinfo_dataset,
                                   make("DstQuantizationInfo", {QuantizationInfo(0.5f, 4)}),
                                   make("DataLayout", {DataLayout::NHWC}),
                                   ActivationFunctionsDataset))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE_NEW(RunLarge,
                           CpuDepthwiseConv2dQuantizedSymmetricPerChannelFixture,
                           framework::DatasetMode::NIGHTLY,
                           combine(datasets::LargeDepthwiseDilatedConvolutionLayerDataset(),
                                   depth_multipliers,
                                   make("InputDataType", DataType::QASYMM8),
                                   make("WeightsDataType", DataType::QSYMM8_PER_CHANNEL),
                                   input_qinfo_dataset,
                                   make("DstQuantizationInfo", {QuantizationInfo(0.5f, 4)}),
                                   make("DataLayout", {DataLayout::NHWC}),
                                   make("ActivationInfo", {ActivationLayerInfo()})))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // Dilation
TEST_SUITE_END() // Generic

TEST_SUITE(Optimized)
FIXTURE_DATA_TEST_CASE_NEW(RunSmall3x3,
                           CpuDepthwiseConv2dQuantizedSymmetricPerChannelFixture,
                           framework::DatasetMode::PRECOMMIT,
                           combine(datasets::SmallOptimizedDepthwiseConvolutionLayerDataset3x3(),
                                   make("DepthMultiplier", 1),
                                   make("InputDataType", DataType::QASYMM8),
                                   make("WeightsDataType", DataType::QSYMM8_PER_CHANNEL),
                                   input_qinfo_dataset,
                                   make("DstQuantizationInfo", {QuantizationInfo(0.5f, 4)}),
                                   make("DataLayout", {DataLayout::NHWC}),
                                   ActivationFunctionsDataset))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE_NEW(RunLarge3x3,
                           CpuDepthwiseConv2dQuantizedSymmetricPerChannelFixture,
                           framework::DatasetMode::NIGHTLY,
                           combine(datasets::LargeOptimizedDepthwiseConvolutionLayerDataset3x3(),
                                   make("DepthMultiplier", 1),
                                   make("InputDataType", DataType::QASYMM8),
                                   make("WeightsDataType", DataType::QSYMM8_PER_CHANNEL),
                                   input_qinfo_dataset,
                                   make("DstQuantizationInfo", {QuantizationInfo(0.5f, 4)}),
                                   make("DataLayout", {DataLayout::NHWC}),
                                   make("ActivationInfo", {ActivationLayerInfo()})))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // Optimized
TEST_SUITE_END() // QSYMM8_PER_CHANNEL
TEST_SUITE_END() // Quantized

TEST_SUITE_END() // CpuDepthwiseConv2d
TEST_SUITE_END() // Operators
TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute

#endif // __aarch64__
