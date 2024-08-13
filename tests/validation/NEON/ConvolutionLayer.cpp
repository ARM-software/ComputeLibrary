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
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/functions/NEConvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NEGEMMConv2d.h"
#include "arm_compute/runtime/NEON/functions/NEGEMMConvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NEWinogradConvolutionLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"

#include "src/core/CPP/Validate.h"
#include "src/core/helpers/MemoryHelpers.h"
#include "src/cpu/operators/CpuGemmConv2d.h"
#include "src/cpu/operators/CpuGemmDirectConv2d.h"
#include "src/cpu/operators/CpuWinogradConv2d.h"

#include "tests/NEON/Accessor.h"
#include "tests/datasets/LargeConvolutionLayerDataset.h"
#include "tests/datasets/SmallConvolutionLayerDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/ConvolutionLayerFixture.h"
#include "tests/validation/fixtures/WinogradConvolutionLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
using framework::dataset::make;

namespace detail
{
template <>
void configure_conv_function<NEGEMMConv2d, Tensor>(NEGEMMConv2d &func,
                                                   Tensor *src, const Tensor *weights, const Tensor *bias, Tensor *dst,
                                                   const PadStrideInfo &info, const WeightsInfo &weights_info,
                                                   const Size2D &dilation, const ActivationLayerInfo &act_info, unsigned int num_groups)
{
    ARM_COMPUTE_UNUSED(weights_info);

    Conv2dInfo conv_info(info, dilation, act_info, false, num_groups);
    func.configure(src, weights, bias, dst, conv_info);
}
} // namespace detail
namespace
{
const RelativeTolerance<float> rel_tolerance_f32(0.01f);              /**< Relative tolerance for FP32 types */
const RelativeTolerance<float> rel_tolerance_winograd_3x3_f32(0.05f); /**< Relative tolerance for FP32 types */
const AbsoluteTolerance<float> abs_tolerance_f32(0.002f);             /**< Absolute tolerance for FP32 types */
const AbsoluteTolerance<float> abs_tolerance_1xN_f32(0.0041f);        /**< Absolute tolerance for FP32 types */

#ifdef ARM_COMPUTE_ENABLE_FP16
const AbsoluteTolerance<half> tolerance_convolution_layer_f16(half(0.4f));
constexpr float               tolerance_num_f16 = 0.15f;
#endif /* ARM_COMPUTE_ENABLE_FP16 */

#ifdef ARM_COMPUTE_ENABLE_FP16
const RelativeTolerance<half_float::half> rel_tolerance_f16(half_float::half(0.2f)); /**< Relative tolerance value for FP16 types */
const AbsoluteTolerance<float>            abs_tolerance_f16(0.2f);                   /**< Absolute tolerance for FP16 types */
constexpr float                           tolerance_num = 0.07f;                     /**< Tolerance number for the FP16 implementation */
#endif                                                                               /* ARM_COMPUTE_ENABLE_FP16 */

#ifdef ARM_COMPUTE_ENABLE_SME
// TODO(COMPMID-6011): SME kernels and the reference model use different rounding mode.
// Temporarily increase the tolerance for quantized data.
constexpr AbsoluteTolerance<float> tolerance_qasymm8(1.0); /**< Tolerance value for comparing reference's output against implementation's output for quantized data types */
#else                                                      // ARM_COMPUTE_ENABLE_SME
constexpr AbsoluteTolerance<float> tolerance_qasymm8(0.0); /**< Tolerance value for comparing reference's output against implementation's output for quantized data types */
#endif                                                     // ARM_COMPUTE_ENABLE_SME

/** CNN data types */
const auto CNNDataTypes = make("DataType",
{
#ifdef ARM_COMPUTE_ENABLE_FP16
    DataType::F16,
#endif /* ARM_COMPUTE_ENABLE_FP16 */
    DataType::F32,
    DataType::QASYMM8,
});
const auto ActivationFunctionsDataset = make("ActivationInfo",
{
    ActivationLayerInfo(),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 0.5f)
});

const auto NoActivation = make("ActivationInfo",
{
    ActivationLayerInfo(),
});

const auto ActivationFunctionsDatasetNightly = make("ActivationInfo",
{
    ActivationLayerInfo(),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
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

const auto QuantizationData = make("QuantizationInfo",
{
    QuantizationInfo(0.5f, 10),
    QuantizationInfo(0.3f, 3),
    QuantizationInfo(1.f, 10),
    QuantizationInfo(1.1f, 10),
});
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(ConvolutionLayer)
DATA_TEST_CASE(SupportedTypes, framework::DatasetMode::ALL, zip(
                make("DataType", {
                    DataType::F32,
                    DataType::QASYMM8,
                    DataType::QASYMM8,
                    DataType::QASYMM8_SIGNED
                }),
                make("WeightsDataType", {
                    DataType::F32,
                    DataType::QASYMM8,
                    DataType::QASYMM8_SIGNED,
                    DataType::QASYMM8
                }),
                make("Expected",
                {
                    true,
                    true,
                    true,
                    false
                })),
data_type_const, weights_data_type_const, expected_const)
{
    TensorInfo input_info   = TensorInfo(TensorShape(3U, 3U, 1U), 1, data_type_const);
    TensorInfo weights_info = TensorInfo(TensorShape(2U, 2U, 1U, 1U), 1, weights_data_type_const);
    TensorInfo output_info  = TensorInfo(TensorShape(2U, 2U, 1U), 1, data_type_const);

    input_info.set_quantization_info(arm_compute::QuantizationInfo(1, 0));
    weights_info.set_quantization_info(arm_compute::QuantizationInfo(1, 0));
    output_info.set_quantization_info(arm_compute::QuantizationInfo(1, 0));

    Status status = NEConvolutionLayer::validate(
                        &input_info,
                        &weights_info,
                        nullptr,
                        &output_info,
                        PadStrideInfo());

    ARM_COMPUTE_EXPECT(bool(status) == expected_const, framework::LogLevel::ERRORS);
}

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(ValidateConvolutionMethod, framework::DatasetMode::ALL, zip(zip(zip(zip(zip(
                                          make("InputInfo", { TensorInfo(TensorShape(18U, 18U, 32U), 1, DataType::F32),
                                                                                  TensorInfo(TensorShape(23U, 27U, 32U, 4U), 1, DataType::F32),
                                                                                  TensorInfo(TensorShape(3U, 3U, 2U, 1U), 1, DataType::F32),
                                                                                  TensorInfo(TensorShape(33U, 27U, 7U, 4U), 1, DataType::F32)
                                          }),
                                          make("WeightsInfo", { TensorInfo(TensorShape(3U, 3U, 32U, 21U), 1, DataType::F32),
                                                                                    TensorInfo(TensorShape(5U, 5U, 32U, 21U), 1, DataType::F32),
                                                                                    TensorInfo(TensorShape(3U, 3U, 5U, 21U), 1, DataType::F32),
                                                                                    TensorInfo(TensorShape(5U, 5U, 7U, 16U), 1, DataType::F16)
                                          })),
                                          make("OutputInfo", { TensorInfo(TensorShape(16U, 16U, 21U), 1, DataType::F32),
                                                                                   TensorInfo(TensorShape(19U, 23U, 21U, 4U), 1, DataType::F32),
                                                                                   TensorInfo(TensorShape(11U, 25U, 21U), 1, DataType::F32),
                                                                                   TensorInfo(TensorShape(11U, 12U, 16U, 4U), 1, DataType::F32)
                                          })),
                                          make("ConvInfo", { PadStrideInfo(1, 1, 0, 0),
                                                                                 PadStrideInfo(1, 1, 0, 0),
                                                                                 PadStrideInfo(2, 1, 0, 0),
                                                                                 PadStrideInfo(3, 2, 1, 0)
                                          })),
                                          make("FastMath", { true,
                                                                                 true,
                                                                                 false,
                                                                                 false
                                          })),
                                                                           make("Expected", { ConvolutionMethod::WINOGRAD, ConvolutionMethod::WINOGRAD, ConvolutionMethod::GEMM, ConvolutionMethod::GEMM })),
               input_info, weights_info, output_info, conv_info, fast_math, expected)
{
    ConvolutionMethod is_valid = NEConvolutionLayer::get_convolution_method(&input_info.clone()->set_is_resizable(true),
                                                                            &weights_info.clone()->set_is_resizable(true),
                                                                            &output_info.clone()->set_is_resizable(true), conv_info, WeightsInfo(), Size2D(1U, 1U), ActivationLayerInfo(), fast_math);
    ARM_COMPUTE_EXPECT(is_valid == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*
TEST_SUITE_END() // ConvolutionLayer

/*
    Testing Strategy of Neon Winograd:
        - There is no need to thoroughly test nchw cases because winograd kernels accept
          nhwc and the tensors are permuted before and after if they're nchw.
        - Except relu and bounded relu, testing activations for a single input
          combination is enough because activation is not fused into winograd and called
          separately.
*/
TEST_SUITE(WinogradLayer)
template <typename T>
using NEWinogradConvolutionLayerFixture = WinogradConvolutionLayerFastMathValidationFixture<Tensor, Accessor, NEWinogradConvolutionLayer, T>;
template <typename T>
using NEWinogradConvolutionLayerMixedDataLayoutFixture = WinogradConvolutionLayerFastMathValidationFixture<Tensor, Accessor, NEWinogradConvolutionLayer, T, T, true, true>;

template <typename T>
using NEWinogradConvolutionLayerNoBiasFixture = WinogradConvolutionLayerFastMathValidationFixture<Tensor, Accessor, NEWinogradConvolutionLayer, T, T, false>;

/** Test case for memory injection in @ref cpu::CpuWinogradConv2d.
 *
 * Configure the operator once and inject memory at run-time in multiple executions.
 *
 * Checks performed in order:
 * - Both runs compute the same output
 */
TEST_CASE(MemoryInjection, framework::DatasetMode::ALL)
{
    auto                winograd = std::make_unique<cpu::CpuWinogradConv2d>();
    const auto          src_info = TensorInfo(TensorShape(8U, 8U, 32U), 1, DataType::F32);
    const auto          w_info   = TensorInfo(TensorShape(1U), 1, DataType::F32);
    const auto          b_info   = TensorInfo(TensorShape(1U, 3U, 32U, 1U), 1, DataType::F32);
    auto                dst_info = TensorInfo(TensorShape(8U, 6U, 1U), 1, DataType::F32);
    const PadStrideInfo pad_info{};

    winograd->configure(&src_info, &b_info, &w_info, &dst_info, pad_info);

    // telhs are newly created every call of this lambda function
    auto a = create_tensor<Tensor>(src_info);
    auto b = create_tensor<Tensor>(b_info);
    auto c = create_tensor<Tensor>(w_info);
    a.allocator()->allocate();
    b.allocator()->allocate();
    c.allocator()->allocate();

    ITensorPack run_pack{ { TensorType::ACL_SRC_0, &a }, { TensorType::ACL_SRC_1, &b }, { TensorType::ACL_SRC_2, &c } };
    ITensorPack prep_pack{ { TensorType::ACL_SRC_1, &b }, { TensorType::ACL_SRC_2, &c } };

    auto mg       = MemoryGroup{};
    auto ws       = manage_workspace<Tensor>(winograd->workspace(), mg, run_pack, prep_pack);
    auto run_conv = [&]() -> Tensor
    {
        auto dst = create_tensor<Tensor>(dst_info);
        dst.allocator()->allocate();

        run_pack.add_tensor(TensorType::ACL_DST, &dst);
        library->fill_tensor_value(Accessor(a), 1.f);
        library->fill_tensor_value(Accessor(b), 2.f);
        library->fill_tensor_value(Accessor(c), 3.f);

        // This operator is configured once and captured by this lambda.
        winograd->prepare(prep_pack);
        winograd->run(run_pack);
        return dst;
    };

    auto result_0 = run_conv();
    auto result_1 = run_conv();

    for(size_t i = 0; i < result_0.info()->tensor_shape().total_size(); ++i)
    {
        ARM_COMPUTE_EXPECT(reinterpret_cast<float *>(result_0.buffer())[i] == reinterpret_cast<float *>(result_1.buffer())[i], framework::LogLevel::ERRORS);
    }
}

/** Test case for memory injection in @ref NEWinogradConvolutionLayer.
 *
 * Make sure @ref NEWinogradConvolutionLayer still works through injecting the memory at configure time using the old API.
 *
 * Checks performed in order:
 * - Both runs compute the same output
 */
TEST_CASE(MultipleExecutionWithConfigure, framework::DatasetMode::ALL)
{
    auto                gemm     = std::make_unique<NEWinogradConvolutionLayer>();
    const auto          src_info = TensorInfo(TensorShape(8U, 8U, 32U), 1, DataType::F32);
    const auto          w_info   = TensorInfo(TensorShape(1U), 1, DataType::F32);
    const auto          b_info   = TensorInfo(TensorShape(1U, 3U, 32U, 1U), 1, DataType::F32);
    auto                dst_info = TensorInfo(TensorShape(8U, 6U, 1U), 1, DataType::F32);
    const PadStrideInfo pad_info{};

    auto run_conv = [&]()
    {
        auto src = create_tensor<Tensor>(src_info);
        auto w   = create_tensor<Tensor>(w_info);
        auto b   = create_tensor<Tensor>(b_info);
        auto dst = create_tensor<Tensor>(dst_info);

        gemm->configure(&src, &b, &w, &dst, pad_info);

        src.allocator()->allocate();
        b.allocator()->allocate();
        w.allocator()->allocate();
        dst.allocator()->allocate();

        library->fill_tensor_value(Accessor(src), 1.f);
        library->fill_tensor_value(Accessor(b), 2.f);
        library->fill_tensor_value(Accessor(w), 3.f);
        gemm->run();
        return dst;
    };

    auto result_0 = run_conv();
    auto result_1 = run_conv();

    for(size_t i = 0; i < result_0.info()->tensor_shape().total_size(); ++i)
    {
        ARM_COMPUTE_EXPECT(reinterpret_cast<float *>(result_0.buffer())[i] == reinterpret_cast<float *>(result_1.buffer())[i], framework::LogLevel::ERRORS);
    }
}

DATA_TEST_CASE(SupportedKernels, framework::DatasetMode::ALL, zip(
                   make("WeightsInfo",
{
    // Shapes are always in NCHW format. When layout is NHWC, the shape is permuted

    // Fp32, NCHW/NHWC (layout does not matter as it's )
    // 3x1, 1x3, 3x3 --> all TRUE
    TensorInfo(TensorShape(3U, 3U, 2U, 8U), 1, DataType::F32, DataLayout::NHWC),
    TensorInfo(TensorShape(1U, 3U, 2U, 8U), 1, DataType::F32, DataLayout::NHWC),
    TensorInfo(TensorShape(3U, 1U, 2U, 8U), 1, DataType::F32, DataLayout::NCHW),

    // 5x1, 1x5, 5x5 --> all TRUE
    TensorInfo(TensorShape(5U, 5U, 2U, 8U), 1, DataType::F32, DataLayout::NCHW),
    TensorInfo(TensorShape(1U, 5U, 2U, 8U), 1, DataType::F32, DataLayout::NHWC),
    TensorInfo(TensorShape(5U, 1U, 2U, 8U), 1, DataType::F32, DataLayout::NCHW),

    // 7x1, 1x7, 7x7
    //  --> all FALSE
    TensorInfo(TensorShape(7U, 7U, 2U, 8U), 1, DataType::F32, DataLayout::NCHW),
    TensorInfo(TensorShape(1U, 7U, 2U, 8U), 1, DataType::F32, DataLayout::NHWC),
    TensorInfo(TensorShape(7U, 1U, 2U, 8U), 1, DataType::F32, DataLayout::NHWC),

    // unsupported kernel sizes
    TensorInfo(TensorShape(2U, 2U, 2U, 8U), 1, DataType::F32, DataLayout::NHWC),
    TensorInfo(TensorShape(5U, 2U, 2U, 8U), 1, DataType::F32, DataLayout::NHWC),
    TensorInfo(TensorShape(3U, 6U, 2U, 8U), 1, DataType::F32, DataLayout::NCHW),

    // Fp16
    TensorInfo(TensorShape(3U, 3U, 2U, 8U), 1, DataType::F16, DataLayout::NHWC),
    TensorInfo(TensorShape(1U, 3U, 2U, 8U), 1, DataType::F16, DataLayout::NHWC),
    TensorInfo(TensorShape(3U, 1U, 2U, 8U), 1, DataType::F16, DataLayout::NCHW),

    // 5x1, 1x5, 5x5 --> all TRUE
    TensorInfo(TensorShape(5U, 5U, 2U, 8U), 1, DataType::F16, DataLayout::NCHW),
    TensorInfo(TensorShape(1U, 5U, 2U, 8U), 1, DataType::F16, DataLayout::NHWC),
    TensorInfo(TensorShape(5U, 1U, 2U, 8U), 1, DataType::F16, DataLayout::NCHW),

    // 7x1, 1x7, 7x7
    //  --> all FALSE
    TensorInfo(TensorShape(7U, 7U, 2U, 8U), 1, DataType::F16, DataLayout::NCHW),
    TensorInfo(TensorShape(1U, 7U, 2U, 8U), 1, DataType::F16, DataLayout::NHWC),
    TensorInfo(TensorShape(7U, 1U, 2U, 8U), 1, DataType::F16, DataLayout::NHWC),

    // unsupported kernel sizes
    TensorInfo(TensorShape(2U, 2U, 2U, 8U), 1, DataType::F16, DataLayout::NHWC),
    TensorInfo(TensorShape(5U, 2U, 2U, 8U), 1, DataType::F16, DataLayout::NHWC),
    TensorInfo(TensorShape(3U, 6U, 2U, 8U), 1, DataType::F16, DataLayout::NCHW),

}),
make("Expected",
{
    // fp32
    true, true, true,    // 3x3, 1x3, 3x1
    true, true, true,    // 5x5, 1x5, 5x1
    false, true, true,   // 7x7, 1x7, 7x1
    false, false, false, // random unsupported kernels

    // fp16
    true, false, false,  // 3x3, 1x3, 3x1
    false, false, false, // 5x5, 1x5, 5x1
    false, false, false, // 7x7, 1x7, 7x1
    false, false, false, // random unsupported kernels
})),
weights_info_const, expected_const)
{
    DataType   data_type   = weights_info_const.data_type();
    DataLayout data_layout = weights_info_const.data_layout();

    TensorInfo input_info   = TensorInfo(TensorShape(17U, 31U, 2U), 1, data_type);
    TensorInfo bias_info    = TensorInfo(TensorShape(8U), 1, data_type);
    TensorInfo weights_info = weights_info_const;

    if(data_layout == DataLayout::NHWC)
    {
        // Convert to NHWC
        PermutationVector perm = PermutationVector(2U, 0U, 1U);

        TensorShape input_shape   = input_info.tensor_shape();
        TensorShape weights_shape = weights_info.tensor_shape();
        permute(input_shape, perm);
        permute(weights_shape, perm);

        input_info.set_tensor_shape(input_shape);
        weights_info.set_tensor_shape(weights_shape);

        input_info.set_data_layout(data_layout);
        weights_info.set_data_layout(data_layout);
        bias_info.set_data_layout(data_layout);
    }

    PadStrideInfo conv_info(1, 1, 0, 0);

    TensorShape output_shape = compute_deep_convolution_shape(input_info, weights_info, conv_info);
    TensorInfo  output_info  = TensorInfo(output_shape, 1, data_type, data_layout);

    Status status = NEWinogradConvolutionLayer::validate(
                        &input_info,
                        &weights_info,
                        &bias_info,
                        &output_info,
                        conv_info,
                        ActivationLayerInfo(),
                        true /* fast math */);

    Status fp16_supported = ::arm_compute::error_on_unsupported_cpu_fp16("N/A", "N/A", 0, &input_info);
    bool   expected       = expected_const && static_cast<bool>(fp16_supported);

    ARM_COMPUTE_EXPECT(bool(status) == expected, framework::LogLevel::ERRORS);
}

TEST_SUITE(FP32)

TEST_SUITE(Conv1x3)
FIXTURE_DATA_TEST_CASE(RunSmall, NEWinogradConvolutionLayerFixture<float>, framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallWinogradConvolutionLayer1x3Dataset(),
                               make("DataType", { DataType::F32 }),
                               ActivationFunctionsDataset,
                               make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference, abs_tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunMixedDataLayout, NEWinogradConvolutionLayerMixedDataLayoutFixture<float>, framework::DatasetMode::PRECOMMIT,
                       combine(
                           make("Input", TensorShape(8U, 8U, 32U)),
                           make("Weight", TensorShape(1U, 3U, 32U, 1U)),
                           make("Bias", TensorShape(1U)),
                           make("Output", TensorShape(8U, 6U, 1U)),
                           make("PadStrideInfo", PadStrideInfo(1, 1, 0, 0)),
                           make("Dilation", Size2D(1U, 1U)),
                           make("DataType", { DataType::F32 }),
                           ActivationFunctionsDataset,
                           make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference, abs_tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEWinogradConvolutionLayerFixture<float>, framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeWinogradConvolutionLayer1x3Dataset(),
                               make("DataType", { DataType::F32 }),
                               make("ActivationInfo", { ActivationLayerInfo() }),
                               make("DataLayout", { DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference, abs_tolerance_1xN_f32);
}

TEST_SUITE_END() // Conv1x3

TEST_SUITE(Conv3x1)
FIXTURE_DATA_TEST_CASE(RunSmall, NEWinogradConvolutionLayerFixture<float>, framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallWinogradConvolutionLayer3x1Dataset(),
                               make("DataType", { DataType::F32 }),
                               ActivationFunctionsDataset,
                               make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference, abs_tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEWinogradConvolutionLayerFixture<float>, framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeWinogradConvolutionLayer3x1Dataset(),
                               make("DataType", { DataType::F32 }),
                               make("ActivationInfo", { ActivationLayerInfo() }),
                               make("DataLayout", { DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference, abs_tolerance_1xN_f32);
}

TEST_SUITE_END() // Conv3x1

TEST_SUITE(Conv1x5)
FIXTURE_DATA_TEST_CASE(RunSmall, NEWinogradConvolutionLayerFixture<float>, framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallWinogradConvolutionLayer1x5Dataset(),
                               make("DataType", { DataType::F32 }),
                               ActivationFunctionsDataset,
                               make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference, abs_tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEWinogradConvolutionLayerFixture<float>, framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeWinogradConvolutionLayer1x5Dataset(),
                               make("DataType", { DataType::F32 }),
                               make("ActivationInfo", { ActivationLayerInfo() }),
                               make("DataLayout", { DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference, abs_tolerance_1xN_f32);
}

TEST_SUITE_END() // Conv1x5

TEST_SUITE(Conv5x1)
FIXTURE_DATA_TEST_CASE(RunSmall, NEWinogradConvolutionLayerFixture<float>, framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallWinogradConvolutionLayer5x1Dataset(),
                               make("DataType", { DataType::F32 }),
                               ActivationFunctionsDataset,
                               make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference, abs_tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEWinogradConvolutionLayerFixture<float>, framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeWinogradConvolutionLayer5x1Dataset(),
                               make("DataType", { DataType::F32 }),
                               make("ActivationInfo", { ActivationLayerInfo() }),
                               make("DataLayout", { DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference, abs_tolerance_1xN_f32);
}

TEST_SUITE_END() // Conv5x1

TEST_SUITE(Conv7x1)
FIXTURE_DATA_TEST_CASE(RunSmall, NEWinogradConvolutionLayerFixture<float>, framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallWinogradConvolutionLayer7x1Dataset(),
                               make("DataType", { DataType::F32 }),
                               ActivationFunctionsDataset,
                               make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference, abs_tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEWinogradConvolutionLayerFixture<float>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(datasets::LargeWinogradConvolutionLayer7x1Dataset(),
                                               make("DataType", { DataType::F32 })),
                                       make("ActivationInfo", { ActivationLayerInfo() })),
                               make("DataLayout", { DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference, abs_tolerance_1xN_f32);
}
TEST_SUITE_END() // Conv7x1

TEST_SUITE(Conv1x7)
FIXTURE_DATA_TEST_CASE(RunSmall, NEWinogradConvolutionLayerFixture<float>, framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallWinogradConvolutionLayer1x7Dataset(),
                               make("DataType", { DataType::F32 }),
                               ActivationFunctionsDataset,
                               make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference, abs_tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEWinogradConvolutionLayerFixture<float>, framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeWinogradConvolutionLayer7x1Dataset(),
                               make("DataType", { DataType::F32 }),
                               make("ActivationInfo", { ActivationLayerInfo() }),
                               make("DataLayout", { DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference, abs_tolerance_1xN_f32);
}
TEST_SUITE_END() // Conv1x7

TEST_SUITE(Conv3x3)
FIXTURE_DATA_TEST_CASE(RunSmall, NEWinogradConvolutionLayerFixture<float>, framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallWinogradConvolutionLayer3x3Dataset(),
                               make("DataType", { DataType::F32 }),
                               ActivationFunctionsDataset,
                               make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))

{
    // Validate output
    validate(Accessor(_target), _reference, abs_tolerance_f32);
}

/// It's enough to run the activations for a single weight/input combination and data type because
/// activation function is called on top of the winograd output as a separate operator
/// TODO(COMPMID-6573): Enable after COMPMID-6573 is resolved
FIXTURE_DATA_TEST_CASE(RunActivations, NEWinogradConvolutionLayerFixture<float>, framework::DatasetMode::DISABLED,
                       combine(
                           make("Input", TensorShape(3U, 3U, 32U)),
                           make("Weight", TensorShape(3U, 3U, 32U, 4U)),
                           make("Bias", TensorShape(4U)),
                           make("Output", TensorShape(1U, 1U, 4U)),
                           make("PadStrideInfo", PadStrideInfo(1, 1, 0, 0)),
                           make("Dilation", Size2D(1U, 1U)),
                           make("DataType", { DataType::F32 }),
                           ActivationFunctionsDatasetNightly,
                           make("DataLayout", { DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference, abs_tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEWinogradConvolutionLayerFixture<float>, framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeWinogradConvolutionLayer3x3Dataset(),
                               make("DataType", { DataType::F32 }),
                               make("ActivationInfo", { ActivationLayerInfo() }),
                               make("DataLayout", { DataLayout::NHWC })))

{
    // Validate output
    // floating point arithmetic the Winograd results will not be exactly the same as direct convolution, especially for big shapes
    validate(Accessor(_target), _reference, rel_tolerance_winograd_3x3_f32, 0.f, float(abs_tolerance_f32));
}
TEST_SUITE_END() // Conv3x3

TEST_SUITE(Conv5x5)
FIXTURE_DATA_TEST_CASE(RunSmall, NEWinogradConvolutionLayerFixture<float>, framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallWinogradConvolutionLayer5x5Dataset(),
                               make("DataType", { DataType::F32 }),
                               ActivationFunctionsDataset,
                               make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))

{
    // Validate output
    validate(Accessor(_target), _reference, abs_tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEWinogradConvolutionLayerFixture<float>, framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeWinogradConvolutionLayer5x5Dataset(),
                               make("DataType", { DataType::F32 }),
                               make("ActivationInfo", { ActivationLayerInfo() }),
                               make("DataLayout", { DataLayout::NHWC })))

{
    // Validate output
    validate(Accessor(_target), _reference, abs_tolerance_f32);
}

TEST_SUITE_END() // Conv5x5

FIXTURE_DATA_TEST_CASE(RunSmallNoBias, NEWinogradConvolutionLayerNoBiasFixture<float>, framework::DatasetMode::PRECOMMIT,
                       combine(framework::dataset::concat(
                                   datasets::SmallWinogradConvolutionLayer3x3Dataset(),
                                   datasets::SmallWinogradConvolutionLayer5x5Dataset()),
                               make("DataType", { DataType::F32 }),
                               ActivationFunctionsDataset,
                               make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference, abs_tolerance_f32);
}

TEST_SUITE_END() // FP32

#ifdef ARM_COMPUTE_ENABLE_FP16
TEST_SUITE(FP16)
using NEWinogradConvolutionLayerFastMathFixture16 = WinogradConvolutionLayerFastMathValidationFixture<Tensor, Accessor, NEWinogradConvolutionLayer, half, float>;

DATA_TEST_CASE(ValidateConvolutionMethod, framework::DatasetMode::ALL, zip(
                   make("InputInfo", { TensorInfo(TensorShape(18U, 18U, 32U), 1, DataType::F16),
                                       TensorInfo(TensorShape(18U, 18U, 32U), 1, DataType::F16)
                                     }),
                   make("WeightsInfo", { TensorInfo(TensorShape(3U, 3U, 32U, 21U), 1, DataType::F16),
                                         TensorInfo(TensorShape(3U, 3U, 32U, 21U), 1, DataType::F16)
                                       }),
                   make("OutputInfo", { TensorInfo(TensorShape(16U, 16U, 21U), 1, DataType::F32),
                                        TensorInfo(TensorShape(16U, 16U, 21U), 1, DataType::F16)
                                      }),
                   make("ConvInfo", { PadStrideInfo(1, 1, 0, 0),
                                      PadStrideInfo(1, 1, 0, 0)
                                    }),
                   make("FastMath",
{
    false, // case fp16 and fast_math False then disable Winograd
    true   // case fp16 and fast_math True then enable Winograd
}),
make("Expected", { ConvolutionMethod::GEMM, ConvolutionMethod::WINOGRAD })),
input_info, weights_info, output_info, conv_info, fast_math, expected)
{
    if(CPUInfo::get().has_fp16())
    {
        ConvolutionMethod is_valid = NEConvolutionLayer::get_convolution_method(&input_info.clone()->set_is_resizable(true),
                                                                                &weights_info.clone()->set_is_resizable(true),
                                                                                &output_info.clone()->set_is_resizable(true), conv_info, WeightsInfo(), Size2D(1U, 1U), ActivationLayerInfo(), fast_math);
        ARM_COMPUTE_EXPECT(is_valid == expected, framework::LogLevel::ERRORS);
    }
    else
    {
        ARM_COMPUTE_TEST_INFO("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_INFO();
    }
}

TEST_SUITE(Conv3x3)
FIXTURE_DATA_TEST_CASE(RunSmall, NEWinogradConvolutionLayerFastMathFixture16, framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallWinogradConvolutionLayer3x3Dataset(),
                               make("DataType", { DataType::F16 }),
                               ActivationFunctionsDataset,
                               make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))

{
    if(CPUInfo::get().has_fp16())
    {
        // Validate output
        validate(Accessor(_target), _reference, tolerance_convolution_layer_f16, tolerance_num_f16);
    }
    else
    {
        ARM_COMPUTE_TEST_INFO("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_INFO();
    }
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEWinogradConvolutionLayerFastMathFixture16, framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeWinogradConvolutionLayer3x3Dataset(),
                               make("DataType", { DataType::F16 }),
                               make("ActivationInfo", { ActivationLayerInfo() }),
                               make("DataLayout", { DataLayout::NHWC })))

{
    if(CPUInfo::get().has_fp16())
    {
        // Validate output
        validate(Accessor(_target), _reference, tolerance_convolution_layer_f16, tolerance_num_f16);
    }
    else
    {
        ARM_COMPUTE_TEST_INFO("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_INFO();
    }
}
TEST_SUITE_END() // Conv3x3
TEST_SUITE_END() // FP16
#endif           /* ARM_COMPUTE_ENABLE_FP16 */
TEST_SUITE_END() // WinogradLayer

#ifdef ARM_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS
TEST_SUITE(FIXED_FORMAT_KERNELS)
TEST_SUITE(VariableWeightUtils)

// UC2_1_* tests: the user requests a specific fixed format, but there is no kernel that supports it.

template <typename ConvolutionClass>
using HasOptImplFixtureNoFastMath = HasOptImplFixture<ConvolutionClass, /*enable_fast_math*/ false>;

template <typename ConvolutionClass>
using HasOptImplFixtureFastMath = HasOptImplFixture<ConvolutionClass, /*enable_fast_math*/ true>;

// UC2_1

FIXTURE_DATA_TEST_CASE(UC2_1_CpuGemmConv2d, HasOptImplFixtureNoFastMath<cpu::CpuGemmConv2d>, framework::DatasetMode::ALL,
                       combine(framework::dataset::make("DataType", { DataType::F32 }),
                               framework::dataset::make("QueryWeightFormat", { arm_compute::WeightFormat::OHWIo2 })))
{
    ARM_COMPUTE_EXPECT(!_kernel_found, framework::LogLevel::ERRORS);
}
FIXTURE_DATA_TEST_CASE(UC2_1_NEGEMMConvolutionLayer, HasOptImplFixtureNoFastMath<NEGEMMConvolutionLayer>, framework::DatasetMode::ALL,
                       combine(framework::dataset::make("DataType", { DataType::F32 }),
                               framework::dataset::make("QueryWeightFormat", { arm_compute::WeightFormat::OHWIo2 })))
{
    ARM_COMPUTE_EXPECT(!_kernel_found, framework::LogLevel::ERRORS);
}

FIXTURE_DATA_TEST_CASE(UC2_1_CpuGemmConv2d_FastMath, HasOptImplFixtureFastMath<cpu::CpuGemmConv2d>, framework::DatasetMode::ALL,
                       combine(framework::dataset::make("DataType", { DataType::F32 }),
                               framework::dataset::make("QueryWeightFormat", { arm_compute::WeightFormat::OHWIo2 })))
{
    ARM_COMPUTE_EXPECT(!_kernel_found, framework::LogLevel::ERRORS);
}

FIXTURE_DATA_TEST_CASE(UC2_1_NEGEMMConvolutionLayer_FastMath, HasOptImplFixtureFastMath<NEGEMMConvolutionLayer>, framework::DatasetMode::ALL,
                       combine(framework::dataset::make("DataType", { DataType::F32 }),
                               framework::dataset::make("QueryWeightFormat", { arm_compute::WeightFormat::OHWIo2 })))
{
    ARM_COMPUTE_EXPECT(!_kernel_found, framework::LogLevel::ERRORS);
}

// UC2_2_* tests: the user requests a specific fixed format, and a
// kernel that support that fixed format is found.

FIXTURE_DATA_TEST_CASE(UC2_2_CpuGemmConv2d, HasOptImplFixtureNoFastMath<cpu::CpuGemmConv2d>, framework::DatasetMode::ALL,
                       combine(framework::dataset::make("DataType", { DataType::F32 }),
                               framework::dataset::make("QueryWeightFormat", { arm_compute::WeightFormat::OHWIo4 })))
{
    ARM_COMPUTE_EXPECT(_kernel_found, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(_computed_weight_format == arm_compute::WeightFormat::OHWIo4, framework::LogLevel::ERRORS);
}

FIXTURE_DATA_TEST_CASE(UC2_2_NEGEMMConvolutionLayer, HasOptImplFixtureNoFastMath<NEGEMMConvolutionLayer>, framework::DatasetMode::ALL,
                       combine(framework::dataset::make("DataType", { DataType::F32 }),
                               framework::dataset::make("QueryWeightFormat", { arm_compute::WeightFormat::OHWIo4 })))
{
    ARM_COMPUTE_EXPECT(_kernel_found, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(_computed_weight_format == arm_compute::WeightFormat::OHWIo4, framework::LogLevel::ERRORS);
}

#if defined(ARM_COMPUTE_ENABLE_BF16)
// These tests currently only works with SVE length 256
// If other SVE length is used a kernel will fail to be found
// This needs to be addressed in order to ensure it doesn't revert to FP32 kernels for systems with SVE length other than 256
FIXTURE_DATA_TEST_CASE(UC2_2_CpuGemmConv2d_FastMath, HasOptImplFixtureFastMath<cpu::CpuGemmConv2d>, framework::DatasetMode::ALL,
                       combine(framework::dataset::make("DataType", { DataType::F32 }),
                               framework::dataset::make("QueryWeightFormat", { arm_compute::WeightFormat::OHWIo8i4_bf16 })))
{
    if(Scheduler::get().cpu_info().has_bf16() && (arm_gemm::utils::get_vector_length<float>() == 8)){
        ARM_COMPUTE_EXPECT(_kernel_found, framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT_EQUAL(_computed_weight_format, arm_compute::WeightFormat::OHWIo8i4_bf16, framework::LogLevel::ERRORS);
    }
    else{
        ARM_COMPUTE_EXPECT(!_kernel_found, framework::LogLevel::ERRORS);
    }
}

FIXTURE_DATA_TEST_CASE(UC2_2_NEGEMMConvolutionLayer_FastMath, HasOptImplFixtureFastMath<NEGEMMConvolutionLayer>, framework::DatasetMode::ALL,
                       combine(framework::dataset::make("DataType", { DataType::F32 }),
                               framework::dataset::make("QueryWeightFormat", { arm_compute::WeightFormat::OHWIo8i4_bf16 })))
{
    if(Scheduler::get().cpu_info().has_bf16() && (arm_gemm::utils::get_vector_length<float>() == 8)){
        ARM_COMPUTE_EXPECT(_kernel_found, framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(_computed_weight_format == arm_compute::WeightFormat::OHWIo8i4_bf16, framework::LogLevel::ERRORS);
    }
    else{
        ARM_COMPUTE_EXPECT(!_kernel_found, framework::LogLevel::ERRORS);
    }
}

#endif // ARM_COMPUTE_ENABLE_BF16

// UC3_1_* tests: the user queries for ANY fixed format, but there is
// no kernel that support the use case specified by the user (for
// example, there is no fixed format kernel for the datatype of the
// problem).

FIXTURE_DATA_TEST_CASE(UC3_1_CpuGemmConv2d, HasOptImplFixtureNoFastMath<cpu::CpuGemmConv2d>, framework::DatasetMode::ALL,
                       combine(framework::dataset::make("DataType", { DataType::S32 }),
                               framework::dataset::make("QueryWeightFormat", { arm_compute::WeightFormat::ANY })))
{
    ARM_COMPUTE_EXPECT(!_kernel_found, framework::LogLevel::ERRORS);
}

FIXTURE_DATA_TEST_CASE(UC3_1_NEGEMMConvolutionLayer, HasOptImplFixtureNoFastMath<NEGEMMConvolutionLayer>, framework::DatasetMode::ALL,
                       combine(framework::dataset::make("DataType", { DataType::S32 }),
                               framework::dataset::make("QueryWeightFormat", { arm_compute::WeightFormat::ANY })))
{
    ARM_COMPUTE_EXPECT(!_kernel_found, framework::LogLevel::ERRORS);
}

FIXTURE_DATA_TEST_CASE(UC3_1_CpuGemmConv2d_FastMath, HasOptImplFixtureFastMath<cpu::CpuGemmConv2d>, framework::DatasetMode::ALL,
                       combine(framework::dataset::make("DataType", { DataType::S32 }),
                               framework::dataset::make("QueryWeightFormat", { arm_compute::WeightFormat::ANY })))
{
    ARM_COMPUTE_EXPECT(!_kernel_found, framework::LogLevel::ERRORS);
}

FIXTURE_DATA_TEST_CASE(UC3_1_NEGEMMConvolutionLayer_FastMath, HasOptImplFixtureFastMath<NEGEMMConvolutionLayer>, framework::DatasetMode::ALL,
                       combine(framework::dataset::make("DataType", { DataType::S32 }),
                               framework::dataset::make("QueryWeightFormat", { arm_compute::WeightFormat::ANY })))
{
    ARM_COMPUTE_EXPECT(!_kernel_found, framework::LogLevel::ERRORS);
}

// UC3_2_* tests: the user queries for ANY fixed format. The search
// succeeded and the fixed format found is prompted back for
// consumption by the user. Note that we just test the
// _computed_weight_format to be anything but not the formats that are
// not fixed formats (ANY and UNSPECIFIED). This is because the weight
// format that the runtime produces depends on the size of the vector
// units of the hardware where the tests is executed. For example, a
// format like OHWIo4 for FP32 data returned for 128-bit NEON hardware
// is replaced by OHWIo8 when running on 256-bit SVE.

FIXTURE_DATA_TEST_CASE(UC3_2_CpuGemmConv2d, HasOptImplFixtureNoFastMath<cpu::CpuGemmConv2d>, framework::DatasetMode::ALL,
                       combine(framework::dataset::make("DataType", { DataType::F32 }),
                               framework::dataset::make("QueryWeightFormat", { arm_compute::WeightFormat::ANY })))
{
    ARM_COMPUTE_EXPECT(_kernel_found, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(_computed_weight_format != arm_compute::WeightFormat::ANY, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(_computed_weight_format != arm_compute::WeightFormat::UNSPECIFIED, framework::LogLevel::ERRORS);
}

FIXTURE_DATA_TEST_CASE(UC3_2_NEGEMMConvolutionLayer, HasOptImplFixtureNoFastMath<NEGEMMConvolutionLayer>, framework::DatasetMode::ALL,
                       combine(framework::dataset::make("DataType", { DataType::F32 }),
                               framework::dataset::make("QueryWeightFormat", { arm_compute::WeightFormat::ANY })))
{
    ARM_COMPUTE_EXPECT(_computed_weight_format != arm_compute::WeightFormat::ANY, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(_computed_weight_format != arm_compute::WeightFormat::UNSPECIFIED, framework::LogLevel::ERRORS);
}

#if defined(ARM_COMPUTE_ENABLE_BF16)

FIXTURE_DATA_TEST_CASE(UC3_2_CpuGemmConv2d_FastMath, HasOptImplFixtureFastMath<cpu::CpuGemmConv2d>, framework::DatasetMode::ALL,
                       combine(framework::dataset::make("DataType", { DataType::F32 }),
                               framework::dataset::make("QueryWeightFormat", { arm_compute::WeightFormat::ANY })))
{
    if(Scheduler::get().cpu_info().has_bf16()){
        ARM_COMPUTE_EXPECT(_kernel_found, framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(_computed_weight_format != arm_compute::WeightFormat::ANY, framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(_computed_weight_format != arm_compute::WeightFormat::UNSPECIFIED, framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(arm_compute::is_fixed_format_fast_math(_computed_weight_format), framework::LogLevel::ERRORS);
    }
    else{
        ARM_COMPUTE_EXPECT(_kernel_found, framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(_computed_weight_format != arm_compute::WeightFormat::ANY, framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(_computed_weight_format != arm_compute::WeightFormat::UNSPECIFIED, framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!arm_compute::is_fixed_format_fast_math(_computed_weight_format), framework::LogLevel::ERRORS);
    }
}

FIXTURE_DATA_TEST_CASE(UC3_2_NEGEMMConvolutionLayer_FastMath, HasOptImplFixtureFastMath<NEGEMMConvolutionLayer>, framework::DatasetMode::ALL,
                       combine(framework::dataset::make("DataType", { DataType::F32 }),
                               framework::dataset::make("QueryWeightFormat", { arm_compute::WeightFormat::ANY })))
{
    if(Scheduler::get().cpu_info().has_bf16()){
        ARM_COMPUTE_EXPECT(_kernel_found, framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(_computed_weight_format != arm_compute::WeightFormat::ANY, framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(_computed_weight_format != arm_compute::WeightFormat::UNSPECIFIED, framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(arm_compute::is_fixed_format_fast_math(_computed_weight_format), framework::LogLevel::ERRORS);
    }
    else{
        ARM_COMPUTE_EXPECT(_kernel_found, framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(_computed_weight_format != arm_compute::WeightFormat::ANY, framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(_computed_weight_format != arm_compute::WeightFormat::UNSPECIFIED, framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!arm_compute::is_fixed_format_fast_math(_computed_weight_format), framework::LogLevel::ERRORS);
    }
}

#endif // ARM_COMPUTE_ENABLE_BF16

namespace
{
using TestCaseType          = std::tuple<TensorShape, TensorShape, arm_compute::WeightFormat>;
auto prepare_weights_shapes = framework::dataset::make("TensorShape",
{
    // OHWIo<interleave_by>i<block_by>
    //
    // OHWI --> O'HWI', where:
    //
    //   O'= smallest multiple of <interleave_by> such that O<=O'
    //   I'= smallest multiple of <block_by> such that I<=I'
    //

    // Change N for OHWIo4
    TestCaseType({ { 1U, 1U, 1U, 1U }, { 1U, 1U, 1U, 4U }, arm_compute::WeightFormat::OHWIo4 }),
    TestCaseType({ { 1U, 1U, 1U, 2U }, { 1U, 1U, 1U, 4U }, arm_compute::WeightFormat::OHWIo4 }),
    TestCaseType({ { 1U, 1U, 1U, 3U }, { 1U, 1U, 1U, 4U }, arm_compute::WeightFormat::OHWIo4 }),
    TestCaseType({ { 1U, 1U, 1U, 4U }, { 1U, 1U, 1U, 4U }, arm_compute::WeightFormat::OHWIo4 }),
    TestCaseType({ { 1U, 1U, 1U, 5U }, { 1U, 1U, 1U, 8U }, arm_compute::WeightFormat::OHWIo4 }),
    TestCaseType({ { 1U, 1U, 1U, 6U }, { 1U, 1U, 1U, 8U }, arm_compute::WeightFormat::OHWIo4 }),
    TestCaseType({ { 1U, 1U, 1U, 7U }, { 1U, 1U, 1U, 8U }, arm_compute::WeightFormat::OHWIo4 }),
    TestCaseType({ { 1U, 1U, 1U, 8U }, { 1U, 1U, 1U, 8U }, arm_compute::WeightFormat::OHWIo4 }),
    TestCaseType({ { 1U, 1U, 1U, 9U }, { 1U, 1U, 1U, 12U }, arm_compute::WeightFormat::OHWIo4 }),
    // // Change N for OHWIo8
    TestCaseType({ { 1U, 1U, 1U, 1U }, { 1U, 1U, 1U, 8U }, arm_compute::WeightFormat::OHWIo8 }),
    TestCaseType({ { 1U, 1U, 1U, 2U }, { 1U, 1U, 1U, 8U }, arm_compute::WeightFormat::OHWIo8 }),
    TestCaseType({ { 1U, 1U, 1U, 3U }, { 1U, 1U, 1U, 8U }, arm_compute::WeightFormat::OHWIo8 }),
    TestCaseType({ { 1U, 1U, 1U, 4U }, { 1U, 1U, 1U, 8U }, arm_compute::WeightFormat::OHWIo8 }),
    TestCaseType({ { 1U, 1U, 1U, 5U }, { 1U, 1U, 1U, 8U }, arm_compute::WeightFormat::OHWIo8 }),
    TestCaseType({ { 1U, 1U, 1U, 6U }, { 1U, 1U, 1U, 8U }, arm_compute::WeightFormat::OHWIo8 }),
    TestCaseType({ { 1U, 1U, 1U, 7U }, { 1U, 1U, 1U, 8U }, arm_compute::WeightFormat::OHWIo8 }),
    TestCaseType({ { 1U, 1U, 1U, 8U }, { 1U, 1U, 1U, 8U }, arm_compute::WeightFormat::OHWIo8 }),
    TestCaseType({ { 1U, 1U, 1U, 9U }, { 1U, 1U, 1U, 16U }, arm_compute::WeightFormat::OHWIo8 }),
    // // Change N for OHWIo4 when H, W and C are not 1
    TestCaseType({ { 3U, 4U, 2U, 1U }, { 3, 4, 2, 4 }, arm_compute::WeightFormat::OHWIo4 }),
    TestCaseType({ { 3U, 4U, 2U, 2U }, { 3, 4, 2, 4 }, arm_compute::WeightFormat::OHWIo4 }),
    TestCaseType({ { 3U, 4U, 2U, 3U }, { 3, 4, 2, 4 }, arm_compute::WeightFormat::OHWIo4 }),
    TestCaseType({ { 3U, 4U, 2U, 4U }, { 3, 4, 2, 4 }, arm_compute::WeightFormat::OHWIo4 }),
    TestCaseType({ { 3U, 4U, 2U, 5U }, { 3, 4, 2, 8 }, arm_compute::WeightFormat::OHWIo4 }),
    TestCaseType({ { 3U, 4U, 2U, 6U }, { 3, 4, 2, 8 }, arm_compute::WeightFormat::OHWIo4 }),
    TestCaseType({ { 3U, 4U, 2U, 7U }, { 3, 4, 2, 8 }, arm_compute::WeightFormat::OHWIo4 }),
    TestCaseType({ { 3U, 4U, 2U, 8U }, { 3, 4, 2, 8 }, arm_compute::WeightFormat::OHWIo4 }),
    TestCaseType({ { 3U, 4U, 2U, 9U }, { 3, 4, 2, 12 }, arm_compute::WeightFormat::OHWIo4 }),

    // // Fix N and move HWI around, with different data layouts and formats
    TestCaseType({ { 2U, 4U, 3U, 5U }, { 2, 4, 3, 8 }, arm_compute::WeightFormat::OHWIo4 }),
    TestCaseType({ { 3U, 4U, 2U, 5U }, { 3, 4, 2, 8 }, arm_compute::WeightFormat::OHWIo4 }),
    TestCaseType({ { 2U, 4U, 3U, 9U }, { 2, 4, 3, 16 }, arm_compute::WeightFormat::OHWIo8 }),
    TestCaseType({ { 3U, 4U, 2U, 9U }, { 3, 4, 2, 16 }, arm_compute::WeightFormat::OHWIo8 }),
    TestCaseType({ { 1024U, 1U, 1U, 1001U }, { 1024, 1, 1, 1008 }, arm_compute::WeightFormat::OHWIo8 }),

    // // Adding <block_by> on I (=C)
    TestCaseType({ { 1U, 4U, 3U, 5U }, { 2, 4, 3, 8 }, arm_compute::WeightFormat::OHWIo4i2 }),
    TestCaseType({ { 2U, 4U, 3U, 5U }, { 2, 4, 3, 8 }, arm_compute::WeightFormat::OHWIo4i2 }),
    TestCaseType({ { 3U, 4U, 3U, 5U }, { 4, 4, 3, 8 }, arm_compute::WeightFormat::OHWIo4i2 }),

    // ---------
    TestCaseType({ { 2, 2, 1, 5 }, { 2, 2, 1, 8 }, arm_compute::WeightFormat::OHWIo4 }),
    TestCaseType({ { 1, 2, 2, 5 }, { 1, 2, 2, 8 }, arm_compute::WeightFormat::OHWIo4 }),

});
} // unnamed namespace

DATA_TEST_CASE(PrepareWeightShape, framework::DatasetMode::ALL,
               prepare_weights_shapes, shapes)
{
    const TensorShape               input_shape    = std::get<0>(shapes);
    const TensorShape               expected_shape = std::get<1>(shapes);
    const arm_compute::WeightFormat wf             = std::get<2>(shapes);
    const DataType                  DT             = DataType::F32;
    const DataLayout                DL             = DataLayout::NHWC;
    const auto                      TI             = TensorInfo(input_shape, 1 /*num_channels, deprecated*/, DT, DL);
    const TensorInfo                computed_info  = ::arm_compute::test::validation::prepare_weights(TI, wf);
    ARM_COMPUTE_EXPECT_EQUAL(computed_info.tensor_shape(), expected_shape, framework::LogLevel::ERRORS);
}

TEST_SUITE_END() // VariableWeightUtils

TEST_SUITE(ExperimentalCpuAPIVariableWeightWithFixtures)

template <typename ScalarType>
using VarWidth = VariableWeightsFixture<cpu::CpuGemmConv2d, Tensor, Accessor, ScalarType, /*enable_fast_math*/ false>;

FIXTURE_DATA_TEST_CASE(RunSmallFloat, VarWidth<float>, framework::DatasetMode::ALL,
                       combine(combine(datasets::SmallConvolutionLayerDataset(),
                                       framework::dataset::make("DataLayout", { DataLayout::NHWC })),
                               framework::dataset::make("ACL Scalar type", { DataType::F32 })))
{
    // Validate output
    validate(Accessor(_target), _reference, rel_tolerance_f32, 0.f, float(abs_tolerance_f32));
}

#if defined(ARM_COMPUTE_ENABLE_FP16)
FIXTURE_DATA_TEST_CASE(RunSmallHalf, VarWidth<half>, framework::DatasetMode::ALL,
                       combine(combine(datasets::SmallConvolutionLayerDataset(),
                                       framework::dataset::make("DataLayout", { DataLayout::NHWC })),
                               framework::dataset::make("ACL Scalar type", { DataType::F16 })))
{
    if(CPUInfo::get().has_fp16())
    {
        // Validate output
        validate(Accessor(_target), _reference, rel_tolerance_f16, 0.f, half(abs_tolerance_f16));
    }
    else
    {
        ARM_COMPUTE_TEST_INFO("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_INFO();
    }
}
#endif // ARM_COMPUTE_ENABLE_FP16

#if defined(ARM_COMPUTE_ENABLE_BF16)
template <typename ScalarType>
using VarWidthFastMath = VariableWeightsFixture<cpu::CpuGemmConv2d, Tensor, Accessor, ScalarType, /*enable_fast_math*/ true>;

FIXTURE_DATA_TEST_CASE(RunSmallFloatFastMath, VarWidthFastMath<float>, framework::DatasetMode::ALL,
                       combine(combine(datasets::SmallConvolutionLayerDataset(),
                                       framework::dataset::make("DataLayout", { DataLayout::NHWC })),
                               framework::dataset::make("ACL Scalar type", { DataType::F32 })))
{
    // Validate output
    validate(Accessor(_target), _reference, rel_tolerance_f32, 0.f, float(abs_tolerance_f32));
}
#endif // ARM_COMPUTE_ENABLE_BF16

TEST_SUITE_END() // ExperimentalCpuAPIVariableWeightWithFixtures

TEST_SUITE(ExperimentalNEAPIVariableWeightWithFixtures)

template <typename ScalarType>
using NEGEMMVarWidth = VariableWeightsFixtureNEInterface<NEGEMMConvolutionLayer, Tensor, Accessor, ScalarType, /*enable_fast_math*/ false>;

FIXTURE_DATA_TEST_CASE(NEGEMMRunSmallFloat, NEGEMMVarWidth<float>, framework::DatasetMode::ALL,
                       combine(combine(datasets::SmallConvolutionLayerDataset(),
                                       framework::dataset::make("DataLayout", { DataLayout::NHWC })),
                               framework::dataset::make("ACL Scalar type", { DataType::F32 })))
{
    // Validate output
    validate(Accessor(_target), _reference, rel_tolerance_f32, 0.f, float(abs_tolerance_f32));
}

#if defined(ARM_COMPUTE_ENABLE_FP16)
FIXTURE_DATA_TEST_CASE(NEGEMMRunSmallHalf, NEGEMMVarWidth<half>, framework::DatasetMode::ALL,
                       combine(combine(datasets::SmallConvolutionLayerDataset(),
                                       framework::dataset::make("DataLayout", { DataLayout::NHWC })),
                               framework::dataset::make("ACL Scalar type", { DataType::F16 })))
{
    if(CPUInfo::get().has_fp16())
    {
        // Validate output
        validate(Accessor(_target), _reference, rel_tolerance_f16, 0.f, half(abs_tolerance_f16));
    }
    else
    {
        ARM_COMPUTE_TEST_INFO("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_INFO();
    }
}
#endif // ARM_COMPUTE_ENABLE_FP16

#if defined(ARM_COMPUTE_ENABLE_BF16)
template <typename ScalarType>
using NEGEMMVarWidthFastMath = VariableWeightsFixtureNEInterface<NEGEMMConvolutionLayer, Tensor, Accessor, ScalarType, /*enable_fast_math*/ true>;

FIXTURE_DATA_TEST_CASE(NEGEMMRunSmallFloatFastMath, NEGEMMVarWidthFastMath<float>, framework::DatasetMode::ALL,
                       combine(combine(datasets::SmallConvolutionLayerDataset(),
                                       framework::dataset::make("DataLayout", { DataLayout::NHWC })),
                               framework::dataset::make("ACL Scalar type", { DataType::F32 })))
{
    // Validate output
    validate(Accessor(_target), _reference, rel_tolerance_f32, 0.f, float(abs_tolerance_f32));
}
#endif // ARM_COMPUTE_ENABLE_BF16

TEST_SUITE_END() // ExperimentalNEAPIVariableWeightWithFixtures
TEST_SUITE_END() // FIXED_FORMAT_KERNELS

#endif // ARM_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS

TEST_SUITE(GEMMConvolutionLayer)
template <typename T>
using NEGEMMConvolutionLayerFixture = ConvolutionValidationFixture<Tensor, Accessor, NEConvolutionLayer, T>;
template <typename T>
using NEGEMMConvolutionLayerPaddedWeightsFixture = ConvolutionValidationPaddedWeightsFixture<Tensor, Accessor, NEConvolutionLayer, T>;
template <typename T>
using NEGEMMConvolutionLayerMixedDataLayoutFixture = ConvolutionValidationFixture<Tensor, Accessor, NEConvolutionLayer, T, true>;

/** Test case for memory injection in @ref cpu::CpuGemmConv2d.
 *
 * Configure the operator once and inject memory at run-time in multiple executions.
 *
 * Checks performed in order:
 * - Both runs compute the same output
 */
TEST_CASE(MemoryInjection, framework::DatasetMode::ALL)
{
    auto        conv        = std::make_unique<cpu::CpuGemmConv2d>();
    const auto  src_info    = TensorInfo(TensorShape(1U, 5U, 2U), 1, DataType::F32, DataLayout::NCHW);
    const auto  weight_info = TensorInfo(TensorShape(1U, 3U, 2U, 3U), 1, DataType::F32, DataLayout::NCHW);
    const auto  bias_info   = TensorInfo(TensorShape(3U), 1, DataType::F32, DataLayout::NCHW);
    auto        dst_info    = TensorInfo(TensorShape(1U, 7U, 3U), 1, DataType::F32, DataLayout::NCHW);
    const auto  conv_info   = PadStrideInfo(1, 1, 0, 0, 2, 2, DimensionRoundingType::FLOOR);
    WeightsInfo weights_info(false, 3U, 3U, 1U);
    conv->configure(&src_info, &weight_info, &bias_info, &dst_info, conv_info, weights_info);

    // tensors are newly created every call of this lambda function
    auto src    = create_tensor<Tensor>(src_info);
    auto weight = create_tensor<Tensor>(weight_info);
    auto bias   = create_tensor<Tensor>(bias_info);
    src.allocator()->allocate();
    weight.allocator()->allocate();
    bias.allocator()->allocate();

    ITensorPack run_pack{ { TensorType::ACL_SRC_0, &src }, { TensorType::ACL_SRC_1, &weight }, { TensorType::ACL_SRC_2, &bias } };
    ITensorPack prep_pack{ { TensorType::ACL_SRC_1, &weight }, { TensorType::ACL_SRC_2, &bias } };

    auto mg = MemoryGroup{};
    auto ws = manage_workspace<Tensor>(conv->workspace(), mg, run_pack, prep_pack);

    auto run_conv = [&]() -> Tensor
    {
        auto dst = create_tensor<Tensor>(dst_info);
        dst.allocator()->allocate();
        run_pack.add_tensor(TensorType::ACL_DST, &dst);

        library->fill_tensor_value(Accessor(src), 1.f);
        library->fill_tensor_value(Accessor(weight), 2.f);
        library->fill_tensor_value(Accessor(bias), 3.f);
        // This operator is configured once and captured by this lambda.
        conv->prepare(prep_pack);
        conv->run(run_pack);
        return dst;
    };
    auto result_0 = run_conv();
    auto result_1 = run_conv();
    for(size_t i = 0; i < result_0.info()->tensor_shape().total_size(); ++i)
    {
        ARM_COMPUTE_EXPECT(reinterpret_cast<float *>(result_0.buffer())[i] == reinterpret_cast<float *>(result_1.buffer())[i], framework::LogLevel::ERRORS);
    }
}

/** Test case for memory injection in @ref NEGEMMConvolutionLayer.
 *
 * Make sure @ref NEGEMMConvolutionLayer still works through injecting the memory at configure time using the old API.
 *
 * Checks performed in order:
 * - Both runs compute the same output
 */
TEST_CASE(MultipleExecutionWithConfigure, framework::DatasetMode::ALL)
{
    auto        conv        = std::make_unique<NEGEMMConvolutionLayer>();
    const auto  src_info    = TensorInfo(TensorShape(1U, 5U, 2U), 1, DataType::F32, DataLayout::NCHW);
    const auto  weight_info = TensorInfo(TensorShape(1U, 3U, 2U, 3U), 1, DataType::F32, DataLayout::NCHW);
    const auto  bias_info   = TensorInfo(TensorShape(3U), 1, DataType::F32, DataLayout::NCHW);
    auto        dst_info    = TensorInfo(TensorShape(1U, 7U, 3U), 1, DataType::F32, DataLayout::NCHW);
    const auto  conv_info   = PadStrideInfo(1, 1, 0, 0, 2, 2, DimensionRoundingType::FLOOR);
    WeightsInfo weights_info(false, 3U, 3U, 1U);
    auto        run_conv = [&]()
    {
        auto src    = create_tensor<Tensor>(src_info);
        auto weight = create_tensor<Tensor>(weight_info);
        auto bias   = create_tensor<Tensor>(bias_info);
        auto dst    = create_tensor<Tensor>(dst_info);
        conv->configure(&src, &weight, &bias, &dst, conv_info, weights_info);
        src.allocator()->allocate();
        weight.allocator()->allocate();
        bias.allocator()->allocate();
        dst.allocator()->allocate();
        library->fill_tensor_value(Accessor(src), 1.f);
        library->fill_tensor_value(Accessor(weight), 2.f);
        library->fill_tensor_value(Accessor(bias), 3.f);
        conv->run();
        return dst;
    };
    auto result_0 = run_conv();
    auto result_1 = run_conv();
    for(size_t i = 0; i < result_0.info()->tensor_shape().total_size(); ++i)
    {
        ARM_COMPUTE_EXPECT(reinterpret_cast<float *>(result_0.buffer())[i] == reinterpret_cast<float *>(result_1.buffer())[i], framework::LogLevel::ERRORS);
    }
}

TEST_SUITE(Float)
#if defined(ARM_COMPUTE_ENABLE_BF16)
TEST_SUITE(BFLOAT16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMConvolutionLayerFixture<float>, framework::DatasetMode::ALL, combine(combine(combine(combine(datasets::SmallConvolutionLayerDataset(),
                                                                                                                    framework::dataset::make("ReshapeWeights", { true })),
                                                                                                                    framework::dataset::make("DataType", Scheduler::get().cpu_info().has_bf16() ? DataType::BFLOAT16 : DataType::F32)),
                                                                                                                    framework::dataset::make("DataLayout", { DataLayout::NHWC })),
                                                                                                            ActivationFunctionsDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, rel_tolerance_f32, 0.f, float(abs_tolerance_f32));
}
TEST_SUITE_END() // BFLOAT16
#endif           /* defined(ARM_COMPUTE_ENABLE_BF16) */

#ifdef ARM_COMPUTE_ENABLE_FP16
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMConvolutionLayerFixture<half>, framework::DatasetMode::ALL, combine(combine(combine(combine(datasets::SmallConvolutionLayerDataset(),
                                                                                                                   framework::dataset::make("ReshapeWeights", { true })),
                                                                                                                   framework::dataset::make("DataType", DataType::F16)),
                                                                                                                   framework::dataset::make("DataLayout", { DataLayout::NCHW })),
                                                                                                           ActivationFunctionsDataset))
{
    if(CPUInfo::get().has_fp16())
    {
        // Validate output
        validate(Accessor(_target), _reference, rel_tolerance_f16, tolerance_num, abs_tolerance_f16);
    }
    else
    {
        ARM_COMPUTE_TEST_INFO("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_INFO();
    }
}
TEST_SUITE_END() // FP16
#endif           /* ARM_COMPUTE_ENABLE_FP16 */

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMConvolutionLayerFixture<float>, framework::DatasetMode::ALL, combine(combine(combine(combine(datasets::SmallConvolutionLayerDataset(),
                                                                                                                    framework::dataset::make("ReshapeWeights", { true })),
                                                                                                                    framework::dataset::make("DataType", DataType::F32)),
                                                                                                                    framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                            ActivationFunctionsDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, rel_tolerance_f32, 0.f, float(abs_tolerance_f32));
}
FIXTURE_DATA_TEST_CASE(RunMixedDataLayout, NEGEMMConvolutionLayerMixedDataLayoutFixture<float>, framework::DatasetMode::ALL,
                       combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                                           framework::dataset::make("Input", TensorShape(23U, 27U, 5U)),
                                                                                           framework::dataset::make("Weights", TensorShape(3U, 3U, 5U, 2U))),
                                                                                       framework::dataset::make("Bias", TensorShape(2U))),
                                                                               framework::dataset::make("Output", TensorShape(11U, 25U, 2U))),
                                                                       framework::dataset::make("PadStrideInfo", PadStrideInfo(2, 1, 0, 0))),
                                                               framework::dataset::make("Dilation", Size2D(1, 1))),
                                                       framework::dataset::make("ReshapeWeights", { true })),
                                               framework::dataset::make("DataType", DataType::F32)),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                               ActivationFunctionsDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, rel_tolerance_f32, 0.f, float(abs_tolerance_f32));
}
/** Padded weights
 * CpuGemmConv2d uses two different paths for reshaping the weights based on if the weight tensor has holes (a common
 * way to have "holes" in tensor is via extended paddings)
 *
 * We only need to test the padded weight path here on a single floating data type and a single layout, because the fallback path is agnostic of them
 */
FIXTURE_DATA_TEST_CASE(RunPaddedWeights, NEGEMMConvolutionLayerPaddedWeightsFixture<float>, framework::DatasetMode::ALL, combine(datasets::SmallConvolutionLayerDataset(),
                                                                                                                    framework::dataset::make("ReshapeWeights", { true }),
                                                                                                                    framework::dataset::make("DataType", DataType::F32),
                                                                                                                    framework::dataset::make("DataLayout", { DataLayout::NHWC })
                                                                                                            ))
{
    // Validate output
    validate(Accessor(_target), _reference, rel_tolerance_f32, 0.f, float(abs_tolerance_f32));
}

// This very large shape test is required to test heuristic paths where the tensor size is > 1e7 bytes
// and weight dimensions larger than 7
FIXTURE_DATA_TEST_CASE(RunVeryLarge, NEGEMMConvolutionLayerFixture<float>, framework::DatasetMode::NIGHTLY,
    combine(datasets::VeryLargeConvolutionLayerDataset(),
        framework::dataset::make("ReshapeWeights", { true }),
        framework::dataset::make("DataType", DataType::F32),
        framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC }),
        NoActivation))
{
    // Validate output
    validate(Accessor(_target), _reference, rel_tolerance_f32, 0.f, float(abs_tolerance_f32));
}

TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float

// TODO(COMPMID-6573): Extend quantized tests with at least one suite where the weight is padded (the legacy case, see floating point's RunPaddedWeights)
template <typename T>
using NEGEMMConvolutionLayerForUpdatedStaticQuantInfoAfterConfigureFixture = ConvolutionValidationForUpdatedStaticQuantInfoAfterConfigureFixture<Tensor, Accessor, NEGEMMConvolutionLayer, T>;
template <typename T>
using NEGEMMConvolutionLayerQuantizedFixture = ConvolutionValidationQuantizedFixture<Tensor, Accessor, NEConvolutionLayer, T>;
template <typename T>
using NEGEMMConvolutionLayerQuantizedMixedDataLayoutFixture = ConvolutionValidationQuantizedFixture<Tensor, Accessor, NEConvolutionLayer, T, true>;

using NEGEMMConvolutionLayerQuantizedMixedSignFixture = ConvolutionValidationQuantizedMixedTypeFixture<Tensor, Accessor, NEConvolutionLayer, uint8_t, int8_t>;

template <typename T>
using NEGEMMConvolutionLayerQuantizedPerChannelFixture = ConvolutionValidationQuantizedPerChannelFixture<Tensor, Accessor, NEConvolutionLayer, T, int8_t>;

const auto QuantizedActivationFunctionsDataset = framework::dataset::make("ActivationInfo",
{
    ActivationLayerInfo(),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 6.f)
});
TEST_SUITE(Quantized)
/// @note: Every asymmetric quantized test where there's no fused activation will have its quantization info ignored
/// This is because instead of using the same quantization information for all the tensors, the fixture generates
/// separate quantization info for each input and the output tensor.
/// When we can also support dynamic quantization with the presence of activation, these two versions should be merged
/// again, with the explicitly specified quantization info removed

#ifdef __aarch64__
TEST_SUITE(UpdateStaticQuantInfoAfterConfigure)
TEST_SUITE(QASYMM8_SIGNED)
FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMConvolutionLayerForUpdatedStaticQuantInfoAfterConfigureFixture<int8_t>, framework::DatasetMode::ALL, combine(combine(combine(combine(combine(datasets::SmallConvolutionLayerDataset(),
                                                                                                                      framework::dataset::make("ReshapeWeights", { true })),
                                                                                                                      framework::dataset::make("DataType", DataType::QASYMM8_SIGNED)),
                                                                                                                      framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                                      framework::dataset::make("QuantizationInfoIfActivationEnabled", { QuantizationInfo(0.01f, -10) })),
                                                                                                                      QuantizedActivationFunctionsDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // QASYMM8_SIGNED

TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMConvolutionLayerForUpdatedStaticQuantInfoAfterConfigureFixture<uint8_t>, framework::DatasetMode::ALL, combine(combine(combine(combine(combine(datasets::SmallConvolutionLayerDataset(),
                                                                                                                       framework::dataset::make("ReshapeWeights", { true })),
                                                                                                                       framework::dataset::make("DataType", DataType::QASYMM8)),
                                                                                                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                                       framework::dataset::make("QuantizationInfoIfActivationEnabled", { QuantizationInfo(2.f / 255.f, 10) })),
                                                                                                                       QuantizedActivationFunctionsDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // QASYMM8
TEST_SUITE_END() // UpdateStaticQuantInfoAfterConfigure
#endif // __aarch64__

TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMConvolutionLayerQuantizedFixture<uint8_t>, framework::DatasetMode::ALL, combine(combine(combine(combine(combine(datasets::SmallConvolutionLayerDataset(),
                                                                                                                       framework::dataset::make("ReshapeWeights", { true })),
                                                                                                                       framework::dataset::make("DataType", DataType::QASYMM8)),
                                                                                                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                                       framework::dataset::make("QuantizationInfoIfActivationEnabled", { QuantizationInfo(2.f / 255.f, 10) })),
                                                                                                                       QuantizedActivationFunctionsDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunMixedDataLayout, NEGEMMConvolutionLayerQuantizedFixture<uint8_t>, framework::DatasetMode::ALL,
                       combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                                                   framework::dataset::make("Input", TensorShape(23U, 27U, 5U)),
                                                                                                   framework::dataset::make("Weights", TensorShape(3U, 3U, 5U, 2U))),
                                                                                               framework::dataset::make("Bias", TensorShape(2U))),
                                                                                       framework::dataset::make("Output", TensorShape(11U, 25U, 2U))),
                                                                               framework::dataset::make("PadStrideInfo", PadStrideInfo(2, 1, 0, 0))),
                                                                       framework::dataset::make("Dilation", Size2D(1, 1))),
                                                               framework::dataset::make("ReshapeWeights", { true })),
                                                       framework::dataset::make("DataType", DataType::QASYMM8)),
                                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                       framework::dataset::make("QuantizationInfoIfActivationEnabled", { QuantizationInfo(2.f / 255.f, 10) })),
                               QuantizedActivationFunctionsDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // QASYMM8

TEST_SUITE(QASYMM8_SIGNED)
FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMConvolutionLayerQuantizedFixture<int8_t>, framework::DatasetMode::ALL, combine(combine(combine(combine(combine(datasets::SmallConvolutionLayerDataset(),
                                                                                                                      framework::dataset::make("ReshapeWeights", { true })),
                                                                                                                      framework::dataset::make("DataType", DataType::QASYMM8_SIGNED)),
                                                                                                                      framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                                      framework::dataset::make("QuantizationInfoIfActivationEnabled", { QuantizationInfo(0.01f, -10) })),
                                                                                                                      QuantizedActivationFunctionsDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunMixedDataLayout, NEGEMMConvolutionLayerQuantizedFixture<int8_t>, framework::DatasetMode::ALL,
                       combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                                                   framework::dataset::make("Input", TensorShape(23U, 27U, 5U)),
                                                                                                   framework::dataset::make("Weights", TensorShape(3U, 3U, 5U, 2U))),
                                                                                               framework::dataset::make("Bias", TensorShape(2U))),
                                                                                       framework::dataset::make("Output", TensorShape(11U, 25U, 2U))),
                                                                               framework::dataset::make("PadStrideInfo", PadStrideInfo(2, 1, 0, 0))),
                                                                       framework::dataset::make("Dilation", Size2D(1, 1))),
                                                               framework::dataset::make("ReshapeWeights", { true })),
                                                       framework::dataset::make("DataType", DataType::QASYMM8_SIGNED)),
                                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                       framework::dataset::make("QuantizationInfoIfActivationEnabled", { QuantizationInfo(2.f / 255.f, 10) })),
                               QuantizedActivationFunctionsDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // QASYMM8_SIGNED

TEST_SUITE(QASYMM8_MIXED)
FIXTURE_DATA_TEST_CASE(
    RunSmall,
    NEGEMMConvolutionLayerQuantizedMixedSignFixture,
    framework::DatasetMode::ALL,
    combine(combine(combine(combine(combine(combine(combine(datasets::SmallConvolutionLayerDataset(),
                                                            framework::dataset::make("ReshapeWeights", {true})),
                                                    framework::dataset::make("DataType", DataType::QASYMM8)),
                                            framework::dataset::make("WeightsDataType", DataType::QASYMM8_SIGNED)),
                                    framework::dataset::make("DataLayout", {DataLayout::NCHW, DataLayout::NHWC})),
                            framework::dataset::make("QuantizationInfoIfActivationEnabled",
{QuantizationInfo(2.f / 255.f, 10)})),
framework::dataset::make("WeightQuantizationInfoIfActivationEnabled",
{QuantizationInfo(2.f / 255.f, 10)})),
QuantizedActivationFunctionsDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(
    RunMixedDataLayout,
    NEGEMMConvolutionLayerQuantizedMixedSignFixture,
    framework::DatasetMode::ALL,
    combine(
        framework::dataset::make("Input", TensorShape(23U, 27U, 5U)),
        framework::dataset::make("Weights", TensorShape(3U, 3U, 5U, 2U)),
        framework::dataset::make("Bias", TensorShape(2U)),
        framework::dataset::make("Output", TensorShape(11U, 25U, 2U)),
        framework::dataset::make("PadStrideInfo", PadStrideInfo(2, 1, 0, 0)),
        framework::dataset::make("Dilation", Size2D(1, 1)),
        framework::dataset::make("ReshapeWeights", {true}),
        framework::dataset::make("DataType", DataType::QASYMM8),
        framework::dataset::make("WeightsDataType", DataType::QASYMM8_SIGNED),
        framework::dataset::make("DataLayout", {DataLayout::NCHW, DataLayout::NHWC}),
        framework::dataset::make("QuantizationInfoIfActivationEnabled", {QuantizationInfo(2.f / 255.f, 10)}),
        framework::dataset::make("WeightQuantizationInfoIfActivationEnabled", {QuantizationInfo(2.f / 255.f, 10)}),
        QuantizedActivationFunctionsDataset)
    )
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // QASYMM8_MIXED

TEST_SUITE(QSYMM8_PER_CHANNEL)
FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMConvolutionLayerQuantizedPerChannelFixture<uint8_t>, framework::DatasetMode::ALL,
                       combine(combine(combine(combine(combine(combine(datasets::SmallConvolutionLayerDataset(),
                                                                       framework::dataset::make("ReshapeWeights", { true })),
                                                               framework::dataset::make("DataType", { DataType::QASYMM8 })),
                                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                               QuantizationData),
                                       QuantizedActivationFunctionsDataset),
                               framework::dataset::make("WeightsDataType", { DataType::QSYMM8_PER_CHANNEL })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunSmallSigned, NEGEMMConvolutionLayerQuantizedPerChannelFixture<int8_t>, framework::DatasetMode::ALL,
                       combine(combine(combine(combine(combine(combine(datasets::SmallConvolutionLayerDataset(),
                                                                       framework::dataset::make("ReshapeWeights", { true })),
                                                               framework::dataset::make("DataType", { DataType::QASYMM8_SIGNED })),
                                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                               QuantizationData),
                                       QuantizedActivationFunctionsDataset),
                               framework::dataset::make("WeightsDataType", { DataType::QSYMM8_PER_CHANNEL })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}

FIXTURE_DATA_TEST_CASE(MemoryStressLargeChannels, NEGEMMConvolutionLayerQuantizedPerChannelFixture<int8_t>,
    framework::DatasetMode::ALL,
        combine(
            make("In", TensorShape(1U)),
            make("Weights", TensorShape(1U, 1U, 1U, 17000U)),
            make("Biases", TensorShape(17000U)),
            make("Out", TensorShape(1U, 1U, 17000U)),
            make("Info", PadStrideInfo(1, 1, 0, 0)),
            make("Dilation", Size2D(1, 1)),
            make("ReshapeWeights", { true }),
            make("DataType", { DataType::QASYMM8_SIGNED }),
            make("DataLayout", { DataLayout::NHWC }),
            make("QuantizationInfo", QuantizationInfo(0.5f, 10)),
            make("ActivationInfo", ActivationLayerInfo()),
            make("WeightsDataType", { DataType::QSYMM8_PER_CHANNEL })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}

TEST_SUITE_END() // QSYMM8_PER_CHANNEL
TEST_SUITE_END() // Quantized

TEST_SUITE_END() // GEMMConvolutionLayer

TEST_SUITE(DirectGEMMConv2d)
template <typename T>
using NEDirectGEMMConv2dLayerFixture = ConvolutionValidationFixture<Tensor, Accessor, NEGEMMConv2d, T>;

/** Test case for memory injection in @ref cpu::CpuGemmDirectConv2d.
 *
 * Configure the operator once and inject memory at run-time in multiple executions.
 *
 * Checks performed in order:
 * - Both runs compute the same output
 */
TEST_CASE(MemoryInjection, framework::DatasetMode::ALL)
{
    auto       conv        = std::make_unique<cpu::CpuGemmDirectConv2d>();
    const auto src_info    = TensorInfo(TensorShape(1U, 5U, 2U), 1, DataType::F32, DataLayout::NHWC);
    const auto weight_info = TensorInfo(TensorShape(1U, 3U, 2U, 3U), 1, DataType::F32, DataLayout::NHWC);
    const auto bias_info   = TensorInfo(TensorShape(3U), 1, DataType::F32, DataLayout::NHWC);
    auto       dst_info    = TensorInfo(TensorShape(1U, 7U, 3U), 1, DataType::F32, DataLayout::NHWC);
    const auto conv_info   = Conv2dInfo{};
    conv->configure(&src_info, &weight_info, &bias_info, &dst_info, conv_info);

    // tensors are newly created every call of this lambda function
    auto src    = create_tensor<Tensor>(src_info);
    auto weight = create_tensor<Tensor>(weight_info);
    auto bias   = create_tensor<Tensor>(bias_info);
    src.allocator()->allocate();
    weight.allocator()->allocate();
    bias.allocator()->allocate();

    ITensorPack run_pack{ { TensorType::ACL_SRC_0, &src }, { TensorType::ACL_SRC_1, &weight }, { TensorType::ACL_SRC_2, &bias } };
    ITensorPack prep_pack{ { TensorType::ACL_SRC_1, &weight }, { TensorType::ACL_SRC_2, &bias } };

    auto mg = MemoryGroup{};
    auto ws = manage_workspace<Tensor>(conv->workspace(), mg, run_pack, prep_pack);

    auto run_conv = [&]() -> Tensor
    {
        auto dst = create_tensor<Tensor>(dst_info);
        dst.allocator()->allocate();
        run_pack.add_tensor(TensorType::ACL_DST, &dst);

        library->fill_tensor_value(Accessor(src), 1.f);
        library->fill_tensor_value(Accessor(weight), 2.f);
        library->fill_tensor_value(Accessor(bias), 3.f);
        // This operator is configured once and captured by this lambda.
        conv->prepare(prep_pack);
        conv->run(run_pack);
        return dst;
    };
    auto result_0 = run_conv();
    auto result_1 = run_conv();
    for(size_t i = 0; i < result_0.info()->tensor_shape().total_size(); ++i)
    {
        ARM_COMPUTE_EXPECT(reinterpret_cast<float *>(result_0.buffer())[i] == reinterpret_cast<float *>(result_1.buffer())[i], framework::LogLevel::ERRORS);
    }
}

/** Test case for memory injection in @ref NEGEMMConv2d.
 *
 * Make sure @ref NEGEMMConv2d still works through injecting the memory at configure time using the old API.
 *
 * Checks performed in order:
 * - Both runs compute the same output
 */
TEST_CASE(MultipleExecutionWithConfigure, framework::DatasetMode::ALL)
{
    auto       conv        = std::make_unique<NEGEMMConv2d>();
    const auto src_info    = TensorInfo(TensorShape(1U, 5U, 2U), 1, DataType::F32, DataLayout::NHWC);
    const auto weight_info = TensorInfo(TensorShape(1U, 3U, 2U, 3U), 1, DataType::F32, DataLayout::NHWC);
    const auto bias_info   = TensorInfo(TensorShape(3U), 1, DataType::F32, DataLayout::NHWC);
    auto       dst_info    = TensorInfo(TensorShape(1U, 7U, 3U), 1, DataType::F32, DataLayout::NHWC);
    const auto conv_info   = Conv2dInfo{};
    auto       run_conv    = [&]()
    {
        auto src    = create_tensor<Tensor>(src_info);
        auto weight = create_tensor<Tensor>(weight_info);
        auto bias   = create_tensor<Tensor>(bias_info);
        auto dst    = create_tensor<Tensor>(dst_info);
        conv->configure(&src, &weight, &bias, &dst, conv_info);
        src.allocator()->allocate();
        weight.allocator()->allocate();
        bias.allocator()->allocate();
        dst.allocator()->allocate();
        library->fill_tensor_value(Accessor(src), 1.f);
        library->fill_tensor_value(Accessor(weight), 2.f);
        library->fill_tensor_value(Accessor(bias), 3.f);
        conv->run();
        return dst;
    };
    auto result_0 = run_conv();
    auto result_1 = run_conv();
    for(size_t i = 0; i < result_0.info()->tensor_shape().total_size(); ++i)
    {
        ARM_COMPUTE_EXPECT(reinterpret_cast<float *>(result_0.buffer())[i] == reinterpret_cast<float *>(result_1.buffer())[i], framework::LogLevel::ERRORS);
    }
}

TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDirectGEMMConv2dLayerFixture<float>, framework::DatasetMode::ALL, combine(combine(combine(combine(datasets::SmallConvolutionLayerDataset(),
                                                                                                                     framework::dataset::make("ReshapeWeights", { true })),
                                                                                                                     framework::dataset::make("DataType", DataType::F32)),
                                                                                                                     framework::dataset::make("DataLayout", { DataLayout::NHWC })),
                                                                                                             ActivationFunctionsDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, rel_tolerance_f32, 0.f, float(abs_tolerance_f32));
}
TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float

#ifdef __aarch64__
template <typename T>
using NEDirectGEMMConv2dLayerQuantizedFixture = ConvolutionValidationQuantizedFixture<Tensor, Accessor, NEGEMMConv2d, T>;

template <typename T>
using NEDirectGEMMConv2dLayerQuantizedPerChannelFixture = ConvolutionValidationQuantizedPerChannelFixture<Tensor, Accessor, NEGEMMConv2d, T, int8_t>;

const auto QuantizedActivationFunctionsDataset = framework::dataset::make("ActivationInfo",
{
    ActivationLayerInfo(),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 6.f)
});
TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDirectGEMMConv2dLayerQuantizedFixture<uint8_t>, framework::DatasetMode::ALL, combine(combine(combine(combine(combine(datasets::SmallConvolutionLayerDataset(),
                                                                                                                        framework::dataset::make("ReshapeWeights", { true })),
                                                                                                                        framework::dataset::make("DataType", DataType::QASYMM8)),
                                                                                                                        framework::dataset::make("DataLayout", { DataLayout::NHWC })),
                                                                                                                        framework::dataset::make("QuantizationInfo", { QuantizationInfo(2.f / 255.f, 10) })),
                                                                                                                        QuantizedActivationFunctionsDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // QASYMM8

TEST_SUITE(QASYMM8_SIGNED)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDirectGEMMConv2dLayerQuantizedFixture<int8_t>, framework::DatasetMode::ALL, combine(combine(combine(combine(combine(datasets::SmallConvolutionLayerDataset(),
                                                                                                                       framework::dataset::make("ReshapeWeights", { true })),
                                                                                                                       framework::dataset::make("DataType", DataType::QASYMM8_SIGNED)),
                                                                                                                       framework::dataset::make("DataLayout", { DataLayout::NHWC })),
                                                                                                                       framework::dataset::make("QuantizationInfo", { QuantizationInfo(0.01f, -10) })),
                                                                                                                       QuantizedActivationFunctionsDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // QASYMM8_SIGNED

TEST_SUITE(QSYMM8_PER_CHANNEL)
FIXTURE_DATA_TEST_CASE(RunSmallSigned, NEDirectGEMMConv2dLayerQuantizedPerChannelFixture<int8_t>, framework::DatasetMode::ALL,
                       combine(combine(combine(combine(combine(combine(datasets::SmallConvolutionLayerDataset(),
                                                                       framework::dataset::make("ReshapeWeights", { true })),
                                                               framework::dataset::make("DataType", { DataType::QASYMM8_SIGNED })),
                                                       framework::dataset::make("DataLayout", { DataLayout::NHWC })),
                                               QuantizationData),
                                       QuantizedActivationFunctionsDataset),
                               framework::dataset::make("WeightsDataType", { DataType::QSYMM8_PER_CHANNEL })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // QSYMM8_PER_CHANNEL
TEST_SUITE_END() // Quantized
#endif           // __aarch64__

TEST_SUITE_END() // DirectGEMMConv2d

TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
