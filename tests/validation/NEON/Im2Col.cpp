/*
 * Copyright (c) 2017-2021, 2024-2026 Arm Limited.
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

#include "src/cpu/kernels/CpuIm2ColKernel.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/framework/Macros.h"
#include "tests/NEON/Accessor.h"
#include "tests/NEON/Helper.h"
#include "tests/validation/fixtures/Im2ColFixture.h"
#include "tests/validation/Validation.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{

using framework::dataset::make;

const auto im2col_shapes =
    make("Shape",
         {TensorShape{11U, 11U, 11U}, TensorShape{16U, 16U, 16U}, TensorShape{27U, 13U, 7U},
          TensorShape{31U, 27U, 17U, 2U}, TensorShape{27U, 13U, 5U, 4U}, TensorShape{11U, 11U, 5U, 5U}});

const auto conv_filter_sizes =
    make("KernelDims", {Size2D(3U, 3U), Size2D(3U, 1U), Size2D(1U, 5U), Size2D(5U, 5U), Size2D(7U, 7U)});

const auto conv_args = combine(
    conv_filter_sizes,
    make("PadStride", {PadStrideInfo(1U, 1U, 0U, 0U), PadStrideInfo(1U, 1U, 1U, 1U), PadStrideInfo(2U, 2U, 0U, 2U)}),
    make("QuantizationInfo", QuantizationInfo(0.5f, 10)),
    make("DataLayout", {DataLayout::NCHW, DataLayout::NHWC}),
    make("NumGroups", {1}));

const auto conv_filter_sizes_small = make("KernelDims", {Size2D(3U, 3U), Size2D(3U, 1U), Size2D(1U, 5U)});

const auto conv_args_small_core =
    combine(conv_filter_sizes_small,
            make("PadStride", {PadStrideInfo(1U, 1U, 0U, 0U), PadStrideInfo(1U, 1U, 1U, 1U)}),
            make("QuantizationInfo", QuantizationInfo(0.5f, 10)));

const auto conv_args_small_nhwc =
    combine(conv_args_small_core, make("DataLayout", {DataLayout::NHWC}), make("NumGroups", {1}));

const auto conv_args_small_nchw =
    combine(conv_args_small_core, make("DataLayout", {DataLayout::NCHW}), make("NumGroups", {1}));

const auto conv_args_small = concat(conv_args_small_nhwc, conv_args_small_nchw);

// Channel padding logic is data type agnostic, therefore it's tested
// on a subset of the data types, including the major use case, Bf16.
const auto conv_args_small_channel_padding = combine(conv_args_small_nhwc, make("ChannelPadRight", {3}));

} // namespace
TEST_SUITE(NEON)
TEST_SUITE(Im2Col)

using CpuIm2Col = NESynthetizeFunctionWithZeroConstantKernelBorder<cpu::kernels::CpuIm2ColKernel>;

// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(
    make("InputInfo", {
        TensorInfo(TensorShape(10U, 12U, 2U), 1, DataType::U8),      // Unsupported data type
        TensorInfo(TensorShape(10U, 12U, 2U), 1, DataType::F32),     // Mismatching data type
        TensorInfo(TensorShape(10U, 12U, 2U), 1, DataType::QASYMM8), // Bias not supported with QASYMM8
        TensorInfo(TensorShape(10U, 12U, 2U), 1, DataType::QASYMM8), // Mismatching shapes
        TensorInfo(TensorShape(10U, 12U, 2U, 2U), 1, DataType::QASYMM8),
    }),
    make("OutputInfo",{
        TensorInfo(TensorShape(3U, 4U, 10U, 2U), 1, DataType::F16),
        TensorInfo(TensorShape(3U, 4U, 10U, 2U), 1, DataType::F16),
        TensorInfo(TensorShape(3U, 3U, 10U, 2U), 1, DataType::QASYMM8),
        TensorInfo(TensorShape(3U, 4U, 10U, 2U), 1, DataType::QASYMM8),
        TensorInfo(TensorShape(18U, 80U, 1U, 2U), 1, DataType::QASYMM8),
    }),
    make("HasBias", { true, true, true, false, false }),
    make("Expected", { false, false, false, false, true })
    ),
    input_info, output_info, has_bias, expected)
{
    bool status = bool(cpu::kernels::CpuIm2ColKernel::validate(&input_info, &output_info, Size2D(3U, 3U), PadStrideInfo(), has_bias));
    ARM_COMPUTE_EXPECT(status == expected, framework::LogLevel::ERRORS);
}
// clang-format on

DATA_TEST_CASE(ChannelPaddingNotSupportedInNCHW,
               framework::DatasetMode::ALL,
               zip(make("InputInfo",
                        {TensorInfo(TensorShape(10U, 12U, 2U, 2U), 1, DataType::F32, DataLayout::NCHW),
                         TensorInfo(TensorShape(2U, 12U, 10U, 2U), 1, DataType::F32, DataLayout::NHWC)}),
                   make("OutputInfo",
                        {TensorInfo(TensorShape(45U, 80U, 1U, 2U), 1, DataType::F32, DataLayout::UNKNOWN),
                         TensorInfo(TensorShape(45U, 80U, 1U, 2U), 1, DataType::F32, DataLayout::UNKNOWN)}),
                   make("ChannelPadRight", {3U, 3U}),
                   make("Expected", {false, true})),
               input_info,
               output_info,
               channel_pad_right,
               expected)
{
    const bool         has_bias   = false;
    const auto         dilation   = Size2D(1U, 1U);
    const unsigned int num_groups = 1U;

    const Status status = cpu::kernels::CpuIm2ColKernel::validate(
        &input_info, &output_info, Size2D(3U, 3U), PadStrideInfo(), has_bias, dilation, num_groups, channel_pad_right);

    ARM_COMPUTE_EXPECT(bool(status) == expected, framework::LogLevel::ERRORS);
}

template <typename T>
using CpuIm2ColFixture = Im2ColOpValidationFixture<Tensor, Accessor, CpuIm2Col, T, false>;

template <typename T>
using CpuIm2ColWithChannelPadFixture = Im2ColOpValidationWithChannelPadFixture<Tensor, Accessor, CpuIm2Col, T, false>;

TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       CpuIm2ColFixture<float>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(im2col_shapes, make("DataType", DataType::F32), conv_args_small))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge,
                       CpuIm2ColFixture<float>,
                       framework::DatasetMode::NIGHTLY,
                       combine(concat(im2col_shapes, datasets::LargeShapes()),
                               make("DataType", DataType::F32),
                               conv_args))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // FP32

#ifdef ARM_COMPUTE_ENABLE_BF16
TEST_SUITE(BF16)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       CpuIm2ColFixture<bfloat16>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(im2col_shapes, make("DataType", DataType::BFLOAT16), conv_args_small))
{
    if (CPUInfo::get().has_bf16())
    {
        // Validate output
        validate(Accessor(_target), _reference);
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("Device does not support Bf16 data type. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}
FIXTURE_DATA_TEST_CASE(RunSmallWithChannelPadding,
                       CpuIm2ColWithChannelPadFixture<bfloat16>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(im2col_shapes, make("DataType", DataType::BFLOAT16), conv_args_small_channel_padding))
{
    if (CPUInfo::get().has_bf16())
    {
        // Validate output
        validate(Accessor(_target), _reference);
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("Device does not support Bf16 data type. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}
TEST_SUITE_END() // BF16
#endif           // ARM_COMPUTE_ENABLE_BF16

#ifdef ARM_COMPUTE_ENABLE_FP16

TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       CpuIm2ColFixture<half>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(im2col_shapes, make("DataType", DataType::F16), conv_args_small))
{
    if (CPUInfo::get().has_fp16())
    {
        // Validate output
        validate(Accessor(_target), _reference);
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}
FIXTURE_DATA_TEST_CASE(RunLarge,
                       CpuIm2ColFixture<half>,
                       framework::DatasetMode::NIGHTLY,
                       combine(concat(im2col_shapes, datasets::LargeShapes()),
                               make("DataType", DataType::F16),
                               conv_args))
{
    if (CPUInfo::get().has_fp16())
    {
        // Validate output
        validate(Accessor(_target), _reference);
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}
TEST_SUITE_END() // FP16

#endif /* ARM_COMPUTE_ENABLE_FP16 */

TEST_SUITE_END() // Float

TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       CpuIm2ColFixture<uint8_t>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(im2col_shapes, make("DataType", DataType::QASYMM8), conv_args_small))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunSmallWithChannelPadding,
                       CpuIm2ColWithChannelPadFixture<uint8_t>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(im2col_shapes, make("DataType", DataType::QASYMM8), conv_args_small_channel_padding))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge,
                       CpuIm2ColFixture<uint8_t>,
                       framework::DatasetMode::NIGHTLY,
                       combine(concat(im2col_shapes, datasets::LargeShapes()),
                               make("DataType", DataType::QASYMM8),
                               conv_args))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // QASYMM8

TEST_SUITE(SpecialCases)
TEST_CASE(PaddedChannelNHWC, framework::DatasetMode::PRECOMMIT)
{
    // Const data
    const TensorShape      src_shape   = TensorShape(7U, 27U, 13U);
    const DataType         data_type   = DataType::F32;
    const DataLayout       data_layout = DataLayout::NHWC;
    const bool             has_bias    = false;
    const unsigned int     num_groups  = 1;
    const Size2D           spatial_kernel(3, 3);
    const QuantizationInfo qinfo{};
    const PadStrideInfo    conv_info(1U, 1U, 0U, 0U);

    // Calculate destination shape
    TensorInfo src_info(src_shape, 1, data_type);
    src_info.set_data_layout(data_layout);
    const TensorShape dst_shape =
        compute_im2col_conv_shape(&src_info, spatial_kernel, conv_info, has_bias, Size2D(1U, 1U), false, num_groups);

    // Compute target
    Tensor src_target = create_tensor<Tensor>(src_shape, data_type, 1, qinfo, data_layout);
    Tensor dst_target = create_tensor<Tensor>(dst_shape, data_type, 1, qinfo);

    // Configure target function
    CpuIm2Col im2col_func;
    im2col_func.configure(src_target.info(), dst_target.info(), spatial_kernel, conv_info, has_bias);

    // Extend padding
    src_target.info()->extend_padding(PaddingSize(3, 5, 9, 1));
    dst_target.info()->extend_padding(PaddingSize(8, 1, 1, 3));

    // Validate and allocate tensors
    ARM_COMPUTE_EXPECT(src_target.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst_target.info()->is_resizable(), framework::LogLevel::ERRORS);

    src_target.allocator()->allocate();
    dst_target.allocator()->allocate();

    ARM_COMPUTE_EXPECT(!src_target.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(!dst_target.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Fill target source
    library->fill_tensor_uniform(Accessor(src_target), 0);

    ITensorPack pack = {{TensorType::ACL_SRC, &src_target}, {TensorType::ACL_DST, &dst_target}};
    // Run target function
    im2col_func.run(pack);

    // Calculate Reference
    SimpleTensor<float> src_ref{src_shape, data_type, 1, qinfo, data_layout};
    SimpleTensor<float> dst_ref{dst_shape, data_type, 1, qinfo, DataLayout::NCHW};

    // Fill reference source
    library->fill_tensor_uniform(src_ref, 0);

#ifndef DOXYGEN_SKIP_THIS
    // Run reference function
    reference::im2col(src_ref, dst_ref, spatial_kernel, conv_info, has_bias, num_groups, 0);
#endif // DOXYGEN_SKIP_THIS

    // Validate
    validate(Accessor(dst_target), dst_ref);
}
TEST_SUITE_END() // Special Cases
TEST_SUITE_END() // Im2Col
TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
