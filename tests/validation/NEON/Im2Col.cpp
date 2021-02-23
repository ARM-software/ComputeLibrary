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
#include "arm_compute/core/Types.h"
#include "src/core/NEON/kernels/NEIm2ColKernel.h"
#include "tests/NEON/Accessor.h"
#include "tests/NEON/Helper.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/Im2ColFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
const auto im2col_shapes = framework::dataset::make("Shape", { TensorShape{ 11U, 11U, 11U }, TensorShape{ 16U, 16U, 16U }, TensorShape{ 27U, 13U, 7U }, TensorShape{ 31U, 27U, 17U, 2U }, TensorShape{ 27U, 13U, 5U, 4U }, TensorShape{ 11U, 11U, 5U, 5U } });

const auto conv_filter_sizes = framework::dataset::make("KernelDims", { Size2D(3U, 3U), Size2D(3U, 1U), Size2D(1U, 5U), Size2D(5U, 5U), Size2D(7U, 7U) });
const auto conv_args         = combine(combine(combine(combine(conv_filter_sizes, framework::dataset::make("PadStride", { PadStrideInfo(1U, 1U, 0U, 0U), PadStrideInfo(1U, 1U, 1U, 1U), PadStrideInfo(2U, 2U, 0U, 2U) })),
                                                       framework::dataset::make("QuantizationInfo", QuantizationInfo(0.5f, 10))),
                                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                       framework::dataset::make("NumGroups", { 1 }));

const auto conv_filter_sizes_small = framework::dataset::make("KernelDims", { Size2D(3U, 3U), Size2D(3U, 1U), Size2D(1U, 5U) });
const auto conv_args_small         = combine(combine(combine(combine(conv_filter_sizes_small, framework::dataset::make("PadStride", { PadStrideInfo(1U, 1U, 0U, 0U), PadStrideInfo(1U, 1U, 1U, 1U) })),
                                                             framework::dataset::make("QuantizationInfo", QuantizationInfo(0.5f, 10))),
                                                     framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                             framework::dataset::make("NumGroups", { 1 }));
} // namespace
TEST_SUITE(NEON)
TEST_SUITE(Im2Col)

using NEIm2Col = NESynthetizeFunction<NEIm2ColKernel>;

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
               framework::dataset::make("InputInfo", { TensorInfo(TensorShape(10U, 12U, 2U), 1, DataType::U8),      // Unsupported data type
                                                       TensorInfo(TensorShape(10U, 12U, 2U), 1, DataType::F32),     // Mismatching data type
                                                       TensorInfo(TensorShape(10U, 12U, 2U), 1, DataType::QASYMM8), // Bias not supported with QASYMM8
                                                       TensorInfo(TensorShape(10U, 12U, 2U), 1, DataType::QASYMM8), // Mismatching shapes
                                                       TensorInfo(TensorShape(10U, 12U, 2U, 2U), 1, DataType::QASYMM8),
                                                     }),
               framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(3U, 4U, 10U, 2U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(3U, 4U, 10U, 2U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(3U, 3U, 10U, 2U), 1, DataType::QASYMM8),
                                                       TensorInfo(TensorShape(3U, 4U, 10U, 2U), 1, DataType::QASYMM8),
                                                       TensorInfo(TensorShape(18U, 80U, 1U, 2U), 1, DataType::QASYMM8),
                                                     })),
               framework::dataset::make("HasBias", { true, true, true, false, false })),
               framework::dataset::make("Expected", { false, false, false, false, true })),
               input_info, output_info, has_bias, expected)
{
    bool status = bool(NEIm2Col::validate(&input_info, &output_info, Size2D(3U, 3U), PadStrideInfo(), has_bias));
    ARM_COMPUTE_EXPECT(status == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using NEIm2ColFixture = Im2ColValidationFixture<Tensor, Accessor, NEIm2Col, T, false>;

TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEIm2ColFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(im2col_shapes, framework::dataset::make("DataType", DataType::F32)),
                                                                                                    conv_args_small))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEIm2ColFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(concat(im2col_shapes, datasets::LargeShapes()), framework::dataset::make("DataType",
                                                                                                          DataType::F32)),
                                                                                                  conv_args))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // FP32

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEIm2ColFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(im2col_shapes, framework::dataset::make("DataType", DataType::F16)),
                                                                                                   conv_args_small))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEIm2ColFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(concat(im2col_shapes, datasets::LargeShapes()), framework::dataset::make("DataType",
                                                                                                         DataType::F16)),
                                                                                                 conv_args))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // FP16

#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

TEST_SUITE_END() // Float

TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, NEIm2ColFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(im2col_shapes, framework::dataset::make("DataType", DataType::QASYMM8)),
                                                                                                      conv_args_small))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEIm2ColFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(concat(im2col_shapes, datasets::LargeShapes()),
                                                                                                            framework::dataset::make("DataType", DataType::QASYMM8)),
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
    const TensorShape dst_shape = compute_im2col_conv_shape(&src_info, spatial_kernel, conv_info, has_bias, Size2D(1U, 1U), false, num_groups);

    // Compute target
    Tensor src_target = create_tensor<Tensor>(src_shape, data_type, 1, qinfo, data_layout);
    Tensor dst_target = create_tensor<Tensor>(dst_shape, data_type, 1, qinfo);

    // Configure target function
    NEIm2Col im2col_func;
    im2col_func.configure(&src_target, &dst_target, spatial_kernel, conv_info, has_bias);

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

    // Run target function
    im2col_func.run();

    // Calculate Reference
    SimpleTensor<float> src_ref{ src_shape, data_type, 1, qinfo, data_layout };
    SimpleTensor<float> dst_ref{ dst_shape, data_type, 1, qinfo, DataLayout::NCHW };

    // Fill reference source
    library->fill_tensor_uniform(src_ref, 0);

#ifndef DOXYGEN_SKIP_THIS
    // Run reference function
    reference::im2col(src_ref, dst_ref, spatial_kernel, conv_info, has_bias, num_groups);
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
