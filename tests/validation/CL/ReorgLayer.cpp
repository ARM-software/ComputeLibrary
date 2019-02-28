/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#include "arm_compute/runtime/CL/functions/CLReorgLayer.h"
#include "tests/CL/CLAccessor.h"
#include "tests/CL/Helper.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ReorgLayerDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/ReorgLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(CL)
TEST_SUITE(ReorgLayer)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
               framework::dataset::make("InputInfo", { TensorInfo(TensorShape(10U, 12U, 1U, 2U), 1, DataType::S64),    // Wrong output tensor
                                                       TensorInfo(TensorShape(10U, 12U, 1U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(10U, 12U, 1U, 2U), 1, DataType::F32),    // Wrong output tensor
                                                       TensorInfo(TensorShape(3U, 12U, 4U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(3U, 12U, 4U, 2U), 1, DataType::F32),     // Wrong data type
                                                     }),
               framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(3U, 4U, 10U, 2U), 1, DataType::S64),
                                                       TensorInfo(TensorShape(5U, 6U, 4U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(5U, 6U, 2, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(1U, 4U, 36U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(1U, 4U, 36U, 2U), 1, DataType::F16),
                                                     })),
               framework::dataset::make("Stride", { 2, 2, 4, 3 })),
               framework::dataset::make("Expected", { false, true, false, true, false })),
               input_info, output_info, stride, expected)
{
    bool status = bool(CLReorgLayer::validate(&input_info, &output_info, stride));
    ARM_COMPUTE_EXPECT(status == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(datasets::SmallReorgLayerDataset(),
                                                                           framework::dataset::make("DataType", { DataType::F32, DataType::F16, DataType::QASYMM8 })),
                                                                   framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
               shape, stride, data_type, data_layout)
{
    // Permute the tensor shape in case of NHWC data layout
    TensorShape shape_to_use = shape;
    if(data_layout == DataLayout::NHWC)
    {
        permute(shape_to_use, PermutationVector(2U, 0U, 1U));
    }

    // Create tensors
    CLTensor src = create_tensor<CLTensor>(shape_to_use, data_type, 1, QuantizationInfo(), data_layout);
    CLTensor dst;

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    CLReorgLayer reorg_layer;

    // Auto-initialize the output within the function
    reorg_layer.configure(&src, &dst, stride);

    // Validate valid region
    const ValidRegion src_valid_region = shape_to_valid_region(shape_to_use);
    const ValidRegion dst_valid_region = shape_to_valid_region(dst.info()->tensor_shape());
    validate(src.info()->valid_region(), src_valid_region);
    validate(dst.info()->valid_region(), dst_valid_region);

    // Validate padding
    const int         step        = 1;
    const PaddingSize src_padding = PaddingCalculator(shape_to_use.x(), step).required_padding();
    const PaddingSize dst_padding = PaddingCalculator(dst.info()->tensor_shape().x(), step).required_padding();
    validate(src.info()->padding(), src_padding);
    validate(dst.info()->padding(), dst_padding);
}

template <typename T>
using CLReorgLayerFixture = ReorgLayerValidationFixture<CLTensor, CLAccessor, CLReorgLayer, T>;

TEST_SUITE(S32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLReorgLayerFixture<int32_t>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallReorgLayerDataset(), framework::dataset::make("DataType",
                                                                                                                  DataType::S32)),
                                                                                                          framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLReorgLayerFixture<int32_t>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeReorgLayerDataset(), framework::dataset::make("DataType",
                                                                                                                DataType::S32)),
                                                                                                        framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // S32

TEST_SUITE(S16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLReorgLayerFixture<int16_t>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallReorgLayerDataset(), framework::dataset::make("DataType",
                                                                                                                  DataType::S16)),
                                                                                                          framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLReorgLayerFixture<int16_t>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeReorgLayerDataset(), framework::dataset::make("DataType",
                                                                                                                DataType::S16)),
                                                                                                        framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // S16

TEST_SUITE(S8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLReorgLayerFixture<int8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallReorgLayerDataset(), framework::dataset::make("DataType",
                                                                                                                 DataType::S8)),
                                                                                                         framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLReorgLayerFixture<int8_t>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeReorgLayerDataset(), framework::dataset::make("DataType", DataType::S8)),
                                                                                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // S8

TEST_SUITE_END() // ReorgLayer
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
