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
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "utils/TypePrinter.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(UNIT)
TEST_SUITE(TensorInfo)

// *INDENT-OFF*
// clang-format off
/** Validates TensorInfo Autopadding */
DATA_TEST_CASE(AutoPadding, framework::DatasetMode::ALL, zip(zip(zip(
               framework::dataset::make("TensorShape", {
               TensorShape{},
               TensorShape{ 10U },
               TensorShape{ 10U, 10U },
               TensorShape{ 10U, 10U, 10U },
               TensorShape{ 10U, 10U, 10U, 10U },
               TensorShape{ 10U, 10U, 10U, 10U, 10U },
               TensorShape{ 10U, 10U, 10U, 10U, 10U, 10U }}),
               framework::dataset::make("PaddingSize", {
               PaddingSize{ 0, 0, 0, 0 },
               PaddingSize{ 0, 36, 0, 4 },
               PaddingSize{ 4, 36, 4, 4 },
               PaddingSize{ 4, 36, 4, 4 },
               PaddingSize{ 4, 36, 4, 4 },
               PaddingSize{ 4, 36, 4, 4 },
               PaddingSize{ 4, 36, 4, 4 }})),
               framework::dataset::make("Strides", {
               Strides{},
               Strides{ 1U, 50U },
               Strides{ 1U, 50U },
               Strides{ 1U, 50U, 900U },
               Strides{ 1U, 50U, 900U, 9000U },
               Strides{ 1U, 50U, 900U, 9000U, 90000U },
               Strides{ 1U, 50U, 900U, 9000U, 90000U, 900000U }})),
               framework::dataset::make("Offset", { 0U, 4U, 204U, 204U, 204U, 204U, 204U })),
               shape, auto_padding, strides, offset)
{
    TensorInfo info{ shape, Format::U8 };

    ARM_COMPUTE_EXPECT(!info.has_padding(), framework::LogLevel::ERRORS);

    info.auto_padding();

    validate(info.padding(), auto_padding);

    ARM_COMPUTE_EXPECT(compare_dimensions(info.strides_in_bytes(), strides), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(info.offset_first_element_in_bytes() == offset, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

/** Validates that TensorInfo is clonable */
TEST_CASE(Clone, framework::DatasetMode::ALL)
{
    // Create tensor info
    TensorInfo info(TensorShape(23U, 17U, 3U), // tensor shape
                    1,                         // number of channels
                    DataType::F32);            // data type

    // Get clone of current tensor info
    std::unique_ptr<ITensorInfo> info_clone = info.clone();
    ARM_COMPUTE_EXPECT(info_clone != nullptr, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(info_clone->total_size() == info.total_size(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(info_clone->num_channels() == info.num_channels(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(info_clone->data_type() == info.data_type(), framework::LogLevel::ERRORS);
}

/** Validates that TensorInfo can chain multiple set commands */
TEST_CASE(TensorInfoBuild, framework::DatasetMode::ALL)
{
    // Create tensor info
    TensorInfo info(TensorShape(23U, 17U, 3U), // tensor shape
                    1,                         // number of channels
                    DataType::F32);            // data type

    // Update data type and number of channels
    info.set_data_type(DataType::S32).set_num_channels(3);
    ARM_COMPUTE_EXPECT(info.data_type() == DataType::S32, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(info.num_channels() == 3, framework::LogLevel::ERRORS);

    // Update data type channels and set fixed point position
    info.set_data_type(DataType::QS8).set_num_channels(1).set_fixed_point_position(3);
    ARM_COMPUTE_EXPECT(info.data_type() == DataType::QS8, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(info.num_channels() == 1, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(info.fixed_point_position() == 3, framework::LogLevel::ERRORS);

    // Update data type and set quantization info
    info.set_data_type(DataType::QASYMM8).set_quantization_info(QuantizationInfo(0.5f, 15));
    ARM_COMPUTE_EXPECT(info.data_type() == DataType::QASYMM8, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(info.quantization_info() == QuantizationInfo(0.5f, 15), framework::LogLevel::ERRORS);

    // Update tensor shape
    info.set_tensor_shape(TensorShape(13U, 15U));
    ARM_COMPUTE_EXPECT(info.tensor_shape() == TensorShape(13U, 15U), framework::LogLevel::ERRORS);
}

TEST_SUITE_END() // TensorInfoValidation
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
