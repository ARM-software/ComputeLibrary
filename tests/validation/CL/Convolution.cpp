/*
 * Copyright (c) 2017, 2018 ARM Limited.
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
#include "arm_compute/runtime/CL/functions/CLConvolution.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/CL/CLAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/BorderModeDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/ConvolutionFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
} // namespace

TEST_SUITE(CL)
TEST_SUITE(CustomConvolution)
TEST_SUITE(CustomConvolutionSquare)
TEST_SUITE(CustomConvolution3x3)

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(combine(concat(datasets::SmallShapes(), datasets::LargeShapes()), framework::dataset::make("DataType", DataType::U8)),
                                                                           datasets::BorderModes()),
                                                                   framework::dataset::make("filter_size", { 3 })),
               shape, data_type, border_mode, filter_size)
{
    // Create tensors
    CLTensor src = create_tensor<CLTensor>(shape, data_type);
    CLTensor dst = create_tensor<CLTensor>(shape, data_type);

    // Create conv matrix
    int16_t conv[9];

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    CLConvolution3x3 convolution;
    convolution.configure(&src, &dst, conv, 0, border_mode);

    // Validate valid region
    const ValidRegion dst_valid_region = shape_to_valid_region(shape, (border_mode == BorderMode::UNDEFINED), BorderSize(filter_size / 2));
    validate(dst.info()->valid_region(), dst_valid_region);

    // Validate padding
    PaddingCalculator calculator(shape.x(), 8);
    calculator.set_border_size(1);
    calculator.set_border_mode(border_mode);

    const PaddingSize dst_padding = calculator.required_padding();

    calculator.set_accessed_elements(16);
    calculator.set_access_offset(-1);

    const PaddingSize src_padding = calculator.required_padding();

    validate(src.info()->padding(), src_padding);
    validate(dst.info()->padding(), dst_padding);
}

template <typename T>
using CLConvolutionFixture = ConvolutionSquareValidationFixture<CLTensor, CLAccessor, CLConvolution3x3, T>;

FIXTURE_DATA_TEST_CASE(RunSmall, CLConvolutionFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), framework::dataset::make("DataType",
                                                                                                                   DataType::U8)),
                                                                                                                   datasets::BorderModes()),
                                                                                                           framework::dataset::make("filter_size", { 3 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, shape_to_valid_region(_reference.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(_height / 2, _width / 2)));
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLConvolutionFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), framework::dataset::make("DataType",
                                                                                                                 DataType::U8)),
                                                                                                                 datasets::BorderModes()),
                                                                                                         framework::dataset::make("filter_size", { 3 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, shape_to_valid_region(_reference.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(_height / 2, _width / 2)));
}
TEST_SUITE_END() /* Custom_Convolution 3x3 */

TEST_SUITE(CustomConvolution5x5)

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(combine(concat(datasets::SmallShapes(), datasets::LargeShapes()), framework::dataset::make("DataType", DataType::U8)),
                                                                           datasets::BorderModes()),
                                                                   framework::dataset::make("filter_size", { 5 })),
               shape, data_type, border_mode, filter_size)
{
    // Create tensors
    CLTensor src = create_tensor<CLTensor>(shape, data_type);
    CLTensor dst = create_tensor<CLTensor>(shape, data_type);

    // Create conv matrix
    int16_t conv[25];

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    CLConvolution5x5 convolution;
    convolution.configure(&src, &dst, conv, 0, border_mode);

    // Validate valid region
    const ValidRegion dst_valid_region = shape_to_valid_region(shape, (border_mode == BorderMode::UNDEFINED), BorderSize(filter_size / 2));
    validate(dst.info()->valid_region(), dst_valid_region);

    // Validate padding
    PaddingCalculator calculator(shape.x(), 8);
    calculator.set_border_size(2);
    calculator.set_border_mode(border_mode);

    const PaddingSize dst_padding = calculator.required_padding();

    calculator.set_accessed_elements(16);
    calculator.set_access_offset(-2);

    const PaddingSize src_padding = calculator.required_padding();

    validate(src.info()->padding(), src_padding);
    validate(dst.info()->padding(), dst_padding);
}

template <typename T>
using CLConvolutionFixture = ConvolutionSquareValidationFixture<CLTensor, CLAccessor, CLConvolution5x5, T>;
FIXTURE_DATA_TEST_CASE(RunSmall, CLConvolutionFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), framework::dataset::make("DataType",
                                                                                                                   DataType::U8)),
                                                                                                                   datasets::BorderModes()),
                                                                                                           framework::dataset::make("filter_size", { 5 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, shape_to_valid_region(_reference.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(_height / 2, _width / 2)));
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLConvolutionFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), framework::dataset::make("DataType",
                                                                                                                 DataType::U8)),
                                                                                                                 datasets::BorderModes()),
                                                                                                         framework::dataset::make("filter_size", { 5 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, shape_to_valid_region(_reference.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(_height / 2, _width / 2)));
}
TEST_SUITE_END() /* Custom Convolution 5x5 */

TEST_SUITE(CustomConvolution7x7)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(combine(concat(datasets::SmallShapes(), datasets::LargeShapes()), framework::dataset::make("DataType", DataType::U8)),
                                                                           datasets::BorderModes()),
                                                                   framework::dataset::make("filter_size", { 7 })),
               shape, data_type, border_mode, filter_size)
{
    // Create tensors
    CLTensor src = create_tensor<CLTensor>(shape, data_type);
    CLTensor dst = create_tensor<CLTensor>(shape, data_type);

    // Create conv matrix
    int16_t conv[49];

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    CLConvolution7x7 convolution;
    convolution.configure(&src, &dst, conv, 0, border_mode);

    // Validate valid region
    const ValidRegion dst_valid_region = shape_to_valid_region(shape, (border_mode == BorderMode::UNDEFINED), BorderSize(filter_size / 2));
    validate(dst.info()->valid_region(), dst_valid_region);

    // Validate padding
    PaddingCalculator calculator(shape.x(), 8);
    calculator.set_border_size(3);
    calculator.set_border_mode(border_mode);

    const PaddingSize dst_padding = calculator.required_padding();

    calculator.set_accessed_elements(16);
    calculator.set_access_offset(-3);

    const PaddingSize src_padding = calculator.required_padding();

    validate(src.info()->padding(), src_padding);
    validate(dst.info()->padding(), dst_padding);
}

template <typename T>
using CLConvolutionFixture = ConvolutionSquareValidationFixture<CLTensor, CLAccessor, CLConvolution7x7, T>;

FIXTURE_DATA_TEST_CASE(RunSmall, CLConvolutionFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), framework::dataset::make("DataType",
                                                                                                                   DataType::U8)),
                                                                                                                   datasets::BorderModes()),
                                                                                                           framework::dataset::make("filter_size", { 7 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, shape_to_valid_region(_reference.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(_height / 2, _width / 2)));
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLConvolutionFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), framework::dataset::make("DataType",
                                                                                                                 DataType::U8)),
                                                                                                                 datasets::BorderModes()),
                                                                                                         framework::dataset::make("filter_size", { 7 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, shape_to_valid_region(_reference.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(_height / 2, _width / 2)));
}
TEST_SUITE_END() /* Custom Convolution 7x7 */

TEST_SUITE(CustomConvolution9x9)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(combine(concat(datasets::SmallShapes(), datasets::LargeShapes()), framework::dataset::make("DataType", DataType::U8)),
                                                                           datasets::BorderModes()),
                                                                   framework::dataset::make("filter_size", { 9 })),
               shape, data_type, border_mode, filter_size)
{
    // Create tensors
    CLTensor src = create_tensor<CLTensor>(shape, data_type);
    CLTensor dst = create_tensor<CLTensor>(shape, data_type);

    // Create conv matrix
    int16_t conv[81];

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    CLConvolution9x9 convolution;
    convolution.configure(&src, &dst, conv, 0, border_mode);

    // Validate valid region
    const ValidRegion dst_valid_region = shape_to_valid_region(shape, (border_mode == BorderMode::UNDEFINED), BorderSize(filter_size / 2));
    validate(dst.info()->valid_region(), dst_valid_region);

    // Validate padding
    PaddingCalculator calculator(shape.x(), 8);
    calculator.set_border_size(4);
    calculator.set_border_mode(border_mode);

    const PaddingSize dst_padding = calculator.required_padding();

    calculator.set_accessed_elements(16);
    calculator.set_access_offset(-4);

    const PaddingSize src_padding = calculator.required_padding();

    validate(src.info()->padding(), src_padding);
    validate(dst.info()->padding(), dst_padding);
}

template <typename T>
using CLConvolutionFixture = ConvolutionSquareValidationFixture<CLTensor, CLAccessor, CLConvolution9x9, T>;

FIXTURE_DATA_TEST_CASE(RunSmall, CLConvolutionFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), framework::dataset::make("DataType",
                                                                                                                   DataType::U8)),
                                                                                                                   datasets::BorderModes()),
                                                                                                           framework::dataset::make("filter_size", { 9 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, shape_to_valid_region(_reference.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(_height / 2, _width / 2)));
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLConvolutionFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), framework::dataset::make("DataType",
                                                                                                                 DataType::U8)),
                                                                                                                 datasets::BorderModes()),
                                                                                                         framework::dataset::make("filter_size", { 9 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, shape_to_valid_region(_reference.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(_height / 2, _width / 2)));
}
TEST_SUITE_END() /* Custom Convolution 9x9 */
TEST_SUITE_END() /* Custom Convolution Square */

TEST_SUITE(CustomConvolutionRectangle)

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(combine(combine(concat(datasets::SmallShapes(), datasets::LargeShapes()), framework::dataset::make("DataType",
                                                                                           DataType::U8)),
                                                                                   datasets::BorderModes()),
                                                                           framework::dataset::make("filter_width", { 3, 5, 7, 9 })),
                                                                   framework::dataset::make("filter_height", { 3, 5, 7, 9 })),
               shape, data_type, border_mode, filter_width, filter_height)
{
    // Create tensors
    CLTensor src = create_tensor<CLTensor>(shape, data_type);
    CLTensor dst = create_tensor<CLTensor>(shape, data_type);

    // Create conv matrix
    int16_t conv[filter_width * filter_height];

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    CLConvolutionRectangle convolution;
    convolution.configure(&src, &dst, conv, filter_width, filter_height, 1, border_mode);

    // Validate valid region
    const ValidRegion dst_valid_region = shape_to_valid_region(shape, (border_mode == BorderMode::UNDEFINED), BorderSize(filter_height / 2, filter_width / 2));
    validate(dst.info()->valid_region(), dst_valid_region);

    // Validate padding
    PaddingCalculator calculator(shape.x(), 8);
    calculator.set_border_size(filter_width / 2);
    calculator.set_border_mode(border_mode);

    const PaddingSize dst_padding = calculator.required_padding();

    calculator.set_accessed_elements(16);
    calculator.set_access_offset(-(filter_width / 2));

    const PaddingSize width_padding = calculator.required_padding();

    calculator.set_border_size(filter_height / 2);
    calculator.set_access_offset(-(filter_height / 2));
    const PaddingSize height_padding = calculator.required_padding();

    validate(src.info()->padding(), width_padding, height_padding);
    validate(dst.info()->padding(), dst_padding);
}

template <typename T>
using CLConvolutionFixture = ConvolutionRectangleValidationFixture<CLTensor, CLAccessor, CLConvolutionRectangle, T>;

FIXTURE_DATA_TEST_CASE(RunSmall, CLConvolutionFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::SmallShapes(), framework::dataset::make("DataType",
                                                                                                                   DataType::U8)),
                                                                                                                   datasets::BorderModes()),
                                                                                                                   framework::dataset::make("filter_width", { 3, 5, 7, 9 })),
                                                                                                           framework::dataset::make("filter_height", { 3, 5, 7, 9 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, shape_to_valid_region(_reference.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(_height / 2, _width / 2)));
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLConvolutionFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::LargeShapes(), framework::dataset::make("DataType",
                                                                                                                 DataType::U8)),
                                                                                                                 datasets::BorderModes()),
                                                                                                                 framework::dataset::make("filter_width", { 3, 5, 7, 9 })),
                                                                                                         framework::dataset::make("filter_height", { 3, 5, 7, 9 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, shape_to_valid_region(_reference.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(_height / 2, _width / 2)));
}
TEST_SUITE_END() /* Custom Convolution Rectangle */

TEST_SUITE(CustomConvolutionSeparable)
TEST_SUITE(CustomConvolutionSeparable5x5)
template <typename T>
using CLConvolutionFixture = ConvolutionSeparableValidationFixture<CLTensor, CLAccessor, CLConvolution5x5, T>;

FIXTURE_DATA_TEST_CASE(RunSmall, CLConvolutionFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), framework::dataset::make("DataType",
                                                                                                                   DataType::U8)),
                                                                                                                   datasets::BorderModes()),
                                                                                                           framework::dataset::make("filter_size", { 5 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, shape_to_valid_region(_reference.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(_height / 2, _width / 2)));
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLConvolutionFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), framework::dataset::make("DataType",
                                                                                                                 DataType::U8)),
                                                                                                                 datasets::BorderModes()),
                                                                                                         framework::dataset::make("filter_size", { 5 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, shape_to_valid_region(_reference.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(_height / 2, _width / 2)));
}
TEST_SUITE_END() /* Custom Convolution Separable 5x5 */

TEST_SUITE(CustomConvolutionSeparablex7x7)
template <typename T>
using CLConvolutionFixture = ConvolutionSeparableValidationFixture<CLTensor, CLAccessor, CLConvolution7x7, T>;

FIXTURE_DATA_TEST_CASE(RunSmall, CLConvolutionFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), framework::dataset::make("DataType",
                                                                                                                   DataType::U8)),
                                                                                                                   datasets::BorderModes()),
                                                                                                           framework::dataset::make("filter_size", { 7 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, shape_to_valid_region(_reference.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(_height / 2, _width / 2)));
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLConvolutionFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), framework::dataset::make("DataType",
                                                                                                                 DataType::U8)),
                                                                                                                 datasets::BorderModes()),
                                                                                                         framework::dataset::make("filter_size", { 7 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, shape_to_valid_region(_reference.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(_height / 2, _width / 2)));
}
TEST_SUITE_END() /* Custom Convolution Separable 7x7 */

TEST_SUITE(CustomConvolutionSeparable9x9)
template <typename T>
using CLConvolutionFixture = ConvolutionSeparableValidationFixture<CLTensor, CLAccessor, CLConvolution9x9, T>;

FIXTURE_DATA_TEST_CASE(RunSmall, CLConvolutionFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), framework::dataset::make("DataType",
                                                                                                                   DataType::U8)),
                                                                                                                   datasets::BorderModes()),
                                                                                                           framework::dataset::make("filter_size", { 9 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, shape_to_valid_region(_reference.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(_height / 2, _width / 2)));
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLConvolutionFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), framework::dataset::make("DataType",
                                                                                                                 DataType::U8)),
                                                                                                                 datasets::BorderModes()),
                                                                                                         framework::dataset::make("filter_size", { 9 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, shape_to_valid_region(_reference.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(_height / 2, _width / 2)));
}
TEST_SUITE_END() /* Custom Convolution Separable 9x9 */

TEST_SUITE_END() /* Custom Convolution Separable */
TEST_SUITE_END() /* Custom Convolution */
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
