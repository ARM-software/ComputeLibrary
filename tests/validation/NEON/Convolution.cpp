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
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/functions/NEConvolution.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
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
/* Convolution3x3 */
constexpr unsigned int filter_size_3x3 = 3;                  /* Size of the kernel/filter in number of elements. */
constexpr BorderSize   border_size_3x3(filter_size_3x3 / 2); /* Border size of the kernel/filter around its central element. */

/* Convolution5x5 */
constexpr unsigned int filter_size_5x5 = 5;                  /* Size of the kernel/filter in number of elements. */
constexpr BorderSize   border_size_5x5(filter_size_5x5 / 2); /* Border size of the kernel/filter around its central element. */

/* Convolution7x7 */
constexpr unsigned int filter_size_7x7 = 7;                  /* Size of the kernel/filter in number of elements. */
constexpr BorderSize   border_size_7x7(filter_size_7x7 / 2); /* Border size of the kernel/filter around its central element. */

/* Convolutionx */
constexpr unsigned int filter_size_9x9 = 9;                  /* Size of the kernel/filter in number of elements. */
constexpr BorderSize   border_size_9x9(filter_size_9x9 / 2); /* Border size of the kernel/filter around its central element. */

/** Create conv matrix with filter size, and fill them with random value
 *
 * @param[in/out] conv        Convolution matrix to be filled with random int16_t
 * @param[in]     filter_size Filter Size.
 */
void create_conv(int16_t *conv, const unsigned int filter_size)
{
    std::mt19937                           gen(library->seed());
    std::uniform_int_distribution<int16_t> distribution_int16(-32768, 32767);

    for(unsigned int i = 0; i < filter_size * filter_size; ++i)
    {
        conv[i] = distribution_int16(gen);
    }
}
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(CustomConvolution)
TEST_SUITE(CustomConvolution3x3)

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(concat(datasets::SmallShapes(), datasets::LargeShapes()), framework::dataset::make("DataType", DataType::U8)),
                                                                   datasets::BorderModes()),
               shape, data_type, border_mode)
{
    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, data_type);
    Tensor dst = create_tensor<Tensor>(shape, data_type);

    // Create conv matrix
    int16_t conv[9];
    create_conv(conv, filter_size_3x3);

    // Generate random scale value between 0 and 255.
    std::mt19937                           gen(library->seed());
    std::uniform_int_distribution<uint8_t> distribution(0, 255);
    uint32_t                               scale = distribution(gen);

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    NEConvolution3x3 convolution;
    convolution.configure(&src, &dst, conv, scale, border_mode);

    // Validate valid region
    const ValidRegion dst_valid_region = shape_to_valid_region(shape, (border_mode == BorderMode::UNDEFINED), border_size_3x3);
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
using NEConvolutionFixture = ConvolutionValidationFixture<Tensor, Accessor, NEConvolution3x3, T, filter_size_3x3>;

FIXTURE_DATA_TEST_CASE(RunSmall, NEConvolutionFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallShapes(), framework::dataset::make("DataType",
                                                                                                                   DataType::U8)),
                                                                                                           datasets::BorderModes()))
{
    // Validate output
    validate(Accessor(_target), _reference, shape_to_valid_region(_reference.shape(), (_border_mode == BorderMode::UNDEFINED), border_size_3x3));
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEConvolutionFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeShapes(), framework::dataset::make("DataType",
                                                                                                                 DataType::U8)),
                                                                                                         datasets::BorderModes()))
{
    // Validate output
    validate(Accessor(_target), _reference, shape_to_valid_region(_reference.shape(), (_border_mode == BorderMode::UNDEFINED), border_size_3x3));
}
TEST_SUITE_END() /* Custom Convolution3x3 */

TEST_SUITE(CustomConvolution5x5)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(concat(datasets::SmallShapes(), datasets::LargeShapes()), framework::dataset::make("DataType", DataType::U8)),
                                                                   datasets::BorderModes()),
               shape, data_type, border_mode)
{
    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, data_type);
    Tensor dst = create_tensor<Tensor>(shape, data_type);

    // Create conv matrix
    int16_t conv[25];
    create_conv(conv, filter_size_5x5);

    // Generate random scale value between 0 and 255.
    std::mt19937                           gen(library->seed());
    std::uniform_int_distribution<uint8_t> distribution(0, 255);
    uint32_t                               scale = distribution(gen);

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    NEConvolution5x5 convolution;
    convolution.configure(&src, &dst, conv, scale, border_mode);

    // Validate valid region
    const ValidRegion dst_valid_region = shape_to_valid_region(shape, (border_mode == BorderMode::UNDEFINED), border_size_5x5);
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
using NEConvolutionFixture = ConvolutionValidationFixture<Tensor, Accessor, NEConvolution5x5, T, filter_size_5x5>;

FIXTURE_DATA_TEST_CASE(RunSmall, NEConvolutionFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallShapes(), framework::dataset::make("DataType",
                                                                                                                   DataType::U8)),
                                                                                                           datasets::BorderModes()))
{
    // Validate output
    validate(Accessor(_target), _reference, shape_to_valid_region(_reference.shape(), (_border_mode == BorderMode::UNDEFINED), border_size_5x5));
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEConvolutionFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeShapes(), framework::dataset::make("DataType",
                                                                                                                 DataType::U8)),
                                                                                                         datasets::BorderModes()))
{
    // Validate output
    validate(Accessor(_target), _reference, shape_to_valid_region(_reference.shape(), (_border_mode == BorderMode::UNDEFINED), border_size_5x5));
}
TEST_SUITE_END() /* Custom Convolution 5x5 */

TEST_SUITE(CustomConvolution7x7)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(concat(datasets::SmallShapes(), datasets::LargeShapes()), framework::dataset::make("DataType", DataType::U8)),
                                                                   datasets::BorderModes()),
               shape, data_type, border_mode)
{
    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, data_type);
    Tensor dst = create_tensor<Tensor>(shape, data_type);

    // Create conv matrix
    int16_t conv[49];
    create_conv(conv, filter_size_7x7);

    // Generate random scale value between 0 and 255.
    std::mt19937                           gen(library->seed());
    std::uniform_int_distribution<uint8_t> distribution(0, 255);
    uint32_t                               scale = distribution(gen);

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    NEConvolution7x7 convolution;
    convolution.configure(&src, &dst, conv, scale, border_mode);

    // Validate valid region
    const ValidRegion dst_valid_region = shape_to_valid_region(shape, (border_mode == BorderMode::UNDEFINED), border_size_7x7);
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
using NEConvolutionFixture = ConvolutionValidationFixture<Tensor, Accessor, NEConvolution7x7, T, filter_size_7x7>;

FIXTURE_DATA_TEST_CASE(RunSmall, NEConvolutionFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallShapes(), framework::dataset::make("DataType",
                                                                                                                   DataType::U8)),
                                                                                                           datasets::BorderModes()))
{
    // Validate output
    validate(Accessor(_target), _reference, shape_to_valid_region(_reference.shape(), (_border_mode == BorderMode::UNDEFINED), border_size_7x7));
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEConvolutionFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeShapes(), framework::dataset::make("DataType",
                                                                                                                 DataType::U8)),
                                                                                                         datasets::BorderModes()))
{
    // Validate output
    validate(Accessor(_target), _reference, shape_to_valid_region(_reference.shape(), (_border_mode == BorderMode::UNDEFINED), border_size_7x7));
}
TEST_SUITE_END() /* Custom Convolution 7x7 */

TEST_SUITE(CustomConvolution9x9)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(concat(datasets::SmallShapes(), datasets::LargeShapes()), framework::dataset::make("DataType", DataType::U8)),
                                                                   datasets::BorderModes()),
               shape, data_type, border_mode)
{
    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, data_type);
    Tensor dst = create_tensor<Tensor>(shape, data_type);

    // Create conv matrix
    int16_t conv[81];
    create_conv(conv, filter_size_9x9);

    // Generate random scale value between 0 and 255.
    std::mt19937                           gen(library->seed());
    std::uniform_int_distribution<uint8_t> distribution(0, 255);
    uint32_t                               scale = distribution(gen);

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    NEConvolution9x9 convolution;
    convolution.configure(&src, &dst, conv, scale, border_mode);

    // Validate valid region
    const ValidRegion dst_valid_region = shape_to_valid_region(shape, (border_mode == BorderMode::UNDEFINED), border_size_9x9);
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
using NEConvolutionFixture = ConvolutionValidationFixture<Tensor, Accessor, NEConvolution9x9, T, filter_size_9x9>;

FIXTURE_DATA_TEST_CASE(RunSmall, NEConvolutionFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallShapes(), framework::dataset::make("DataType",
                                                                                                                   DataType::U8)),
                                                                                                           datasets::BorderModes()))
{
    // Validate output
    validate(Accessor(_target), _reference, shape_to_valid_region(_reference.shape(), (_border_mode == BorderMode::UNDEFINED), border_size_9x9));
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEConvolutionFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeShapes(), framework::dataset::make("DataType",
                                                                                                                 DataType::U8)),
                                                                                                         datasets::BorderModes()))
{
    // Validate output
    validate(Accessor(_target), _reference, shape_to_valid_region(_reference.shape(), (_border_mode == BorderMode::UNDEFINED), border_size_9x9));
}
TEST_SUITE_END() /* Custom Convolution 9x9 */
TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
