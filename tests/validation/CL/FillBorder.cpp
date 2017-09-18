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
#include "arm_compute/core/CL/kernels/CLFillBorderKernel.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "tests/CL/CLAccessor.h"
#include "tests/Globals.h"
#include "tests/datasets/BorderModeDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(CL)
TEST_SUITE(FillBorder)

// *INDENT-OFF*
// clang-format off
const auto PaddingSizesDataset = concat(concat(
                                 framework::dataset::make("PaddingSize", PaddingSize{ 0 }),
                                 framework::dataset::make("PaddingSize", PaddingSize{ 1, 0, 1, 2 })),
                                 framework::dataset::make("PaddingSize", PaddingSize{ 10 }));

const auto BorderSizesDataset  = framework::dataset::make("BorderSize", 0, 6);

DATA_TEST_CASE(FillBorder, framework::DatasetMode::ALL, combine(combine(combine(combine(
               datasets::SmallShapes(),
               datasets::BorderModes()),
               BorderSizesDataset),
               PaddingSizesDataset),
               framework::dataset::make("DataType", DataType::U8)),
               shape, border_mode, size, padding, data_type)
// clang-format on
// *INDENT-ON*
{
    BorderSize border_size{ static_cast<unsigned int>(size) };

    std::mt19937                           generator(library->seed());
    std::uniform_int_distribution<uint8_t> distribution_u8(0, 255);
    const uint8_t                          border_value = distribution_u8(generator);
    const uint8_t                          tensor_value = distribution_u8(generator);

    // Create tensors
    CLTensor src = create_tensor<CLTensor>(shape, data_type);

    src.info()->extend_padding(padding);

    // Allocate tensor
    src.allocator()->allocate();

    // Check padding is as required
    validate(src.info()->padding(), padding);

    // Fill tensor with constant value
    std::uniform_int_distribution<uint8_t> distribution{ tensor_value, tensor_value };
    library->fill(CLAccessor(src), distribution, 0);

    // Create and configure kernel
    CLFillBorderKernel fill_border;
    fill_border.configure(&src, border_size, border_mode, border_value);

    // Run kernel
    fill_border.run(fill_border.window(), CLScheduler::get().queue());

    // Validate border
    border_size.limit(padding);
    validate(CLAccessor(src), border_size, border_mode, &border_value);

    // Validate tensor
    validate(CLAccessor(src), &tensor_value);
}

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
