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
#include "CL/CLAccessor.h"
#include "TypePrinter.h"
#include "Utils.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/validation_old/Datasets.h"
#include "tests/validation_old/Validation.h"

#include "arm_compute/core/CL/kernels/CLFillBorderKernel.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"

#include "tests/validation_old/boost_wrapper.h"

#include <random>
#include <string>

using namespace arm_compute;
using namespace arm_compute::test;
using namespace arm_compute::test::validation;

#ifndef DOXYGEN_SKIP_THIS
BOOST_AUTO_TEST_SUITE(CL)

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(FillBorder, BorderModes() * boost::unit_test::data::make({ PaddingSize{ 0 }, PaddingSize{ 1, 0, 1, 2 }, PaddingSize{ 10 } }), border_mode, padding)
{
    constexpr uint8_t border_value = 42U;
    constexpr uint8_t tensor_value = 89U;
    BorderSize        border_size{ 5 };

    // Create tensors
    CLTensor src = create_tensor<CLTensor>(TensorShape{ 10U, 10U, 2U }, DataType::U8);

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

BOOST_AUTO_TEST_SUITE_END()
#endif /* DOXYGEN_SKIP_THIS */
