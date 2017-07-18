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
#include "AssetsLibrary.h"
#include "Globals.h"
#include "NEON/NEAccessor.h"
#include "PaddingCalculator.h"
#include "TypePrinter.h"
#include "Utils.h"
#include "validation/Datasets.h"
#include "validation/Reference.h"
#include "validation/Validation.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/functions/NEMeanStdDev.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"

#include "boost_wrapper.h"

#include <random>
#include <string>

using namespace arm_compute;
using namespace arm_compute::test;
using namespace arm_compute::test::neon;
using namespace arm_compute::test::validation;

namespace
{
/** Compute Neon mean and standard deviation function.
 *
 * @param[in] shape Shape of the input tensors.
 *
 * @return Computed mean and standard deviation.
 */
std::pair<float, float> compute_mean_and_standard_deviation(const TensorShape &shape)
{
    // Create tensor
    Tensor src = create_tensor<Tensor>(shape, DataType::U8);

    // Create output variables
    float mean    = 0.f;
    float std_dev = 0.f;

    // Create mean and standard deviation configure function
    NEMeanStdDev mean_std_dev_image;
    mean_std_dev_image.configure(&src, &mean, &std_dev);

    // Allocate tensors
    src.allocator()->allocate();

    BOOST_TEST(!src.info()->is_resizable());

    // Fill tensor
    library->fill_tensor_uniform(NEAccessor(src), 0);

    // Compute function
    mean_std_dev_image.run();

    return std::make_pair(mean, std_dev);
}
} // namespace

#ifndef DOXYGEN_SKIP_THIS
BOOST_AUTO_TEST_SUITE(NEON)
BOOST_AUTO_TEST_SUITE(MeanStdDev)

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(Configuration, Small2DShapes() + Large2DShapes(), shape)
{
    // Create tensor
    Tensor src = create_tensor<Tensor>(shape, DataType::U8);

    // Create output variables
    float mean    = 0.f;
    float std_dev = 0.f;

    BOOST_TEST(src.info()->is_resizable());

    // Create mean and standard deviation configure function
    NEMeanStdDev mean_std_dev_image;
    mean_std_dev_image.configure(&src, &mean, &std_dev);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(src.info()->valid_region(), valid_region);

    // Validate padding
    validate(src.info()->padding(), PaddingCalculator(shape.x(), 16).required_padding());
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall, Small2DShapes(), shape)
{
    // Compute function
    std::pair<float, float> output = compute_mean_and_standard_deviation(shape);

    // Compute reference
    std::pair<float, float> ref_output = Reference::compute_reference_mean_and_standard_deviation(shape);

    // Validate output
    validate(output.first, ref_output.first);
    validate(output.second, ref_output.second, 0.f, 0.001f);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(RunLarge, Large2DShapes(), shape)
{
    // Compute function
    std::pair<float, float> output = compute_mean_and_standard_deviation(shape);

    // Compute reference
    std::pair<float, float> ref_output = Reference::compute_reference_mean_and_standard_deviation(shape);

    // Validate output
    validate(output.first, ref_output.first);
    validate(output.second, ref_output.second, 0.f, 0.001f);
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
#endif /* DOXYGEN_SKIP_THIS */
