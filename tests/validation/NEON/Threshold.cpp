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
#include "NEON/Accessor.h"
#include "PaddingCalculator.h"
#include "TypePrinter.h"
#include "Utils.h"
#include "dataset/ThresholdDataset.h"
#include "validation/Datasets.h"
#include "validation/Reference.h"
#include "validation/Validation.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/functions/NEThreshold.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"

#include "boost_wrapper.h"

#include <random>
#include <string>

using namespace arm_compute;
using namespace arm_compute::test;
using namespace arm_compute::test::validation;

namespace
{
/** Compute Threshold function.
 *
 * @param[in] shape       Shape of the input and output tensors.
 * @param[in] threshold   Threshold. When the threshold type is RANGE, this is used as the lower threshold.
 * @param[in] false_value value to set when the condition is not respected.
 * @param[in] true_value  value to set when the condition is respected.
 * @param[in] type        Thresholding type. Either RANGE or BINARY.
 * @param[in] upper       Upper threshold. Only used when the thresholding type is RANGE.
 *
 * @return Computed output tensor.
 */
Tensor compute_threshold(const TensorShape &shape, uint8_t threshold, uint8_t false_value, uint8_t true_value, ThresholdType type, uint8_t upper)
{
    // Create tensors
    Tensor src1 = create_tensor<Tensor>(shape, DataType::U8);
    Tensor dst  = create_tensor<Tensor>(shape, DataType::U8);

    // Create and configure function
    NEThreshold thrsh;
    thrsh.configure(&src1, &dst, threshold, false_value, true_value, type, upper);

    // Allocate tensors
    src1.allocator()->allocate();
    dst.allocator()->allocate();

    BOOST_TEST(!src1.info()->is_resizable());
    BOOST_TEST(!dst.info()->is_resizable());

    // Fill tensors
    library->fill_tensor_uniform(Accessor(src1), 0);

    // Compute function
    thrsh.run();

    return dst;
}
} // namespace

#ifndef DOXYGEN_SKIP_THIS
BOOST_AUTO_TEST_SUITE(NEON)
BOOST_AUTO_TEST_SUITE(Threshold)

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(Configuration,
                     (SmallShapes() + LargeShapes()) * ThresholdDataset(),
                     shape, thrshConf)
{
    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, DataType::U8);
    Tensor dst = create_tensor<Tensor>(shape, DataType::U8);

    BOOST_TEST(src.info()->is_resizable());
    BOOST_TEST(dst.info()->is_resizable());

    // Create and configure function
    NEThreshold thrsh;
    thrsh.configure(&src, &dst, thrshConf.threshold, thrshConf.false_value, thrshConf.true_value, thrshConf.type, thrshConf.upper);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(src.info()->valid_region(), valid_region);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 16).required_padding();
    validate(src.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall,
                     SmallShapes() * ThresholdDataset(),
                     shape, thrshConf)
{
    // Compute function
    Tensor dst = compute_threshold(shape, thrshConf.threshold, thrshConf.false_value, thrshConf.true_value, thrshConf.type, thrshConf.upper);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_threshold(shape, thrshConf.threshold, thrshConf.false_value, thrshConf.true_value, thrshConf.type, thrshConf.upper);

    // Validate output
    validate(Accessor(dst), ref_dst);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(RunLarge,
                     LargeShapes() * ThresholdDataset(),
                     shape, thrshConf)
{
    // Compute function
    Tensor dst = compute_threshold(shape, thrshConf.threshold, thrshConf.false_value, thrshConf.true_value, thrshConf.type, thrshConf.upper);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_threshold(shape, thrshConf.threshold, thrshConf.false_value, thrshConf.true_value, thrshConf.type, thrshConf.upper);

    // Validate output
    validate(Accessor(dst), ref_dst);
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
#endif /* DOXYGEN_SKIP_THIS */
