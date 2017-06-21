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
#include "CL/Helper.h"
#include "Globals.h"
#include "PaddingCalculator.h"
#include "TensorLibrary.h"
#include "TypePrinter.h"
#include "Utils.h"
#include "dataset/ThresholdDataset.h"
#include "validation/Datasets.h"
#include "validation/Reference.h"
#include "validation/Validation.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLThreshold.h"

#include "boost_wrapper.h"

#include <random>
#include <string>

using namespace arm_compute;
using namespace arm_compute::test;
using namespace arm_compute::test::cl;
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
CLTensor compute_threshold(const TensorShape &shape, uint8_t threshold, uint8_t false_value, uint8_t true_value, ThresholdType type, uint8_t upper)
{
    // Create tensors
    CLTensor src = create_tensor(shape, DataType::U8);
    CLTensor dst = create_tensor(shape, DataType::U8);

    // Create and configure function
    CLThreshold thrsh;
    thrsh.configure(&src, &dst, threshold, false_value, true_value, type, upper);

    // Allocate tensors
    src.allocator()->allocate();
    dst.allocator()->allocate();

    BOOST_TEST(!src.info()->is_resizable());
    BOOST_TEST(!dst.info()->is_resizable());

    // Fill tensors
    library->fill_tensor_uniform(CLAccessor(src), 0);

    // Compute function
    thrsh.run();

    return dst;
}
} // namespace

#ifndef DOXYGEN_SKIP_THIS
BOOST_AUTO_TEST_SUITE(CL)
BOOST_AUTO_TEST_SUITE(Threshold)

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(Configuration,
                     (SmallShapes() + LargeShapes()) * ThresholdDataset(),
                     shape, threshold_conf)
{
    // Create tensors
    CLTensor src = create_tensor(shape, DataType::U8);
    CLTensor dst = create_tensor(shape, DataType::U8);

    BOOST_TEST(src.info()->is_resizable());
    BOOST_TEST(dst.info()->is_resizable());

    // Create and configure function
    CLThreshold cl_threshold;
    cl_threshold.configure(&src, &dst, threshold_conf.threshold, threshold_conf.false_value, threshold_conf.true_value, threshold_conf.type, threshold_conf.upper);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(src.info()->valid_region(), valid_region);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const PaddingSize padding(0, PaddingCalculator(shape.x(), 16).required_padding(), 0, 0);
    validate(src.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall,
                     SmallShapes() * ThresholdDataset(),
                     shape, threshold_conf)
{
    // Compute function
    CLTensor dst = compute_threshold(shape, threshold_conf.threshold, threshold_conf.false_value, threshold_conf.true_value, threshold_conf.type, threshold_conf.upper);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_threshold(shape, threshold_conf.threshold, threshold_conf.false_value, threshold_conf.true_value, threshold_conf.type, threshold_conf.upper);

    // Validate output
    validate(CLAccessor(dst), ref_dst);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(RunLarge,
                     LargeShapes() * ThresholdDataset(),
                     shape, threshold_conf)
{
    // Compute function
    CLTensor dst = compute_threshold(shape, threshold_conf.threshold, threshold_conf.false_value, threshold_conf.true_value, threshold_conf.type, threshold_conf.upper);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_threshold(shape, threshold_conf.threshold, threshold_conf.false_value, threshold_conf.true_value, threshold_conf.type, threshold_conf.upper);

    // Validate output
    validate(CLAccessor(dst), ref_dst);
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
#endif
