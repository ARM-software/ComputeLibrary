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
#include "NEON/Accessor.h"
#include "TypePrinter.h"
#include "Utils.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/validation_old/Datasets.h"
#include "tests/validation_old/ReferenceCPP.h"
#include "tests/validation_old/Validation.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/NEON/NEFixedPoint.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"

#include "tests/validation_old/boost_wrapper.h"

#include <random>
#include <string>

using namespace arm_compute;
using namespace arm_compute::test;
using namespace arm_compute::test::validation;

namespace
{
const float tolerance = 4.0f; /**< Tolerance value for comparing reference's output against implementation's output */

/** Compute Neon inverse square root function for signed 8bit fixed point.
 *
 * @param[in] shape Shape of the input and output tensors.
 *
 * @return Computed output tensor.
 */
Tensor compute_invsqrt_qs8(const TensorShape &shape, int fixed_point_position)
{
    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, DataType::QS8, 1, fixed_point_position);
    Tensor dst = create_tensor<Tensor>(shape, DataType::QS8, 1, fixed_point_position);

    constexpr unsigned int num_elems_processed_per_iteration = 16;
    Window                 window                            = calculate_max_window(*src.info(), Steps(num_elems_processed_per_iteration));
    AccessWindowHorizontal input_access(src.info(), 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal output_access(dst.info(), 0, num_elems_processed_per_iteration);

    update_window_and_padding(window, input_access, output_access);
    output_access.set_valid_region(window, src.info()->valid_region());

    // Allocate tensors
    src.allocator()->allocate();
    dst.allocator()->allocate();

    BOOST_TEST(!src.info()->is_resizable());
    BOOST_TEST(!dst.info()->is_resizable());

    // Fill tensors. Keep the range between [1, 127).
    std::uniform_int_distribution<> distribution(1, 127);
    library->fill(Accessor(src), distribution, 0);

    Iterator input(&src, window);
    Iterator output(&dst, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        qint8x16_t in = vld1q_s8(reinterpret_cast<const qint8_t *>(input.ptr()));
        vst1q_s8(reinterpret_cast<qint8_t *>(output.ptr()), vqinvsqrtq_qs8(in, fixed_point_position));
    },
    input, output);

    return dst;
}
} // namespace

#ifndef DOXYGEN_SKIP_THIS
BOOST_AUTO_TEST_SUITE(NEON)
BOOST_AUTO_TEST_SUITE(FixedPoint)
BOOST_AUTO_TEST_SUITE(QS8)
BOOST_AUTO_TEST_SUITE(Invsqrt)

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(Small1DShape, SmallShapes() * boost::unit_test::data::xrange(1, 6), shape, fixed_point_position)
{
    // Compute function
    Tensor dst = compute_invsqrt_qs8(shape, fixed_point_position);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_fixed_point_operation(shape, DataType::QS8, DataType::QS8, FixedPointOp::INV_SQRT, fixed_point_position);

    // Validate output
    validate(Accessor(dst), ref_dst, tolerance, 0);
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
#endif /* DOXYGEN_SKIP_THIS */
