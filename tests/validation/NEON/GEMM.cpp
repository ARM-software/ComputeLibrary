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
#include "Globals.h"
#include "NEON/NEAccessor.h"
#include "TensorLibrary.h"
#include "TypePrinter.h"
#include "Utils.h"
#include "dataset/GEMMDataset.h"
#include "validation/Datasets.h"
#include "validation/Reference.h"
#include "validation/Validation.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/functions/NEGEMM.h"
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
const float tolerance_f32 = 1e-03f; /**< Tolerance value for comparing reference's output against implementation's output for DataType::F32 */
const float tolerance_qs8 = 1.0f;   /**< Tolerance value for comparing reference's output against implementation's output for DataType::QS8 */

Tensor compute_gemm(const TensorShape &src_shape1, const TensorShape &src_shape2, const TensorShape &src_shape3,
                    const TensorShape &out_shape, float alpha, float beta, DataType dt, int fixed_point_position = 0)
{
    // Create tensors
    Tensor src1 = create_tensor<Tensor>(src_shape1, dt, 1, fixed_point_position);
    Tensor src2 = create_tensor<Tensor>(src_shape2, dt, 1, fixed_point_position);
    Tensor src3 = create_tensor<Tensor>(src_shape3, dt, 1, fixed_point_position);
    Tensor dst  = create_tensor<Tensor>(out_shape, dt, 1, fixed_point_position);

    // Create and configure function
    NEGEMM gemm;
    gemm.configure(&src1, &src2, &src3, &dst, alpha, beta);

    // Allocate tensors
    src1.allocator()->allocate();
    src2.allocator()->allocate();
    src3.allocator()->allocate();
    dst.allocator()->allocate();

    BOOST_TEST(!src1.info()->is_resizable());
    BOOST_TEST(!src2.info()->is_resizable());
    BOOST_TEST(!src3.info()->is_resizable());
    BOOST_TEST(!dst.info()->is_resizable());

    // Fill tensors
    if(dt == DataType::F16 || dt == DataType::F32)
    {
        std::uniform_real_distribution<> distribution(-1.0f, 1.0f);
        library->fill(NEAccessor(src1), distribution, 0);
        library->fill(NEAccessor(src2), distribution, 1);
        library->fill(NEAccessor(src3), distribution, 2);
    }
    else
    {
        library->fill_tensor_uniform(NEAccessor(src1), 0);
        library->fill_tensor_uniform(NEAccessor(src2), 1);
        library->fill_tensor_uniform(NEAccessor(src3), 2);
    }

    // Compute function
    gemm.run();

    return dst;
}
} // namespace

#ifndef DOXYGEN_SKIP_THIS
BOOST_AUTO_TEST_SUITE(NEON)
BOOST_AUTO_TEST_SUITE(GEMM)

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(Configuration,
                     SmallGEMMDataset() * boost::unit_test::data::make({ DataType::F32, DataType::QS8 }),
                     gemm_set, dt)
{
    // Set fixed point position data type allowed
    int fixed_point_position = (dt == DataType::F32) ? 0 : 3;

    // Create tensors
    Tensor src1 = create_tensor<Tensor>(gemm_set.shape_a, dt, 1, fixed_point_position);
    Tensor src2 = create_tensor<Tensor>(gemm_set.shape_b, dt, 1, fixed_point_position);
    Tensor src3 = create_tensor<Tensor>(gemm_set.shape_c, dt, 1, fixed_point_position);
    Tensor dst  = create_tensor<Tensor>(gemm_set.shape_d, dt, 1, fixed_point_position);

    BOOST_TEST(src1.info()->is_resizable());
    BOOST_TEST(src2.info()->is_resizable());
    BOOST_TEST(src3.info()->is_resizable());
    BOOST_TEST(dst.info()->is_resizable());

    // Create and configure function
    NEGEMM gemm;
    gemm.configure(&src1, &src2, &src3, &dst, gemm_set.alpha, gemm_set.beta);

    // Validate valid region
    const ValidRegion src1_valid_region = shape_to_valid_region(gemm_set.shape_a);
    const ValidRegion src2_valid_region = shape_to_valid_region(gemm_set.shape_b);
    const ValidRegion src3_valid_region = shape_to_valid_region(gemm_set.shape_c);
    const ValidRegion dst_valid_region  = shape_to_valid_region(gemm_set.shape_d);

    validate(src1.info()->valid_region(), src1_valid_region);
    validate(src2.info()->valid_region(), src2_valid_region);
    validate(src3.info()->valid_region(), src3_valid_region);
    validate(dst.info()->valid_region(), dst_valid_region);
}

#ifdef ARM_COMPUTE_ENABLE_FP16
BOOST_AUTO_TEST_SUITE(Float16)
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(SmallGEMM, SmallGEMMDataset() * boost::unit_test::data::make(DataType::F16),
                     gemm_set, dt)
{
    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_gemm(gemm_set.shape_a, gemm_set.shape_b, gemm_set.shape_c, gemm_set.shape_d, gemm_set.alpha, gemm_set.beta, dt);

    // Compute function
    Tensor dst = compute_gemm(gemm_set.shape_a, gemm_set.shape_b, gemm_set.shape_c, gemm_set.shape_d, gemm_set.alpha, gemm_set.beta, dt);

    // Validate output
    validate(NEAccessor(dst), ref_dst, tolerance_f32);
}
BOOST_AUTO_TEST_SUITE_END()
#endif /* ARM_COMPUTE_ENABLE_FP16 */

BOOST_AUTO_TEST_SUITE(Float)
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(SmallGEMM, SmallGEMMDataset() * boost::unit_test::data::make(DataType::F32),
                     gemm_set, dt)
{
    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_gemm(gemm_set.shape_a, gemm_set.shape_b, gemm_set.shape_c, gemm_set.shape_d, gemm_set.alpha, gemm_set.beta, dt);

    // Compute function
    Tensor dst = compute_gemm(gemm_set.shape_a, gemm_set.shape_b, gemm_set.shape_c, gemm_set.shape_d, gemm_set.alpha, gemm_set.beta, dt);

    // Validate output
    validate(NEAccessor(dst), ref_dst, tolerance_f32);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(LargeGEMM, LargeGEMMDataset() * boost::unit_test::data::make(DataType::F32),
                     gemm_set, dt)
{
    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_gemm(gemm_set.shape_a, gemm_set.shape_b, gemm_set.shape_c, gemm_set.shape_d, gemm_set.alpha, gemm_set.beta, dt);

    // Compute function
    Tensor dst = compute_gemm(gemm_set.shape_a, gemm_set.shape_b, gemm_set.shape_c, gemm_set.shape_d, gemm_set.alpha, gemm_set.beta, dt);

    // Validate output
    validate(NEAccessor(dst), ref_dst, tolerance_f32);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(Quantized)
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(SmallGEMM, SmallGEMMDataset() * boost::unit_test::data::make(DataType::QS8) * boost::unit_test::data::xrange(1, 7),
                     gemm_set, dt, fixed_point_position)
{
    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_gemm(gemm_set.shape_a, gemm_set.shape_b, gemm_set.shape_c, gemm_set.shape_d, gemm_set.alpha, gemm_set.beta, dt, fixed_point_position);

    // Compute function
    Tensor dst = compute_gemm(gemm_set.shape_a, gemm_set.shape_b, gemm_set.shape_c, gemm_set.shape_d, gemm_set.alpha, gemm_set.beta, dt, fixed_point_position);

    // Validate output
    validate(NEAccessor(dst), ref_dst, tolerance_qs8);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(LargeGEMM, LargeGEMMDataset() * boost::unit_test::data::make(DataType::QS8) * boost::unit_test::data::xrange(1, 7),
                     gemm_set, dt, fixed_point_position)
{
    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_gemm(gemm_set.shape_a, gemm_set.shape_b, gemm_set.shape_c, gemm_set.shape_d, gemm_set.alpha, gemm_set.beta, dt, fixed_point_position);

    // Compute function
    Tensor dst = compute_gemm(gemm_set.shape_a, gemm_set.shape_b, gemm_set.shape_c, gemm_set.shape_d, gemm_set.alpha, gemm_set.beta, dt, fixed_point_position);

    // Validate output
    validate(NEAccessor(dst), ref_dst, tolerance_qs8);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
#endif /* DOXYGEN_SKIP_THIS */
