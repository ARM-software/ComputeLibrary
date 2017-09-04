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
#include "NEON/Helper.h"
#include "NEON/NEAccessor.h"
#include "TypePrinter.h"
#include "dataset/BatchNormalizationLayerDataset.h"
#include "tests/validation/Helpers.h"
#include "validation/Datasets.h"
#include "validation/Reference.h"
#include "validation/Validation.h"

#include "arm_compute/runtime/NEON/functions/NEBatchNormalizationLayer.h"

#include <random>

using namespace arm_compute;
using namespace arm_compute::test;
using namespace arm_compute::test::neon;
using namespace arm_compute::test::validation;

namespace
{
const float tolerance_f = 1e-05; /**< Tolerance value for comparing reference's output against floating point implementation's output */
const float tolerance_q = 3;     /**< Tolerance value for comparing reference's output against quantized implementation's output */

/** Compute Neon batch normalization function.
 *
 * @param[in] shape     Shape of the input and output tensors.
 * @param[in] dt        Data type of input and output tensors.
 * @param[in] norm_info Normalization Layer information.
 *
 * @return Computed output tensor.
 */
Tensor compute_reference_batch_normalization_layer(const TensorShape &shape0, const TensorShape &shape1, DataType dt, float epsilon, int fixed_point_position = 0)
{
    // Create tensors
    Tensor src   = create_tensor(shape0, dt, 1, fixed_point_position);
    Tensor dst   = create_tensor(shape0, dt, 1, fixed_point_position);
    Tensor mean  = create_tensor(shape1, dt, 1, fixed_point_position);
    Tensor var   = create_tensor(shape1, dt, 1, fixed_point_position);
    Tensor beta  = create_tensor(shape1, dt, 1, fixed_point_position);
    Tensor gamma = create_tensor(shape1, dt, 1, fixed_point_position);

    // Create and configure function
    NEBatchNormalizationLayer norm;
    norm.configure(&src, &dst, &mean, &var, &beta, &gamma, epsilon);

    // Allocate tensors
    src.allocator()->allocate();
    dst.allocator()->allocate();
    mean.allocator()->allocate();
    var.allocator()->allocate();
    beta.allocator()->allocate();
    gamma.allocator()->allocate();

    BOOST_TEST(!src.info()->is_resizable());
    BOOST_TEST(!dst.info()->is_resizable());
    BOOST_TEST(!mean.info()->is_resizable());
    BOOST_TEST(!var.info()->is_resizable());
    BOOST_TEST(!beta.info()->is_resizable());
    BOOST_TEST(!gamma.info()->is_resizable());

    // Fill tensors
    if(dt == DataType::F32)
    {
        float min_bound = 0.f;
        float max_bound = 0.f;
        std::tie(min_bound, max_bound) = get_batchnormalization_layer_test_bounds<float>();
        std::uniform_real_distribution<> distribution(min_bound, max_bound);
        std::uniform_real_distribution<> distribution_var(0, max_bound);
        library->fill(NEAccessor(src), distribution, 0);
        library->fill(NEAccessor(mean), distribution, 1);
        library->fill(NEAccessor(var), distribution_var, 0);
        library->fill(NEAccessor(beta), distribution, 3);
        library->fill(NEAccessor(gamma), distribution, 4);
    }
    else
    {
        int min_bound = 0;
        int max_bound = 0;
        std::tie(min_bound, max_bound) = get_batchnormalization_layer_test_bounds<int8_t>(fixed_point_position);
        std::uniform_int_distribution<> distribution(min_bound, max_bound);
        std::uniform_int_distribution<> distribution_var(0, max_bound);
        library->fill(NEAccessor(src), distribution, 0);
        library->fill(NEAccessor(mean), distribution, 1);
        library->fill(NEAccessor(var), distribution_var, 0);
        library->fill(NEAccessor(beta), distribution, 3);
        library->fill(NEAccessor(gamma), distribution, 4);
    }

    // Compute function
    norm.run();

    return dst;
}
} // namespace

#ifndef DOXYGEN_SKIP_THIS
BOOST_AUTO_TEST_SUITE(NEON)
BOOST_AUTO_TEST_SUITE(BatchNormalizationLayer)

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(Configuration, RandomBatchNormalizationLayerDataset() * (boost::unit_test::data::make(DataType::F32) + boost::unit_test::data::make(DataType::QS8)), obj, dt)
{
    // Set fixed point position data type allowed
    int fixed_point_position = (arm_compute::is_data_type_fixed_point(dt)) ? 3 : 0;

    // Create tensors
    Tensor src   = create_tensor(obj.shape0, dt, 1, fixed_point_position);
    Tensor dst   = create_tensor(obj.shape0, dt, 1, fixed_point_position);
    Tensor mean  = create_tensor(obj.shape1, dt, 1, fixed_point_position);
    Tensor var   = create_tensor(obj.shape1, dt, 1, fixed_point_position);
    Tensor beta  = create_tensor(obj.shape1, dt, 1, fixed_point_position);
    Tensor gamma = create_tensor(obj.shape1, dt, 1, fixed_point_position);

    BOOST_TEST(src.info()->is_resizable());
    BOOST_TEST(dst.info()->is_resizable());
    BOOST_TEST(mean.info()->is_resizable());
    BOOST_TEST(var.info()->is_resizable());
    BOOST_TEST(beta.info()->is_resizable());
    BOOST_TEST(gamma.info()->is_resizable());

    // Create and configure function
    NEBatchNormalizationLayer norm;
    norm.configure(&src, &dst, &mean, &var, &beta, &gamma, obj.epsilon);

    // Validate valid region
    const ValidRegion valid_region     = shape_to_valid_region(obj.shape0);
    const ValidRegion valid_region_vec = shape_to_valid_region(obj.shape1);
    validate(src.info()->valid_region(), valid_region);
    validate(dst.info()->valid_region(), valid_region);
    validate(mean.info()->valid_region(), valid_region_vec);
    validate(var.info()->valid_region(), valid_region_vec);
    validate(beta.info()->valid_region(), valid_region_vec);
    validate(gamma.info()->valid_region(), valid_region_vec);
}

BOOST_AUTO_TEST_SUITE(Float)
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(Random,
                     RandomBatchNormalizationLayerDataset() * boost::unit_test::data::make(DataType::F32),
                     obj, dt)
{
    // Compute function
    Tensor dst = compute_reference_batch_normalization_layer(obj.shape0, obj.shape1, dt, obj.epsilon);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_batch_normalization_layer(obj.shape0, obj.shape1, dt, obj.epsilon);

    // Validate output
    validate(NEAccessor(dst), ref_dst, tolerance_f, 0);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(Quantized)
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(Random,
                     RandomBatchNormalizationLayerDataset() * boost::unit_test::data::make(DataType::QS8) * boost::unit_test::data::xrange(1, 6),
                     obj, dt, fixed_point_position)
{
    // Compute function
    Tensor dst = compute_reference_batch_normalization_layer(obj.shape0, obj.shape1, dt, obj.epsilon, fixed_point_position);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_batch_normalization_layer(obj.shape0, obj.shape1, dt, obj.epsilon, fixed_point_position);

    // Validate output
    validate(NEAccessor(dst), ref_dst, tolerance_q, 0);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
#endif
