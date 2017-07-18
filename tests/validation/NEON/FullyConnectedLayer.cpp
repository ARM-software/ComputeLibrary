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
#include "dataset/FullyConnectedLayerDataset.h"
#include "tests/Globals.h"
#include "tests/Utils.h"
#include "validation/Datasets.h"
#include "validation/Reference.h"
#include "validation/Validation.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/runtime/NEON/functions/NEFullyConnectedLayer.h"

#include <random>

using namespace arm_compute;
using namespace arm_compute::test;
using namespace arm_compute::test::validation;

namespace
{
const float tolerance_f32 = 1e-03f; /**< Tolerance value for comparing reference's output against implementation's output for DataType::F32 */
const float tolerance_q   = 1.0f;   /**< Tolerance value for comparing reference's output against implementation's output for fixed point data types */
#ifdef ARM_COMPUTE_ENABLE_FP16
const float tolerance_f16 = 0.01f; /**< Tolerance value for comparing reference's output against implementation's output for DataType::F16 */
#endif                             /*ARM_COMPUTE_ENABLE_FP16*/

Tensor compute_fully_connected_layer(const TensorShape &input_shape, const TensorShape &weights_shape, const TensorShape &bias_shape, const TensorShape &output_shape, DataType dt,
                                     bool transpose_weights, int fixed_point_position)
{
    // Create tensors
    Tensor src  = create_tensor<Tensor>(input_shape, dt, 1, fixed_point_position);
    Tensor bias = create_tensor<Tensor>(bias_shape, dt, 1, fixed_point_position);
    Tensor dst  = create_tensor<Tensor>(output_shape, dt, 1, fixed_point_position);

    // Swap the first and second dimension of weights' shape if transpose_weights is true
    TensorShape ws = weights_shape;
    if(transpose_weights)
    {
        const size_t dimx = ws.x();
        ws.set(0, ws.y());
        ws.set(1, dimx);
    }

    Tensor weights = create_tensor<Tensor>(ws, dt, 1, fixed_point_position);

    // Create and configure function.
    // Note: We pass the weights already transposed
    NEFullyConnectedLayer fc;
    fc.configure(&src, &weights, &bias, &dst, false);

    // Allocate tensors
    src.allocator()->allocate();
    weights.allocator()->allocate();
    bias.allocator()->allocate();
    dst.allocator()->allocate();

    BOOST_TEST(!src.info()->is_resizable());
    BOOST_TEST(!weights.info()->is_resizable());
    BOOST_TEST(!bias.info()->is_resizable());
    BOOST_TEST(!dst.info()->is_resizable());

    // Fill tensors
    if(dt == DataType::F16 || dt == DataType::F32)
    {
        std::uniform_real_distribution<> distribution(-1.0f, 1.0f);
        library->fill(Accessor(src), distribution, 0);
        library->fill(Accessor(weights), distribution, 1);
        library->fill(Accessor(bias), distribution, 2);
    }
    else
    {
        library->fill_tensor_uniform(Accessor(src), 0);
        library->fill_tensor_uniform(Accessor(weights), 1);
        library->fill_tensor_uniform(Accessor(bias), 2);
    }

    // Compute NEFullyConnectedLayer function
    fc.run();

    return dst;
}
} // namespace

#ifndef DOXYGEN_SKIP_THIS
BOOST_AUTO_TEST_SUITE(NEON)
BOOST_AUTO_TEST_SUITE(FullyConnectedLayer)

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(Configuration,
                     SmallFullyConnectedLayerDataset() * boost::unit_test::data::make({ DataType::F32, DataType::QS8, DataType::QS16 }),
                     fc_set, dt)
{
    // Set fixed point position data type allowed
    int fixed_point_position = (dt == DataType::F32) ? 0 : 3;

    // Create tensors
    Tensor src  = create_tensor<Tensor>(fc_set.src_shape, dt, 1, fixed_point_position);
    Tensor bias = create_tensor<Tensor>(fc_set.bias_shape, dt, 1, fixed_point_position);
    Tensor dst  = create_tensor<Tensor>(fc_set.dst_shape, dt, 1, fixed_point_position);

    // Swap the first and second dimension of weights' shape if transpose_weights is true
    TensorShape ws = fc_set.weights_shape;
    if(fc_set.transpose_weights)
    {
        const size_t dimx = ws.x();
        ws.set(0, ws.y());
        ws.set(1, dimx);
    }

    Tensor weights = create_tensor<Tensor>(ws, dt, 1, fixed_point_position);

    BOOST_TEST(src.info()->is_resizable());
    BOOST_TEST(weights.info()->is_resizable());
    BOOST_TEST(bias.info()->is_resizable());
    BOOST_TEST(dst.info()->is_resizable());

    // Create and configure function.
    // Note: We pass the weights already transposed
    NEFullyConnectedLayer fc;
    fc.configure(&src, &weights, &bias, &dst, false);

    // Validate valid region
    const ValidRegion src_valid_region     = shape_to_valid_region(fc_set.src_shape);
    const ValidRegion weights_valid_region = shape_to_valid_region(ws);
    const ValidRegion bias_valid_region    = shape_to_valid_region(fc_set.bias_shape);
    const ValidRegion dst_valid_region     = shape_to_valid_region(fc_set.dst_shape);

    validate(src.info()->valid_region(), src_valid_region);
    validate(weights.info()->valid_region(), weights_valid_region);
    validate(bias.info()->valid_region(), bias_valid_region);
    validate(dst.info()->valid_region(), dst_valid_region);
}

#ifdef ARM_COMPUTE_ENABLE_FP16
BOOST_AUTO_TEST_SUITE(Float16)
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall,
                     SmallFullyConnectedLayerDataset() * boost::unit_test::data::make({ DataType::F16 }),
                     fc_set, dt)
{
    // Compute function
    Tensor dst = compute_fully_connected_layer(fc_set.src_shape, fc_set.weights_shape, fc_set.bias_shape, fc_set.dst_shape, dt, fc_set.transpose_weights, 0);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_fully_connected_layer(fc_set.src_shape, fc_set.weights_shape, fc_set.bias_shape, fc_set.dst_shape, dt, fc_set.transpose_weights, 0);

    // Validate output
    validate(Accessor(dst), ref_dst, tolerance_f16);
}
BOOST_AUTO_TEST_SUITE_END()
#endif /* ARM_COMPUTE_ENABLE_FP16 */

BOOST_AUTO_TEST_SUITE(Float)
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall,
                     SmallFullyConnectedLayerDataset() * boost::unit_test::data::make({ DataType::F32 }),
                     fc_set, dt)
{
    // Compute function
    Tensor dst = compute_fully_connected_layer(fc_set.src_shape, fc_set.weights_shape, fc_set.bias_shape, fc_set.dst_shape, dt, fc_set.transpose_weights, 0);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_fully_connected_layer(fc_set.src_shape, fc_set.weights_shape, fc_set.bias_shape, fc_set.dst_shape, dt, fc_set.transpose_weights, 0);

    // Validate output
    validate(Accessor(dst), ref_dst, tolerance_f32);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(RunLarge,
                     LargeFullyConnectedLayerDataset() * boost::unit_test::data::make({ DataType::F32 }),
                     fc_set, dt)
{
    // Compute function
    Tensor dst = compute_fully_connected_layer(fc_set.src_shape, fc_set.weights_shape, fc_set.bias_shape, fc_set.dst_shape, dt, fc_set.transpose_weights, 0);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_fully_connected_layer(fc_set.src_shape, fc_set.weights_shape, fc_set.bias_shape, fc_set.dst_shape, dt, fc_set.transpose_weights, 0);

    // Validate output
    validate(Accessor(dst), ref_dst, tolerance_f32);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(Quantized)
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall,
                     SmallFullyConnectedLayerDataset() * boost::unit_test::data::make({ DataType::QS8, DataType::QS16 }) * boost::unit_test::data::xrange(4, 7),
                     fc_set, dt, fixed_point_position)
{
    // Compute function
    Tensor dst = compute_fully_connected_layer(fc_set.src_shape, fc_set.weights_shape, fc_set.bias_shape, fc_set.dst_shape, dt, fc_set.transpose_weights, fixed_point_position);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_fully_connected_layer(fc_set.src_shape, fc_set.weights_shape, fc_set.bias_shape, fc_set.dst_shape, dt, fc_set.transpose_weights, fixed_point_position);

    // Validate output
    validate(Accessor(dst), ref_dst, tolerance_q);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(RunLarge,
                     LargeFullyConnectedLayerDataset() * boost::unit_test::data::make({ DataType::QS8, DataType::QS16 }) * boost::unit_test::data::xrange(4, 7),
                     fc_set, dt, fixed_point_position)
{
    // Compute function
    Tensor dst = compute_fully_connected_layer(fc_set.src_shape, fc_set.weights_shape, fc_set.bias_shape, fc_set.dst_shape, dt, fc_set.transpose_weights, fixed_point_position);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_fully_connected_layer(fc_set.src_shape, fc_set.weights_shape, fc_set.bias_shape, fc_set.dst_shape, dt, fc_set.transpose_weights, fixed_point_position);

    // Validate output
    validate(Accessor(dst), ref_dst, tolerance_q);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
#endif /* DOXYGEN_SKIP_THIS */
