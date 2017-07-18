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
#include "arm_compute/runtime/NEON/functions/NEPoolingLayer.h"
#include "tests/Globals.h"
#include "tests/Utils.h"
#include "tests/dataset/PoolingLayerDataset.h"
#include "validation/Datasets.h"
#include "validation/Reference.h"
#include "validation/Validation.h"

#include <random>

using namespace arm_compute;
using namespace arm_compute::test;
using namespace arm_compute::test::validation;

namespace
{
const float tolerance_q = 0;     /**< Tolerance value for comparing reference's output against implementation's output for quantized input */
const float tolerance_f = 1e-05; /**< Tolerance value for comparing reference's output against implementation's output for float input */

/** Compute Neon pooling layer function.
 *
 * @param[in] shape     Shape of the input and output tensors.
 * @param[in] dt        Data type of input and output tensors.
 * @param[in] pool_info Pooling Layer information.
 *
 * @return Computed output tensor.
 */
Tensor compute_pooling_layer(const TensorShape &shape_in, const TensorShape &shape_out, DataType dt, PoolingLayerInfo pool_info, int fixed_point_position = 0)
{
    // Create tensors
    Tensor src = create_tensor<Tensor>(shape_in, dt, 1, fixed_point_position);
    Tensor dst = create_tensor<Tensor>(shape_out, dt, 1, fixed_point_position);

    // Create and configure function
    NEPoolingLayer pool;
    pool.configure(&src, &dst, pool_info);

    // Allocate tensors
    src.allocator()->allocate();
    dst.allocator()->allocate();

    BOOST_TEST(!src.info()->is_resizable());
    BOOST_TEST(!dst.info()->is_resizable());

    // Fill tensors
    int min = 0;
    int max = 0;
    switch(dt)
    {
        case DataType::F32:
            min = -1;
            max = 1;
            break;
        case DataType::QS8:
            min = -(1 << fixed_point_position);
            max = (1 << fixed_point_position);
            break;
        default:
            ARM_COMPUTE_ERROR("DataType not supported.");
    }
    std::uniform_real_distribution<> distribution(min, max);
    library->fill(Accessor(src), distribution, 0);

    // Compute function
    pool.run();

    return dst;
}

TensorShape get_output_shape(TensorShape in_shape, const PoolingLayerInfo &pool_info)
{
    TensorShape out_shape(in_shape);
    const std::pair<unsigned int, unsigned int> scaled_dims = arm_compute::scaled_dimensions(in_shape.x(),
                                                                                             in_shape.y(),
                                                                                             pool_info.pool_size(),
                                                                                             pool_info.pool_size(),
                                                                                             pool_info.pad_stride_info());
    out_shape.set(0, scaled_dims.first);
    out_shape.set(1, scaled_dims.second);
    return out_shape;
}
} // namespace

#ifndef DOXYGEN_SKIP_THIS
BOOST_AUTO_TEST_SUITE(NEON)
BOOST_AUTO_TEST_SUITE(PoolingLayer)

BOOST_AUTO_TEST_SUITE(Float)
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RandomDataset,
                     RandomPoolingLayerDataset() * boost::unit_test::data::make(DataType::F32),
                     obj, dt)
{
    // Compute function
    Tensor dst = compute_pooling_layer(obj.src_shape, obj.dst_shape, dt, obj.info);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_pooling_layer(obj.src_shape, obj.dst_shape, dt, obj.info);

    // Validate output
    validate(Accessor(dst), ref_dst, tolerance_f, 0);
}

BOOST_DATA_TEST_CASE(RunSmall7x7,
                     SmallShapes() * CNNFloatDataTypes() * PoolingTypes() * boost::unit_test::data::make({ 2, 3, 7 }) * boost::unit_test::data::make({ 1, 2 }) * boost::unit_test::data::make({ 0, 1 }),
                     src_shape, dt, pool_type, pool_size, pool_stride, pool_pad)
{
    PoolingLayerInfo pool_info(pool_type, pool_size, PadStrideInfo(pool_stride, pool_stride, pool_pad, pool_pad, DimensionRoundingType::CEIL));
    TensorShape      dst_shape = get_output_shape(src_shape, pool_info);

    // Compute function
    Tensor dst = compute_pooling_layer(src_shape, dst_shape, dt, pool_info);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_pooling_layer(src_shape, dst_shape, dt, pool_info);

    // Validate output
    validate(Accessor(dst), ref_dst, tolerance_f, 0);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(Quantized)
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RandomDataset,
                     RandomPoolingLayerDataset() * boost::unit_test::data::make(DataType::QS8) * boost::unit_test::data::xrange(1, 5),
                     obj, dt, fixed_point_position)
{
    // Compute function
    Tensor dst = compute_pooling_layer(obj.src_shape, obj.dst_shape, dt, obj.info, fixed_point_position);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_pooling_layer(obj.src_shape, obj.dst_shape, dt, obj.info, fixed_point_position);

    // Validate output
    validate(Accessor(dst), ref_dst, tolerance_q, 0);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
#endif /* DOXYGEN_SKIP_THIS */
