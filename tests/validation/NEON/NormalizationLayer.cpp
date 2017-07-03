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
#include "NEON/NEAccessor.h"
#include "TypePrinter.h"
#include "tests/Globals.h"
#include "tests/Utils.h"
#include "validation/Datasets.h"
#include "validation/Reference.h"
#include "validation/Validation.h"

#include "arm_compute/runtime/NEON/functions/NENormalizationLayer.h"

#include <random>

using namespace arm_compute;
using namespace arm_compute::test;
using namespace arm_compute::test::neon;
using namespace arm_compute::test::validation;

namespace
{
/** Define tolerance of the normalization layer depending on values data type.
 *
 * @param[in] dt Data type of the tensors' values.
 *
 * @return Tolerance depending on the data type.
 */
float normalization_layer_tolerance(DataType dt)
{
    switch(dt)
    {
        case DataType::QS8:
            return 2.0f;
        case DataType::F32:
            return 1e-05;
        default:
            return 0.f;
    }
}

/** Compute Neon normalization layer function.
 *
 * @param[in] shape                Shape of the input and output tensors.
 * @param[in] dt                   Data type of input and output tensors.
 * @param[in] norm_info            Normalization Layer information.
 * @param[in] fixed_point_position (Optional) Fixed point position that expresses the number of bits for the fractional part of the number when the tensor's data type is QS8 or QS16 (default = 0).
 *
 * @return Computed output tensor.
 */
Tensor compute_normalization_layer(const TensorShape &shape, DataType dt, NormalizationLayerInfo norm_info, int fixed_point_position = 0)
{
    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, dt, 1, fixed_point_position);
    Tensor dst = create_tensor<Tensor>(shape, dt, 1, fixed_point_position);

    // Create and configure function
    NENormalizationLayer norm;
    norm.configure(&src, &dst, norm_info);

    // Allocate tensors
    src.allocator()->allocate();
    dst.allocator()->allocate();

    BOOST_TEST(!src.info()->is_resizable());
    BOOST_TEST(!dst.info()->is_resizable());

    // Fill tensors
    if(dt == DataType::QS8)
    {
        const int8_t one_fixed_point       = 1 << fixed_point_position;
        const int8_t minus_one_fixed_point = -one_fixed_point;
        library->fill_tensor_uniform(NEAccessor(src), 0, minus_one_fixed_point, one_fixed_point);
    }
    else
    {
        library->fill_tensor_uniform(NEAccessor(src), 0);
    }

    // Compute function
    norm.run();

    return dst;
}
} // namespace

#ifndef DOXYGEN_SKIP_THIS
BOOST_AUTO_TEST_SUITE(NEON)
BOOST_AUTO_TEST_SUITE(NormalizationLayer)

BOOST_AUTO_TEST_SUITE(Float)
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall,
                     SmallShapes() * DataType::F32 *NormalizationTypes() * boost::unit_test::data::xrange(3, 9, 2) * boost::unit_test::data::make({ 0.5f, 1.0f, 2.0f }),
                     shape, dt, norm_type, norm_size, beta)
{
    // Provide normalization layer information
    NormalizationLayerInfo norm_info(norm_type, norm_size, 5, beta);

    // Compute function
    Tensor dst = compute_normalization_layer(shape, dt, norm_info);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_normalization_layer(shape, dt, norm_info);

    // Validate output
    validate(NEAccessor(dst), ref_dst, normalization_layer_tolerance(DataType::F32));
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(Quantized)
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall,
                     SmallShapes() * DataType::QS8 *NormalizationTypes() * boost::unit_test::data::xrange(3, 7, 2) * (boost::unit_test::data::xrange(1, 6) * boost::unit_test::data::make({ 0.5f, 1.0f, 2.0f })),
                     shape, dt, norm_type, norm_size, fixed_point_position, beta)
{
    // Provide normalization layer information
    NormalizationLayerInfo norm_info(norm_type, norm_size, 5, beta, 1.f);

    // Compute function
    Tensor dst = compute_normalization_layer(shape, dt, norm_info, fixed_point_position);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_normalization_layer(shape, dt, norm_info, fixed_point_position);

    // Validate output
    validate(NEAccessor(dst), ref_dst, normalization_layer_tolerance(DataType::QS8));
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
#endif /* DOXYGEN_SKIP_THIS */
