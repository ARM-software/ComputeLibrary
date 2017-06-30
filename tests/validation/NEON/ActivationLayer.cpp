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
#include "PaddingCalculator.h"
#include "TensorLibrary.h"
#include "TypePrinter.h"
#include "Utils.h"
#include "validation/Datasets.h"
#include "validation/Helpers.h"
#include "validation/Reference.h"
#include "validation/Validation.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/functions/NEActivationLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"

#include "boost_wrapper.h"

#include <random>
#include <string>
#include <tuple>

using namespace arm_compute;
using namespace arm_compute::test;
using namespace arm_compute::test::neon;
using namespace arm_compute::test::validation;

namespace
{
/** Define tolerance of the activation layer
 *
 * @param[in] activation           The activation function used.
 * @param[in] fixed_point_position Number of bits for the fractional part..
 *
 * @return Tolerance depending on the activation function.
 */
float activation_layer_tolerance(ActivationLayerInfo::ActivationFunction activation, int fixed_point_position = 0)
{
    switch(activation)
    {
        case ActivationLayerInfo::ActivationFunction::LOGISTIC:
        case ActivationLayerInfo::ActivationFunction::SOFT_RELU:
        case ActivationLayerInfo::ActivationFunction::SQRT:
        case ActivationLayerInfo::ActivationFunction::TANH:
            return (fixed_point_position != 0) ? 5.f : 0.00001f;
            break;
        default:
            return 0.f;
    }
}

/** Compute Neon activation layer function.
 *
 * @param[in] in_place             Compute the activation layer in-place.
 * @param[in] shape                Shape of the input and output tensors.
 * @param[in] dt                   Shape Data type of tensors.
 * @param[in] act_info             Activation layer information.
 * @param[in] fixed_point_position (Optional) Number of bits for the fractional part of fixed point numbers.
 *
 * @return Computed output tensor.
 */
Tensor compute_activation_layer(bool in_place, const TensorShape &shape, DataType dt, ActivationLayerInfo act_info, int fixed_point_position = 0)
{
    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, dt, 1, fixed_point_position);
    Tensor dst = create_tensor<Tensor>(shape, dt, 1, fixed_point_position);

    // Create and configure function
    NEActivationLayer act_layer;

    if(in_place)
    {
        act_layer.configure(&src, nullptr, act_info);
    }
    else
    {
        act_layer.configure(&src, &dst, act_info);
    }

    // Allocate tensors
    src.allocator()->allocate();
    BOOST_TEST(!src.info()->is_resizable());

    if(!in_place)
    {
        dst.allocator()->allocate();
        BOOST_TEST(!dst.info()->is_resizable());
    }

    // Fill tensors
    if(dt == DataType::F32)
    {
        float min_bound = 0;
        float max_bound = 0;
        std::tie(min_bound, max_bound) = get_activation_layer_test_bounds<float>(act_info.activation());
        std::uniform_real_distribution<> distribution(min_bound, max_bound);
        library->fill(NEAccessor(src), distribution, 0);
    }
    else
    {
        int min_bound = 0;
        int max_bound = 0;
        std::tie(min_bound, max_bound) = get_activation_layer_test_bounds<int8_t>(act_info.activation(), fixed_point_position);
        std::uniform_int_distribution<> distribution(min_bound, max_bound);
        library->fill(NEAccessor(src), distribution, 0);
    }

    // Compute function
    act_layer.run();

    if(in_place)
    {
        return src;
    }
    else
    {
        return dst;
    }
}
} // namespace

#ifndef DOXYGEN_SKIP_THIS
BOOST_AUTO_TEST_SUITE(NEON)
BOOST_AUTO_TEST_SUITE(ActivationLayer)

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(Configuration, boost::unit_test::data::make({ false, true }) * (SmallShapes() + LargeShapes()) * CNNDataTypes(), in_place, shape, dt)
{
    // Set fixed point position data type allowed
    const int fixed_point_position = (arm_compute::is_data_type_fixed_point(dt)) ? 3 : 0;

    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, dt, 1, fixed_point_position);
    Tensor dst = create_tensor<Tensor>(shape, dt, 1, fixed_point_position);

    BOOST_TEST(src.info()->is_resizable());
    BOOST_TEST(dst.info()->is_resizable());

    // Create and configure function
    NEActivationLayer act_layer;

    if(in_place)
    {
        act_layer.configure(&src, nullptr, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::ABS));
    }
    else
    {
        act_layer.configure(&src, &dst, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::ABS));
    }

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(src.info()->valid_region(), valid_region);

    if(!in_place)
    {
        validate(dst.info()->valid_region(), valid_region);
    }

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 16).required_padding();
    validate(src.info()->padding(), padding);

    if(!in_place)
    {
        validate(dst.info()->padding(), padding);
    }
}

BOOST_AUTO_TEST_SUITE(Float)
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall, boost::unit_test::data::make({ false, true }) * SmallShapes() * CNNFloatDataTypes() * ActivationFunctions(), in_place, shape, dt, act_function)
{
    // Create activation layer info
    ActivationLayerInfo act_info(act_function, 1.f, 1.f);

    // Compute function
    Tensor dst = compute_activation_layer(in_place, shape, dt, act_info);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_activation_layer(shape, dt, act_info);

    // Validate output
    validate(NEAccessor(dst), ref_dst, activation_layer_tolerance(act_function));
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(RunLarge, boost::unit_test::data::make({ false, true }) * LargeShapes() * CNNFloatDataTypes() * ActivationFunctions(), in_place, shape, dt, act_function)
{
    // Create activation layer info
    ActivationLayerInfo act_info(act_function, 1.f, 1.f);

    // Compute function
    Tensor dst = compute_activation_layer(in_place, shape, dt, act_info);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_activation_layer(shape, dt, act_info);

    // Validate output
    validate(NEAccessor(dst), ref_dst, activation_layer_tolerance(act_function));
}
BOOST_AUTO_TEST_SUITE_END()

/** @note We test for fixed point precision [3,5] because [1,2] and [6,7] ranges
 *        cause overflowing issues in most of the transcendentals functions.
 */
BOOST_AUTO_TEST_SUITE(Quantized)
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall, boost::unit_test::data::make({ false, true }) * SmallShapes() * ActivationFunctions() * boost::unit_test::data::xrange(3, 6, 1),
                     in_place, shape, act_function, fixed_point_position)
{
    // Create activation layer info
    ActivationLayerInfo act_info(act_function, 1.f, 1.f);

    // Compute function
    Tensor dst = compute_activation_layer(in_place, shape, DataType::QS8, act_info, fixed_point_position);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_activation_layer(shape, DataType::QS8, act_info, fixed_point_position);

    // Validate output
    validate(NEAccessor(dst), ref_dst, activation_layer_tolerance(act_function, fixed_point_position));
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
#endif
