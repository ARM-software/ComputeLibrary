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
#include "validation/Helpers.h"
#include "validation/Reference.h"
#include "validation/Validation.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/functions/NEDepthConcatenate.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"

#include "support/ToolchainSupport.h"

#include "boost_wrapper.h"

#include <algorithm>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace arm_compute;
using namespace arm_compute::test;
using namespace arm_compute::test::neon;
using namespace arm_compute::test::validation;

namespace
{
/** Compute NEON depth concatenate layer function.
 *
 * @param[in] shapes               List of shapes to concatenate
 * @param[in] dt                   Datatype of tensors
 * @param[in] fixed_point_position (Optional) Number of bits for the fractional part of fixed point numbers.
 *
 * @return Computed output tensor.
 */
Tensor compute_depth_concatenate_layer(const std::vector<TensorShape> &shapes, DataType dt, int fixed_point_position = 0)
{
    std::vector<std::unique_ptr<Tensor>> srcs{};
    TensorShape                          dst_shape = calculate_depth_concatenate_shape(shapes);

    // Create tensors
    for(unsigned int i = 0; i < shapes.size(); ++i)
    {
        srcs.push_back(support::cpp14::make_unique<Tensor>());
        srcs[i]->allocator()->init(TensorInfo(shapes[i], 1, dt, fixed_point_position));
    }
    Tensor dst = create_tensor<Tensor>(dst_shape, dt, 1, fixed_point_position);

    // Create a vector of raw pointer
    std::vector<ITensor *> srcs_raw{};
    srcs_raw.resize(srcs.size());
    std::transform(srcs.begin(), srcs.end(), srcs_raw.begin(), [](std::unique_ptr<Tensor> const & t)
    {
        return t.get();
    });

    // Create and configure function
    NEDepthConcatenate depth_concat;
    depth_concat.configure(srcs_raw, &dst);

    // Allocate tensors
    for(auto &t : srcs)
    {
        t->allocator()->allocate();
    }
    dst.allocator()->allocate();

    for(const auto &t : srcs)
    {
        BOOST_TEST(!t->info()->is_resizable());
    }
    BOOST_TEST(!dst.info()->is_resizable());

    // Fill tensors
    for(unsigned int i = 0; i < srcs.size(); ++i)
    {
        library->fill_tensor_uniform(NEAccessor(*srcs[i]), i);
    }

    // Compute function
    depth_concat.run();

    return dst;
}
} // namespace

#ifndef DOXYGEN_SKIP_THIS
BOOST_AUTO_TEST_SUITE(NEON)
BOOST_AUTO_TEST_SUITE(DepthConcatenateLayer)

BOOST_AUTO_TEST_SUITE(Float)
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall, SmallShapes() * CNNFloatDataTypes(), shape, dt)
{
    // Create input shapes
    std::vector<unsigned int> depths = { 4, 6, 11, 13 };
    std::vector<TensorShape>  shapes(depths.size(), shape);
    for(unsigned int i = 0; i < shapes.size(); ++i)
    {
        shapes[i].set(2, depths[i]);
    }

    // Compute function
    Tensor dst = compute_depth_concatenate_layer(shapes, dt);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_depth_concatenate_layer(shapes, dt);

    // Validate output
    validate(NEAccessor(dst), ref_dst);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmallPad, CNNFloatDataTypes(), dt)
{
    // Create input shapes
    std::vector<TensorShape> shapes{ TensorShape(12u, 12u, 14u, 8u), TensorShape(14u, 14u, 12u, 8u), TensorShape(16u, 16u, 11u, 8u) };

    // Compute function
    Tensor dst = compute_depth_concatenate_layer(shapes, dt);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_depth_concatenate_layer(shapes, dt);

    // Validate output
    validate(NEAccessor(dst), ref_dst);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(Quantized)
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall, SmallShapes() * CNNFixedPointDataTypes() * boost::unit_test::data::xrange(3, 6, 1), shape, dt, fixed_point_position)
{
    // Create input shapes
    std::vector<unsigned int> depths = { 4, 6, 11, 13 };
    std::vector<TensorShape>  shapes(depths.size(), shape);
    for(unsigned int i = 0; i < shapes.size(); ++i)
    {
        shapes[i].set(2, depths[i]);
    }

    // Compute function
    Tensor dst = compute_depth_concatenate_layer(shapes, dt, fixed_point_position);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_depth_concatenate_layer(shapes, dt, fixed_point_position);

    // Validate output
    validate(NEAccessor(dst), ref_dst);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmallPad, CNNFixedPointDataTypes() * boost::unit_test::data::xrange(3, 5, 1), dt, fixed_point_position)
{
    // Create input shapes
    std::vector<TensorShape> shapes{ TensorShape(12u, 12u, 14u, 8u), TensorShape(14u, 14u, 12u, 8u), TensorShape(16u, 16u, 11u, 8u) };

    // Compute function
    Tensor dst = compute_depth_concatenate_layer(shapes, dt, fixed_point_position);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_depth_concatenate_layer(shapes, dt, fixed_point_position);

    // Validate output
    validate(NEAccessor(dst), ref_dst);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
#endif /* DOXYGEN_SKIP_THIS */
