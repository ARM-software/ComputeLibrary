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

#include "TypePrinter.h"
#include "dataset/ConvolutionLayerDataset.h"
#include "tests/Globals.h"
#include "tests/Utils.h"
#include "validation/Datasets.h"
#include "validation/Reference.h"
#include "validation/Validation.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/functions/CLConvolutionLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"

#include <random>

using namespace arm_compute;
using namespace arm_compute::test;
using namespace arm_compute::test::validation;

namespace
{
const float tolerance_f16 = 1.1f;   /**< Tolerance value for comparing reference's output against implementation's output for DataType::F16 */
const float tolerance_f32 = 1e-03f; /**< Tolerance value for comparing reference's output against implementation's output for DataType::F32 */
const float tolerance_q   = 1.0f;   /**< Tolerance value for comparing reference's output against implementation's output for fixed point data types */

CLTensor compute_convolution_layer(const TensorShape &input_shape, const TensorShape &weights_shape, const TensorShape &bias_shape, const TensorShape &output_shape, DataType dt,
                                   const PadStrideInfo &conv_info, int fixed_point_position)
{
    // Create tensors
    CLTensor src     = create_tensor<CLTensor>(input_shape, dt, 1, fixed_point_position);
    CLTensor weights = create_tensor<CLTensor>(weights_shape, dt, 1, fixed_point_position);
    CLTensor bias    = create_tensor<CLTensor>(bias_shape, dt, 1, fixed_point_position);
    CLTensor dst     = create_tensor<CLTensor>(output_shape, dt, 1, fixed_point_position);

    // Create and configure function
    CLConvolutionLayer conv;
    conv.configure(&src, &weights, &bias, &dst, conv_info);

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
    if(dt == DataType::F32 || dt == DataType::F16)
    {
        std::uniform_real_distribution<> distribution(-1.0f, 1.0f);
        library->fill(CLAccessor(src), distribution, 0);
        library->fill(CLAccessor(weights), distribution, 1);
        library->fill(CLAccessor(bias), distribution, 2);
    }
    else
    {
        library->fill_tensor_uniform(CLAccessor(src), 0);
        library->fill_tensor_uniform(CLAccessor(weights), 1);
        library->fill_tensor_uniform(CLAccessor(bias), 2);
    }

    // Compute CLConvolutionLayer function
    conv.run();

    return dst;
}
} // namespace

#ifndef DOXYGEN_SKIP_THIS
BOOST_AUTO_TEST_SUITE(CL)
BOOST_AUTO_TEST_SUITE(ConvolutionLayer)
BOOST_AUTO_TEST_SUITE(GEMM)

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(Configuration,
                     AlexNetConvolutionLayerDataset() * boost::unit_test::data::make({ DataType::F32, DataType::QS8, DataType::QS16 }),
                     conv_set, dt)
{
    // Set fixed point position data type allowed
    int fixed_point_position = (dt == DataType::F32) ? 0 : 3;

    // Create tensors
    CLTensor src     = create_tensor<CLTensor>(conv_set.src_shape, dt, 1, fixed_point_position);
    CLTensor weights = create_tensor<CLTensor>(conv_set.weights_shape, dt, 1, fixed_point_position);
    CLTensor bias    = create_tensor<CLTensor>(conv_set.bias_shape, dt, 1, fixed_point_position);
    CLTensor dst     = create_tensor<CLTensor>(conv_set.dst_shape, dt, 1, fixed_point_position);

    BOOST_TEST(src.info()->is_resizable());
    BOOST_TEST(weights.info()->is_resizable());
    BOOST_TEST(bias.info()->is_resizable());
    BOOST_TEST(dst.info()->is_resizable());

    // Create and configure function
    CLConvolutionLayer conv;
    conv.configure(&src, &weights, &bias, &dst, conv_set.info);

    // Validate valid region
    const ValidRegion src_valid_region     = shape_to_valid_region(conv_set.src_shape);
    const ValidRegion weights_valid_region = shape_to_valid_region(conv_set.weights_shape);
    const ValidRegion bias_valid_region    = shape_to_valid_region(conv_set.bias_shape);
    const ValidRegion dst_valid_region     = shape_to_valid_region(conv_set.dst_shape);

    validate(src.info()->valid_region(), src_valid_region);
    validate(weights.info()->valid_region(), weights_valid_region);
    validate(bias.info()->valid_region(), bias_valid_region);
    validate(dst.info()->valid_region(), dst_valid_region);
}

BOOST_AUTO_TEST_SUITE(Float16)
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(SmallConvolutionLayer,
                     SmallConvolutionLayerDataset() * boost::unit_test::data::make(DataType::F16),
                     conv_set, dt)
{
    // Compute function
    CLTensor dst = compute_convolution_layer(conv_set.src_shape, conv_set.weights_shape, conv_set.bias_shape, conv_set.dst_shape, dt, conv_set.info, 0);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_convolution_layer(conv_set.src_shape, conv_set.weights_shape, conv_set.bias_shape, conv_set.dst_shape, dt, conv_set.info, 0);

    // Validate output
    validate(CLAccessor(dst), ref_dst, tolerance_f16);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(Float)
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(SmallConvolutionLayer,
                     SmallConvolutionLayerDataset() * boost::unit_test::data::make(DataType::F32),
                     conv_set, dt)
{
    // Compute function
    CLTensor dst = compute_convolution_layer(conv_set.src_shape, conv_set.weights_shape, conv_set.bias_shape, conv_set.dst_shape, dt, conv_set.info, 0);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_convolution_layer(conv_set.src_shape, conv_set.weights_shape, conv_set.bias_shape, conv_set.dst_shape, dt, conv_set.info, 0);

    // Validate output
    validate(CLAccessor(dst), ref_dst, tolerance_f32);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(LargeConvolutionLayer,
                     AlexNetConvolutionLayerDataset() * boost::unit_test::data::make(DataType::F32),
                     conv_set, dt)
{
    // Compute function
    CLTensor dst = compute_convolution_layer(conv_set.src_shape, conv_set.weights_shape, conv_set.bias_shape, conv_set.dst_shape, dt, conv_set.info, 0);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_convolution_layer(conv_set.src_shape, conv_set.weights_shape, conv_set.bias_shape, conv_set.dst_shape, dt, conv_set.info, 0);

    // Validate output
    validate(CLAccessor(dst), ref_dst, tolerance_f32);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(Quantized)
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(SmallConvolutionLayer,
                     SmallConvolutionLayerDataset() * boost::unit_test::data::make({ DataType::QS8, DataType::QS16 }) * boost::unit_test::data::xrange(4, 7),
                     conv_set, dt, fixed_point_position)
{
    // Compute function
    CLTensor dst = compute_convolution_layer(conv_set.src_shape, conv_set.weights_shape, conv_set.bias_shape, conv_set.dst_shape, dt, conv_set.info, fixed_point_position);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_convolution_layer(conv_set.src_shape, conv_set.weights_shape, conv_set.bias_shape, conv_set.dst_shape, dt, conv_set.info, fixed_point_position);

    // Validate output
    validate(CLAccessor(dst), ref_dst, tolerance_q);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(LargeConvolutionLayer,
                     AlexNetConvolutionLayerDataset() * boost::unit_test::data::make({ DataType::QS8, DataType::QS16 }) * boost::unit_test::data::xrange(4, 7),
                     conv_set, dt, fixed_point_position)
{
    // Compute function
    CLTensor dst = compute_convolution_layer(conv_set.src_shape, conv_set.weights_shape, conv_set.bias_shape, conv_set.dst_shape, dt, conv_set.info, fixed_point_position);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_convolution_layer(conv_set.src_shape, conv_set.weights_shape, conv_set.bias_shape, conv_set.dst_shape, dt, conv_set.info, fixed_point_position);

    // Validate output
    validate(CLAccessor(dst), ref_dst, tolerance_q);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
#endif /* DOXYGEN_SKIP_THIS */
