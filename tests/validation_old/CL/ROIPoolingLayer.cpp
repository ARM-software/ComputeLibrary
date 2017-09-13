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
#include "CL/CLArrayAccessor.h"
#include "TypePrinter.h"
#include "arm_compute/runtime/CL/CLArray.h"
#include "arm_compute/runtime/CL/functions/CLROIPoolingLayer.h"
#include "tests/Globals.h"
#include "tests/Utils.h"
#include "tests/validation_old/Datasets.h"
#include "tests/validation_old/Reference.h"
#include "tests/validation_old/Validation.h"
#include "tests/validation_old/ValidationUserConfiguration.h"

#include <random>
#include <vector>

using namespace arm_compute;
using namespace arm_compute::test;
using namespace arm_compute::test::validation;

namespace
{
CLTensor compute_roi_pooling_layer(const TensorShape &shape, DataType dt, const std::vector<ROI> &rois, ROIPoolingLayerInfo pool_info)
{
    TensorShape shape_dst;
    shape_dst.set(0, pool_info.pooled_width());
    shape_dst.set(1, pool_info.pooled_height());
    shape_dst.set(2, shape.z());
    shape_dst.set(3, rois.size());

    // Create tensors
    CLTensor src = create_tensor<CLTensor>(shape, dt);
    CLTensor dst = create_tensor<CLTensor>(shape_dst, dt);

    // Create ROI array
    CLArray<ROI> rois_array(rois.size());
    fill_array(CLArrayAccessor<ROI>(rois_array), rois);

    // Create and configure function
    CLROIPoolingLayer roi_pool;
    roi_pool.configure(&src, &rois_array, &dst, pool_info);

    // Allocate tensors
    src.allocator()->allocate();
    dst.allocator()->allocate();

    BOOST_TEST(!src.info()->is_resizable());
    BOOST_TEST(!dst.info()->is_resizable());

    // Fill tensors
    std::uniform_real_distribution<> distribution(-1, 1);
    library->fill(CLAccessor(src), distribution, 0);

    // Compute function
    roi_pool.run();

    return dst;
}
} // namespace

#ifndef DOXYGEN_SKIP_THIS
BOOST_AUTO_TEST_SUITE(CL)
BOOST_AUTO_TEST_SUITE(ROIPoolingLayer)

BOOST_AUTO_TEST_SUITE(Float)
//FIXME: COMPMID-528
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::disabled())
BOOST_DATA_TEST_CASE(RunSmall, boost::unit_test::data::make({ DataType::F16, DataType::F32 }) * boost::unit_test::data::make({ 10, 20, 40 }) * boost::unit_test::data::make({ 7, 9 }) *
                     boost::unit_test::data::make({ 1.f / 8.f, 1.f / 16.f }),
                     dt, num_rois, roi_pool_size, roi_scale)
{
    TensorShape         shape(50U, 47U, 2U, 3U);
    ROIPoolingLayerInfo pool_info(roi_pool_size, roi_pool_size, roi_scale);

    // Construct ROI vector
    std::vector<ROI> rois = generate_random_rois(shape, pool_info, num_rois, user_config.seed);

    // Compute function
    CLTensor dst = compute_roi_pooling_layer(shape, dt, rois, pool_info);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_roi_pooling_layer(shape, dt, rois, pool_info);

    // Validate output
    validate(CLAccessor(dst), ref_dst);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
#endif /* DOXYGEN_SKIP_THIS */
