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
#include "arm_compute/runtime/NEON/functions/NEROIPoolingLayer.h"
#include "validation/Datasets.h"
#include "validation/Helpers.h"
#include "validation/Reference.h"
#include "validation/Validation.h"
#include "validation/ValidationUserConfiguration.h"

#include <random>

using namespace arm_compute;
using namespace arm_compute::test;
using namespace arm_compute::test::neon;
using namespace arm_compute::test::validation;

namespace
{
Tensor compute_roi_pooling_layer(const TensorShape &shape, DataType dt, const std::vector<ROI> &rois, ROIPoolingLayerInfo pool_info)
{
    TensorShape shape_dst;
    shape_dst.set(0, pool_info.pooled_width());
    shape_dst.set(1, pool_info.pooled_height());
    shape_dst.set(2, shape.z());
    shape_dst.set(3, rois.size());

    // Create tensors
    Tensor     src        = create_tensor(shape, dt);
    Tensor     dst        = create_tensor(shape_dst, dt);
    Array<ROI> rois_array = create_array(rois);

    // Create and configure function
    NEROIPoolingLayer roi_pool;
    roi_pool.configure(&src, &rois_array, &dst, pool_info);

    // Allocate tensors
    src.allocator()->allocate();
    dst.allocator()->allocate();

    BOOST_TEST(!src.info()->is_resizable());
    BOOST_TEST(!dst.info()->is_resizable());

    // Fill tensors
    std::uniform_real_distribution<> distribution(-1, 1);
    library->fill(NEAccessor(src), distribution, 0);

    // Compute function
    roi_pool.run();

    return dst;
}
} // namespace

#ifndef DOXYGEN_SKIP_THIS
BOOST_AUTO_TEST_SUITE(NEON)
BOOST_AUTO_TEST_SUITE(ROIPoolingLayer)

BOOST_AUTO_TEST_SUITE(Float)
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall, CNNFloatDataTypes() * boost::unit_test::data::make({ 10, 20, 40 }) * boost::unit_test::data::make({ 7, 9 }) * boost::unit_test::data::make({ 1.f / 8.f, 1.f / 16.f }),
                     dt, num_rois, roi_pool_size, roi_scale)
{
    TensorShape         shape(50U, 47U, 2U, 3U);
    ROIPoolingLayerInfo pool_info(roi_pool_size, roi_pool_size, roi_scale);

    // Construct ROI vector
    std::vector<ROI> rois = generate_random_rois(shape, pool_info, num_rois, user_config.seed);

    // Compute function
    Tensor dst = compute_roi_pooling_layer(shape, dt, rois, pool_info);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_roi_pooling_layer(shape, dt, rois, pool_info);

    // Validate output
    validate(NEAccessor(dst), ref_dst);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
#endif
