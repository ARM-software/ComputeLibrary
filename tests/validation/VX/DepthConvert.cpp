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
#include "TensorLibrary.h"
#include "TypePrinter.h"
#include "Utils.h"
#include "VX/VXAccessor.h"
#include "dataset/ThresholdDataset.h"
#include "validation/Datasets.h"
#include "validation/ReferenceCPP.h"
#include "validation/VX/VXFixture.h"
#include "validation/VX/VXHelpers.h"
#include "validation/Validation.h"

#include "boost_wrapper.h"

#include <VX/vx.h>

#include <random>
#include <string>

using namespace arm_compute;
using namespace arm_compute::test;
using namespace arm_compute::test::vx;
using namespace arm_compute::test::validation;
using namespace arm_compute::test::validation::vx;

namespace
{
/** Compute VX Depth Covert function.
 *
 * @param[in] context    VX context.
 * @param[in] image_name Image name
 * @param[in] dt_in      Data type of input.
 * @param[in] vxdt_out   VX data type of output.
 * @param[in] policy     Conversion policy.
 * @param[in] shift      Value for down/up conversions. Must be 0 <= shift < 8.

 * @note The caller is responsible for releasing the returned vx_image.
 *
 * @return Computed output image.
 *
 */
vx_image compute_depth_convert(vx_context &context, const std::string &image_name,
                               Format dt_in, vx_df_image_e vxdt_out, ConvertPolicy policy, uint32_t shift)
{
    vx_convert_policy_e vxpolicy = ((policy == ConvertPolicy::SATURATE) ? VX_CONVERT_POLICY_SATURATE : VX_CONVERT_POLICY_WRAP);
    vx_scalar           vxshift  = vxCreateScalar(context, VX_TYPE_INT32, &shift);

    // Create graph
    vx_graph graph = vxCreateGraph(context);

    // Create images
    RawTensor raw = library->get(image_name, dt_in);
    vx_image  src = vxCreateImage(context, raw.shape()[0], raw.shape()[1], get_vximage_format(dt_in));
    vx_image  dst = vxCreateImage(context, raw.shape()[0], raw.shape()[1], vxdt_out);

    // Fill images
    library->fill(VXAccessor(src), image_name, dt_in);

    // Create and configure node
    vxConvertDepthNode(graph, src, dst, vxpolicy, vxshift);

    // Compute function
    if(vxVerifyGraph(graph) == VX_SUCCESS)
    {
        vxProcessGraph(graph);
    }

    // Release objects
    vxReleaseImage(&src);
    vxReleaseGraph(&graph);
    vxReleaseScalar(&vxshift);

    return dst;
}

/** Compute VX reference Depth Covert.
 *
 * @param[in] image_name Image name.
 * @param[in] dt_in      Data type of input.
 * @param[in] dt_out     Data type of output.
 * @param[in] policy     Conversion policy.
 * @param[in] shift      Value for down/up conversions. Must be 0 <= shift < 8. *
 *
 * @return Computed raw tensor.
 */
RawTensor compute_reference(const std::string &image_name, Format dt_in, Format dt_out, ConvertPolicy policy, uint32_t shift)
{
    // Create reference
    RawTensor ref_src = library->get(image_name, dt_in);
    RawTensor ref_dst = library->get(ref_src.shape(), dt_out);

    // Fill reference
    library->fill(ref_src, image_name, dt_in);

    // Compute reference
    ReferenceCPP::depth_convert(ref_src, ref_dst, policy, shift);

    return ref_dst;
}
} // namespace

#ifndef DOXYGEN_SKIP_THIS
BOOST_FIXTURE_TEST_SUITE(VX, VXFixture)
BOOST_AUTO_TEST_SUITE(DepthConvert)

BOOST_AUTO_TEST_SUITE(U8_to_U16)
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall, SmallImages() * boost::unit_test::data::make({ ConvertPolicy::SATURATE, ConvertPolicy::WRAP }) * boost::unit_test::data::xrange(0, 7, 1),
                     image_name, policy, shift)
{
    // Compute function
    vx_image dst = compute_depth_convert(context, image_name, Format::U8, VX_DF_IMAGE_U16, policy, shift);

    // Compute reference
    RawTensor ref_dst = compute_reference(image_name, Format::U8, Format::U16, policy, shift);

    // Validate output
    validate(VXAccessor(dst), ref_dst);

    vxReleaseImage(&dst);
}
BOOST_TEST_DECORATOR(*boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(RunLarge, LargeImages() * boost::unit_test::data::make({ ConvertPolicy::SATURATE, ConvertPolicy::WRAP })
                     * boost::unit_test::data::xrange(0, 7, 1),
                     image_name, policy, shift)
{
    // Compute function
    vx_image dst = compute_depth_convert(context, image_name, Format::U8, VX_DF_IMAGE_U16, policy, shift);

    // Compute reference
    RawTensor ref_dst = compute_reference(image_name, Format::U8, Format::U16, policy, shift);

    // Validate output
    validate(VXAccessor(dst), ref_dst);

    vxReleaseImage(&dst);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(U8_to_S16)
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall, SmallImages() * boost::unit_test::data::make({ ConvertPolicy::SATURATE, ConvertPolicy::WRAP })
                     * boost::unit_test::data::xrange(0, 7, 1),
                     image_name, policy, shift)
{
    // Compute function
    vx_image dst = compute_depth_convert(context, image_name, Format::U8, VX_DF_IMAGE_S16, policy, shift);

    // Compute reference
    RawTensor ref_dst = compute_reference(image_name, Format::U8, Format::S16, policy, shift);

    // Validate output
    validate(VXAccessor(dst), ref_dst);

    vxReleaseImage(&dst);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("nightly"))

BOOST_DATA_TEST_CASE(RunLarge, LargeImages() * boost::unit_test::data::make({ ConvertPolicy::SATURATE, ConvertPolicy::WRAP })
                     * boost::unit_test::data::xrange(0, 7, 1),
                     image_name, policy, shift)
{
    // Compute function
    vx_image dst = compute_depth_convert(context, image_name, Format::U8, VX_DF_IMAGE_S16, policy, shift);

    // Compute reference
    RawTensor ref_dst = compute_reference(image_name, Format::U8, Format::S16, policy, shift);

    // Validate output
    validate(VXAccessor(dst), ref_dst);

    vxReleaseImage(&dst);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(U16_to_U8)
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall, SmallImages() * boost::unit_test::data::make({ ConvertPolicy::SATURATE, ConvertPolicy::WRAP })
                     * boost::unit_test::data::xrange(0, 7, 1),
                     image_name, policy, shift)
{
    // Compute function
    vx_image dst = compute_depth_convert(context, image_name, Format::U16, VX_DF_IMAGE_U8, policy, shift);

    // Compute reference
    RawTensor ref_dst = compute_reference(image_name, Format::U16, Format::U8, policy, shift);

    // Validate output
    validate(VXAccessor(dst), ref_dst);

    vxReleaseImage(&dst);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(RunLarge, LargeImages() * boost::unit_test::data::make({ ConvertPolicy::SATURATE, ConvertPolicy::WRAP })
                     * boost::unit_test::data::xrange(0, 7, 1),
                     image_name, policy, shift)
{
    // Compute function
    vx_image dst = compute_depth_convert(context, image_name, Format::U16, VX_DF_IMAGE_U8, policy, shift);

    // Compute reference
    RawTensor ref_dst = compute_reference(image_name, Format::U16, Format::U8, policy, shift);

    // Validate output
    validate(VXAccessor(dst), ref_dst);

    vxReleaseImage(&dst);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(U16_to_U32)
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall, SmallImages() * boost::unit_test::data::make({ ConvertPolicy::SATURATE, ConvertPolicy::WRAP })
                     * boost::unit_test::data::xrange(0, 7, 1),
                     image_name, policy, shift)
{
    // Compute function
    vx_image dst = compute_depth_convert(context, image_name, Format::U16, VX_DF_IMAGE_U32, policy, shift);

    // Compute reference
    RawTensor ref_dst = compute_reference(image_name, Format::U16, Format::U32, policy, shift);

    // Validate output
    validate(VXAccessor(dst), ref_dst);

    vxReleaseImage(&dst);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(RunLarge, LargeImages() * boost::unit_test::data::make({ ConvertPolicy::SATURATE, ConvertPolicy::WRAP })
                     * boost::unit_test::data::xrange(0, 7, 1),
                     image_name, policy, shift)
{
    // Compute function
    vx_image dst = compute_depth_convert(context, image_name, Format::U16, VX_DF_IMAGE_U32, policy, shift);

    // Compute reference
    RawTensor ref_dst = compute_reference(image_name, Format::U16, Format::U32, policy, shift);

    // Validate output
    validate(VXAccessor(dst), ref_dst);

    vxReleaseImage(&dst);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(S16_to_U8)
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall, SmallImages() * boost::unit_test::data::make({ ConvertPolicy::SATURATE, ConvertPolicy::WRAP })
                     * boost::unit_test::data::xrange(0, 7, 1),
                     image_name, policy, shift)
{
    // Compute function
    vx_image dst = compute_depth_convert(context, image_name, Format::S16, VX_DF_IMAGE_U8, policy, shift);

    // Compute reference
    RawTensor ref_dst = compute_reference(image_name, Format::S16, Format::U8, policy, shift);

    // Validate output
    validate(VXAccessor(dst), ref_dst);

    vxReleaseImage(&dst);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(RunLarge, LargeImages() * boost::unit_test::data::make({ ConvertPolicy::SATURATE, ConvertPolicy::WRAP })
                     * boost::unit_test::data::xrange(0, 7, 1),
                     image_name, policy, shift)
{
    // Compute function
    vx_image dst = compute_depth_convert(context, image_name, Format::S16, VX_DF_IMAGE_U8, policy, shift);

    // Compute reference
    RawTensor ref_dst = compute_reference(image_name, Format::S16, Format::U8, policy, shift);

    // Validate output
    validate(VXAccessor(dst), ref_dst);

    vxReleaseImage(&dst);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(S16_to_S32)
BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit"))
BOOST_DATA_TEST_CASE(RunSmall, SmallImages() * boost::unit_test::data::make({ ConvertPolicy::SATURATE, ConvertPolicy::WRAP })
                     * boost::unit_test::data::xrange(0, 7, 1),
                     image_name, policy, shift)
{
    // Compute function
    vx_image dst = compute_depth_convert(context, image_name, Format::S16, VX_DF_IMAGE_S32, policy, shift);

    // Compute reference
    RawTensor ref_dst = compute_reference(image_name, Format::S16, Format::S32, policy, shift);

    // Validate output
    validate(VXAccessor(dst), ref_dst);

    vxReleaseImage(&dst);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(RunLarge, LargeImages() * boost::unit_test::data::make({ ConvertPolicy::SATURATE, ConvertPolicy::WRAP })
                     * boost::unit_test::data::xrange(0, 7, 1),
                     image_name, policy, shift)
{
    // Compute function
    vx_image dst = compute_depth_convert(context, image_name, Format::S16, VX_DF_IMAGE_S32, policy, shift);

    // Compute reference
    RawTensor ref_dst = compute_reference(image_name, Format::S16, Format::S32, policy, shift);

    // Validate output
    validate(VXAccessor(dst), ref_dst);

    vxReleaseImage(&dst);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
#endif
