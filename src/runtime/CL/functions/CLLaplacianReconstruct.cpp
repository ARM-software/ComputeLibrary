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
#include "arm_compute/runtime/CL/functions/CLLaplacianReconstruct.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/IPyramid.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "support/ToolchainSupport.h"

#include <cstddef>

using namespace arm_compute;

CLLaplacianReconstruct::CLLaplacianReconstruct() // NOLINT
    : _tmp_pyr(),
      _addf(),
      _scalef(),
      _depthf()
{
}

void CLLaplacianReconstruct::configure(const CLPyramid *pyramid, const ICLTensor *input, ICLTensor *output, BorderMode border_mode, uint8_t constant_border_value)
{
    ARM_COMPUTE_ERROR_ON(nullptr == pyramid);
    ARM_COMPUTE_ERROR_ON(input == output);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::S16);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON(input->info()->num_dimensions() != pyramid->get_pyramid_level(0)->info()->num_dimensions());
    ARM_COMPUTE_ERROR_ON(output->info()->num_dimensions() != pyramid->get_pyramid_level(0)->info()->num_dimensions());
    ARM_COMPUTE_ERROR_ON(output->info()->dimension(0) != pyramid->get_pyramid_level(0)->info()->dimension(0));
    ARM_COMPUTE_ERROR_ON(output->info()->dimension(1) != pyramid->get_pyramid_level(0)->info()->dimension(1));
    ARM_COMPUTE_ERROR_ON(input->info()->dimension(0) != pyramid->get_pyramid_level(pyramid->info()->num_levels() - 1)->info()->dimension(0));
    ARM_COMPUTE_ERROR_ON(input->info()->dimension(1) != pyramid->get_pyramid_level(pyramid->info()->num_levels() - 1)->info()->dimension(1));

    const size_t num_levels = pyramid->info()->num_levels();

    // Create and initialize the tmp pyramid: I(n-2) = upsample( input + Laplace(n-1) )
    PyramidInfo pyramid_info;
    pyramid_info.init(num_levels, 0.5f, output->info()->tensor_shape(), arm_compute::Format::S16);
    _tmp_pyr.init(pyramid_info);

    // Allocate add and scale functions. Level 0 does not need to be scaled.
    _addf   = arm_compute::support::cpp14::make_unique<CLArithmeticAddition[]>(num_levels);
    _scalef = arm_compute::support::cpp14::make_unique<CLScale[]>(num_levels - 1);

    const size_t last_level = num_levels - 1;

    _addf[last_level].configure(input, pyramid->get_pyramid_level(last_level), _tmp_pyr.get_pyramid_level(last_level), ConvertPolicy::SATURATE);

    // Scale levels n-1 to 1, and add levels n-2 to 0
    for(size_t l = 0; l < last_level; ++l)
    {
        _scalef[l].configure(_tmp_pyr.get_pyramid_level(l + 1), _tmp_pyr.get_pyramid_level(l), arm_compute::InterpolationPolicy::NEAREST_NEIGHBOR, border_mode, constant_border_value);
        _addf[l].configure(_tmp_pyr.get_pyramid_level(l), pyramid->get_pyramid_level(l), _tmp_pyr.get_pyramid_level(l), ConvertPolicy::SATURATE);
    }

    // Convert level 0 from S16 to U8
    _depthf.configure(_tmp_pyr.get_pyramid_level(0), output, ConvertPolicy::SATURATE, 0);

    _tmp_pyr.allocate();
}

void CLLaplacianReconstruct::run()
{
    ARM_COMPUTE_ERROR_ON_MSG(_addf == nullptr, "Unconfigured function");

    const size_t last_level = _tmp_pyr.info()->num_levels() - 1;

    _addf[last_level].run();

    // Run l = [last_level - 1, 0]
    for(size_t l = last_level; l-- > 0;)
    {
        _scalef[l].run();
        _addf[l].run();
    }

    _depthf.run();
}
