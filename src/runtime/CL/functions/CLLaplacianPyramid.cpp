/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#include "arm_compute/runtime/CL/functions/CLLaplacianPyramid.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/IPyramid.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/functions/CLDepthConvertLayer.h"
#include "arm_compute/runtime/CL/functions/CLElementwiseOperations.h"
#include "arm_compute/runtime/CL/functions/CLGaussian5x5.h"
#include "arm_compute/runtime/CL/functions/CLGaussianPyramid.h"
#include "support/ToolchainSupport.h"

using namespace arm_compute;

CLLaplacianPyramid::CLLaplacianPyramid() // NOLINT
    : _num_levels(0),
      _gaussian_pyr_function(),
      _convf(),
      _subf(),
      _depth_function(),
      _gauss_pyr(),
      _conv_pyr()
{
}

void CLLaplacianPyramid::configure(ICLTensor *input, CLPyramid *pyramid, ICLTensor *output, BorderMode border_mode, uint8_t constant_border_value)
{
    ARM_COMPUTE_ERROR_ON(nullptr == pyramid);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::S16);
    ARM_COMPUTE_ERROR_ON(0 == pyramid->info()->num_levels());
    ARM_COMPUTE_ERROR_ON(input->info()->dimension(0) != pyramid->info()->width());
    ARM_COMPUTE_ERROR_ON(input->info()->dimension(1) != pyramid->info()->height());
    ARM_COMPUTE_ERROR_ON(output->info()->dimension(0) != pyramid->get_pyramid_level(pyramid->info()->num_levels() - 1)->info()->dimension(0));
    ARM_COMPUTE_ERROR_ON(output->info()->dimension(1) != pyramid->get_pyramid_level(pyramid->info()->num_levels() - 1)->info()->dimension(1));

    _num_levels = pyramid->info()->num_levels();

    // Create and initialize the gaussian pyramid and the convoluted pyramid
    PyramidInfo pyramid_info;
    pyramid_info.init(_num_levels, 0.5f, pyramid->info()->tensor_shape(), arm_compute::Format::U8);

    _gauss_pyr.init(pyramid_info);
    _conv_pyr.init(pyramid_info);

    // Create Gaussian Pyramid function
    _gaussian_pyr_function.configure(input, &_gauss_pyr, border_mode, constant_border_value);

    _convf.resize(_num_levels);
    _subf.resize(_num_levels);

    for(unsigned int i = 0; i < _num_levels; ++i)
    {
        _convf[i].configure(_gauss_pyr.get_pyramid_level(i), _conv_pyr.get_pyramid_level(i), border_mode, constant_border_value);
        _subf[i].configure(_gauss_pyr.get_pyramid_level(i), _conv_pyr.get_pyramid_level(i), pyramid->get_pyramid_level(i), ConvertPolicy::WRAP);
    }

    _depth_function.configure(_conv_pyr.get_pyramid_level(_num_levels - 1), output, ConvertPolicy::WRAP, 0);

    _gauss_pyr.allocate();
    _conv_pyr.allocate();
}

void CLLaplacianPyramid::run()
{
    ARM_COMPUTE_ERROR_ON_MSG(0 == _num_levels, "Unconfigured function");

    _gaussian_pyr_function.run(); // compute gaussian pyramid

    for(unsigned int i = 0; i < _num_levels; ++i)
    {
        _convf[i].run(); // convolute gaussian pyramid
    }

    for(unsigned int i = 0; i < _num_levels; ++i)
    {
        _subf[i].run(); // compute laplacian image
    }

    _depth_function.run();
}
