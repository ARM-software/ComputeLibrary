/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#include "arm_compute/runtime/NEON/functions/NELaplacianPyramid.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IPyramid.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/functions/NEArithmeticSubtraction.h"
#include "arm_compute/runtime/NEON/functions/NEDepthConvert.h"
#include "arm_compute/runtime/NEON/functions/NEGaussian5x5.h"
#include "arm_compute/runtime/NEON/functions/NEGaussianPyramid.h"
#include "arm_compute/runtime/Tensor.h"

using namespace arm_compute;

NELaplacianPyramid::NELaplacianPyramid()
    : _num_levels(0), _gaussian_pyr_function(), _convf(), _subf(), _gauss_pyr(), _conv_pyr(), _depth_function()
{
}

void NELaplacianPyramid::run()
{
    ARM_COMPUTE_ERROR_ON_MSG(0 == _num_levels, "Unconfigured function");

    // Compute Gaussian Pyramid
    _gaussian_pyr_function.run();

    for(unsigned int i = 0; i < _num_levels; ++i)
    {
        // Apply Gaussian filter to gaussian pyramid image
        _convf[i].run();
    }

    for(unsigned int i = 0; i < _num_levels; ++i)
    {
        // Compute laplacian image
        _subf[i].run();
    }

    _depth_function.run();
}

void NELaplacianPyramid::configure(const ITensor *input, IPyramid *pyramid, ITensor *output, BorderMode border_mode, uint8_t constant_border_value)
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

    _convf = arm_compute::cpp14::make_unique<NEGaussian5x5[]>(_num_levels);
    _subf  = arm_compute::cpp14::make_unique<NEArithmeticSubtraction[]>(_num_levels);

    for(unsigned int i = 0; i < _num_levels; ++i)
    {
        _convf[i].configure(_gauss_pyr.get_pyramid_level(i), _conv_pyr.get_pyramid_level(i), border_mode, constant_border_value);
        _subf[i].configure(_gauss_pyr.get_pyramid_level(i), _conv_pyr.get_pyramid_level(i), pyramid->get_pyramid_level(i), ConvertPolicy::WRAP);
    }

    _depth_function.configure(_conv_pyr.get_pyramid_level(_num_levels - 1), output, ConvertPolicy::WRAP, 0);

    _gauss_pyr.allocate();
    _conv_pyr.allocate();
}
