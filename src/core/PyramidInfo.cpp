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
#include "arm_compute/core/PyramidInfo.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/TensorShape.h"

#include <cmath>

using namespace arm_compute;

PyramidInfo::PyramidInfo()
    : _num_levels(0), _tensor_shape(), _format(Format::UNKNOWN), _scale(0.0f)
{
}

PyramidInfo::PyramidInfo(size_t num_levels, float scale, size_t width, size_t height, Format format)
    : PyramidInfo()
{
    init(num_levels, scale, width, height, format);
}

PyramidInfo::PyramidInfo(size_t num_levels, float scale, const TensorShape &tensor_shape, Format format)
    : PyramidInfo()
{
    init(num_levels, scale, tensor_shape, format);
}

void PyramidInfo::init(size_t num_levels, float scale, size_t width, size_t height, Format format)
{
    init(num_levels, scale, TensorShape(width, height), format);
}

void PyramidInfo::init(size_t num_levels, float scale, const TensorShape &tensor_shape, Format format)
{
    ARM_COMPUTE_ERROR_ON(0 == num_levels);
    ARM_COMPUTE_ERROR_ON(0.0f == scale);
    ARM_COMPUTE_ERROR_ON(0 == tensor_shape.x());
    ARM_COMPUTE_ERROR_ON(0 == tensor_shape.y());
    ARM_COMPUTE_ERROR_ON(Format::IYUV == format);
    ARM_COMPUTE_ERROR_ON(Format::NV12 == format);
    ARM_COMPUTE_ERROR_ON(Format::NV21 == format);
    ARM_COMPUTE_ERROR_ON(Format::UYVY422 == format);
    ARM_COMPUTE_ERROR_ON(Format::YUV444 == format);
    ARM_COMPUTE_ERROR_ON(Format::YUYV422 == format);
    ARM_COMPUTE_ERROR_ON_MSG(0 != _num_levels, "PyramidInfo already initialized");
    ARM_COMPUTE_ERROR_ON(0 == (tensor_shape.x() * pow(scale, num_levels)));
    ARM_COMPUTE_ERROR_ON(0 == (tensor_shape.y() * pow(scale, num_levels)));

    _num_levels   = num_levels;
    _format       = format;
    _scale        = scale;
    _tensor_shape = tensor_shape;
}

size_t PyramidInfo::num_levels() const
{
    return _num_levels;
}

size_t PyramidInfo::width() const
{
    return _tensor_shape.x();
}

size_t PyramidInfo::height() const
{
    return _tensor_shape.y();
}

const TensorShape &PyramidInfo::tensor_shape() const
{
    return _tensor_shape;
}

Format PyramidInfo::format() const
{
    return _format;
}

float PyramidInfo::scale() const
{
    return _scale;
}
