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
#include "arm_compute/runtime/CL/CLPyramid.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/PyramidInfo.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/TensorShape.h"
#include "support/ToolchainSupport.h"

#include <array>
#include <cmath>

using namespace arm_compute;

CLPyramid::CLPyramid()
    : _info(), _pyramid(nullptr)
{
}

void CLPyramid::init(const PyramidInfo &info)
{
    internal_init(info, false);
}

void CLPyramid::init_auto_padding(const PyramidInfo &info)
{
    internal_init(info, true);
}

void CLPyramid::internal_init(const PyramidInfo &info, bool auto_padding)
{
    _info    = info;
    _pyramid = arm_compute::support::cpp14::make_unique<CLTensor[]>(_info.num_levels());

    size_t      w            = _info.width();
    size_t      h            = _info.height();
    size_t      ref_w        = w;
    size_t      ref_h        = h;
    const bool  is_orb_scale = (SCALE_PYRAMID_ORB == _info.scale());
    TensorShape tensor_shape = _info.tensor_shape();

    // Note: Look-up table used by the OpenVX sample implementation
    const std::array<float, 4> c_orbscale =
    {
        {
            0.5f,
            SCALE_PYRAMID_ORB,
            SCALE_PYRAMID_ORB * SCALE_PYRAMID_ORB,
            SCALE_PYRAMID_ORB *SCALE_PYRAMID_ORB * SCALE_PYRAMID_ORB
        }
    };

    for(size_t i = 0; i < _info.num_levels(); ++i)
    {
        TensorInfo tensor_info(tensor_shape, _info.format());

        if(auto_padding)
        {
            tensor_info.auto_padding();
        }

        _pyramid[i].allocator()->init(tensor_info);

        if(is_orb_scale)
        {
            const float orb_scale = c_orbscale[(i + 1) % 4];
            w                     = std::ceil(ref_w * orb_scale);
            h                     = std::ceil(ref_h * orb_scale);

            if(0 == ((i + 1) % 4))
            {
                ref_w = w;
                ref_h = h;
            }
        }
        else
        {
            w = (w + 1) * _info.scale();
            h = (h + 1) * _info.scale();
        }

        // Update tensor_shape
        tensor_shape.set(0, w);
        tensor_shape.set(1, h);
    }
}

void CLPyramid::allocate()
{
    ARM_COMPUTE_ERROR_ON(_pyramid == nullptr);

    for(size_t i = 0; i < _info.num_levels(); ++i)
    {
        (_pyramid.get() + i)->allocator()->allocate();
    }
}

const PyramidInfo *CLPyramid::info() const
{
    return &_info;
}

CLTensor *CLPyramid::get_pyramid_level(size_t index) const
{
    ARM_COMPUTE_ERROR_ON(index >= _info.num_levels());

    return (_pyramid.get() + index);
}
