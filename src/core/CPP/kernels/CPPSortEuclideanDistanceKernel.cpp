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
#include "arm_compute/core/CPP/kernels/CPPSortEuclideanDistanceKernel.h"

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <cmath>

using namespace arm_compute;

namespace
{
bool keypoint_compare(const InternalKeypoint &lhs, const InternalKeypoint &rhs)
{
    return std::get<2>(lhs) > std::get<2>(rhs);
}
} // namespace

CPPSortEuclideanDistanceKernel::CPPSortEuclideanDistanceKernel()
    : _num_corner_candidates(), _min_distance(0.0f), _in_out(nullptr), _output(nullptr)
{
}

void CPPSortEuclideanDistanceKernel::configure(InternalKeypoint *in_out, IKeyPointArray *output, const int32_t *num_corner_candidates, float min_distance)
{
    ARM_COMPUTE_ERROR_ON(nullptr == in_out);
    ARM_COMPUTE_ERROR_ON(nullptr == output);
    ARM_COMPUTE_ERROR_ON(nullptr == num_corner_candidates);
    ARM_COMPUTE_ERROR_ON(!((min_distance > 0) && (min_distance <= 30)));

    _in_out                = in_out;
    _output                = output;
    _min_distance          = min_distance * min_distance; // We compare squares of distances
    _num_corner_candidates = num_corner_candidates;
    ICPPKernel::configure(Window()); // Default 1 iteration window
}

bool CPPSortEuclideanDistanceKernel::is_parallelisable() const
{
    return false;
}

void CPPSortEuclideanDistanceKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_UNUSED(window);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_WINDOWS(ICPPKernel::window(), window);

    const int32_t num_corner_candidates = *_num_corner_candidates;

    /* Sort list of corner candidates */
    std::sort(_in_out, _in_out + num_corner_candidates, keypoint_compare);

    /* Euclidean distance */
    for(int32_t i = 0; i < num_corner_candidates; ++i)
    {
        if(std::get<2>(_in_out[i]) != 0.0f)
        {
            KeyPoint   keypt;
            const auto xc = std::get<0>(_in_out[i]);
            const auto yc = std::get<1>(_in_out[i]);

            keypt.x               = xc;
            keypt.y               = yc;
            keypt.strength        = std::get<2>(_in_out[i]);
            keypt.tracking_status = 1;

            /* Store corner */
            _output->push_back(keypt);
            for(int32_t k = i + 1; k < num_corner_candidates; ++k)
            {
                const float dx = std::fabs(std::get<0>(_in_out[k]) - xc);
                const float dy = std::fabs(std::get<1>(_in_out[k]) - yc);

                if((dx < _min_distance) && (dy < _min_distance))
                {
                    const float d = (dx * dx + dy * dy);

                    if(d < _min_distance)
                    {
                        /* Invalidate keypoint */
                        std::get<2>(_in_out[k]) = 0.0f;
                    }
                }
            }
        }
    }
}
