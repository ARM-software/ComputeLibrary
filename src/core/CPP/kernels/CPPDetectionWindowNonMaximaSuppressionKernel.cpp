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
#include "arm_compute/core/CPP/kernels/CPPDetectionWindowNonMaximaSuppressionKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"

#include <algorithm>
#include <cmath>

using namespace arm_compute;

namespace
{
bool compare_detection_window(const DetectionWindow &lhs, const DetectionWindow &rhs)
{
    return lhs.score > rhs.score;
}
} // namespace

CPPDetectionWindowNonMaximaSuppressionKernel::CPPDetectionWindowNonMaximaSuppressionKernel()
    : _input_output(nullptr), _min_distance(0.0f)
{
}

bool CPPDetectionWindowNonMaximaSuppressionKernel::is_parallelisable() const
{
    return false;
}

void CPPDetectionWindowNonMaximaSuppressionKernel::configure(IDetectionWindowArray *input_output, float min_distance)
{
    ARM_COMPUTE_ERROR_ON(nullptr == input_output);

    _input_output = input_output;
    _min_distance = min_distance;

    IKernel::configure(Window()); // Default 1 iteration window
}

void CPPDetectionWindowNonMaximaSuppressionKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_UNUSED(window);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_WINDOWS(IKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_input_output->buffer() == nullptr);

    const size_t num_candidates = _input_output->num_values();
    size_t       num_detections = 0;

    // Sort list of candidates
    std::sort(_input_output->buffer(), _input_output->buffer() + num_candidates, compare_detection_window);

    const float min_distance_pow2 = _min_distance * _min_distance;

    // Euclidean distance
    for(size_t i = 0; i < num_candidates; ++i)
    {
        if(0.0f != _input_output->at(i).score)
        {
            DetectionWindow cur;
            cur.x         = _input_output->at(i).x;
            cur.y         = _input_output->at(i).y;
            cur.width     = _input_output->at(i).width;
            cur.height    = _input_output->at(i).height;
            cur.idx_class = _input_output->at(i).idx_class;
            cur.score     = _input_output->at(i).score;

            // Store window
            _input_output->at(num_detections) = cur;

            ++num_detections;

            const float xc = cur.x + cur.width * 0.5f;
            const float yc = cur.y + cur.height * 0.5f;

            for(size_t k = i + 1; k < num_candidates; ++k)
            {
                const float xn = _input_output->at(k).x + _input_output->at(k).width * 0.5f;
                const float yn = _input_output->at(k).y + _input_output->at(k).height * 0.5f;

                const float dx = std::fabs(xn - xc);
                const float dy = std::fabs(yn - yc);

                if(dx < _min_distance && dy < _min_distance)
                {
                    const float d = dx * dx + dy * dy;

                    if(d < min_distance_pow2)
                    {
                        // Invalidate keypoint
                        _input_output->at(k).score = 0.0f;
                    }
                }
            }
        }
    }

    _input_output->resize(num_detections);
}
