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
#include "arm_compute/core/CPP/kernels/CPPCornerCandidatesKernel.h"

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

using namespace arm_compute;

namespace
{
inline void check_corner(float x, float y, float strength, InternalKeypoint *output, int32_t *num_corner_candidates, arm_compute::Mutex *corner_candidates_mutex)
{
    if(strength != 0.0f)
    {
        /* Set index and update num_corner_candidate */
        std::unique_lock<arm_compute::Mutex> lock(*corner_candidates_mutex);

        const int32_t idx = *num_corner_candidates;

        *num_corner_candidates += 1;

        lock.unlock();

        /* Add keypoint */
        output[idx] = std::make_tuple(x, y, strength);
    }
}

inline void corner_candidates(const float *__restrict input, InternalKeypoint *__restrict output, int32_t x, int32_t y, int32_t *num_corner_candidates, arm_compute::Mutex *corner_candidates_mutex)
{
    check_corner(x, y, *input, output, num_corner_candidates, corner_candidates_mutex);
}
} // namespace

bool keypoint_compare(const InternalKeypoint &lhs, const InternalKeypoint &rhs)
{
    return std::get<2>(lhs) > std::get<2>(rhs);
}

CPPCornerCandidatesKernel::CPPCornerCandidatesKernel()
    : _num_corner_candidates(nullptr), _corner_candidates_mutex(), _input(nullptr), _output(nullptr)
{
}

void CPPCornerCandidatesKernel::configure(const IImage *input, InternalKeypoint *output, int32_t *num_corner_candidates)
{
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(input);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON(nullptr == output);
    ARM_COMPUTE_ERROR_ON(nullptr == num_corner_candidates);
    ARM_COMPUTE_ERROR_ON(*num_corner_candidates != 0);

    _input                 = input;
    _output                = output;
    _num_corner_candidates = num_corner_candidates;

    const unsigned int num_elems_processed_per_iteration = 1;

    // Configure kernel window
    Window win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));

    update_window_and_padding(win, AccessWindowHorizontal(input->info(), 0, num_elems_processed_per_iteration));

    INEKernel::configure(win);
}

void CPPCornerCandidatesKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    Iterator input(_input, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        corner_candidates(reinterpret_cast<float *>(input.ptr()), &_output[0], id.x(), id.y(), _num_corner_candidates, &_corner_candidates_mutex);
    },
    input);
}
