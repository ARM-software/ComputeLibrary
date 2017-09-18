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
#include "arm_compute/runtime/NEON/functions/NEHarrisCorners.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/NEON/kernels/NEFillBorderKernel.h"
#include "arm_compute/core/NEON/kernels/NEHarrisCornersKernel.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/Array.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/NEON/functions/NESobel3x3.h"
#include "arm_compute/runtime/NEON/functions/NESobel5x5.h"
#include "arm_compute/runtime/NEON/functions/NESobel7x7.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "support/ToolchainSupport.h"

#include <cmath>
#include <utility>

using namespace arm_compute;

NEHarrisCorners::NEHarrisCorners(std::shared_ptr<IMemoryManager> memory_manager) // NOLINT
    : _memory_group(std::move(memory_manager)),
      _sobel(),
      _harris_score(),
      _non_max_suppr(),
      _candidates(),
      _sort_euclidean(),
      _border_gx(),
      _border_gy(),
      _gx(),
      _gy(),
      _score(),
      _nonmax(),
      _corners_list(),
      _num_corner_candidates(0)
{
}

void NEHarrisCorners::configure(IImage *input, float threshold, float min_dist,
                                float sensitivity, int32_t gradient_size, int32_t block_size, KeyPointArray *corners,
                                BorderMode border_mode, uint8_t constant_border_value, bool use_fp16)
{
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(input);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON(!(block_size == 3 || block_size == 5 || block_size == 7));

    const TensorShape shape = input->info()->tensor_shape();
    TensorInfo        tensor_info_gxgy;

    if(gradient_size < 7)
    {
        tensor_info_gxgy.init(shape, Format::S16);
    }
    else
    {
        tensor_info_gxgy.init(shape, Format::S32);
    }

    _gx.allocator()->init(tensor_info_gxgy);
    _gy.allocator()->init(tensor_info_gxgy);

    // Manage intermediate buffers
    _memory_group.manage(&_gx);
    _memory_group.manage(&_gy);

    TensorInfo tensor_info_score(shape, Format::F32);
    _score.allocator()->init(tensor_info_score);
    _nonmax.allocator()->init(tensor_info_score);

    _corners_list = arm_compute::support::cpp14::make_unique<InternalKeypoint[]>(shape.x() * shape.y());

    // Set/init Sobel kernel accordingly with gradient_size
    switch(gradient_size)
    {
        case 3:
        {
            auto k = arm_compute::support::cpp14::make_unique<NESobel3x3>();
            k->configure(input, &_gx, &_gy, border_mode, constant_border_value);
            _sobel = std::move(k);
            break;
        }
        case 5:
        {
            auto k = arm_compute::support::cpp14::make_unique<NESobel5x5>();
            k->configure(input, &_gx, &_gy, border_mode, constant_border_value);
            _sobel = std::move(k);
            break;
        }
        case 7:
        {
            auto k = arm_compute::support::cpp14::make_unique<NESobel7x7>();
            k->configure(input, &_gx, &_gy, border_mode, constant_border_value);
            _sobel = std::move(k);
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Gradient size not implemented");
    }

    // Normalization factor
    const float norm_factor = 1.0f / (255.0f * pow(4.0f, gradient_size / 2) * block_size);

    // Manage intermediate buffers
    _memory_group.manage(&_score);

    if(use_fp16)
    {
        switch(block_size)
        {
            case 3:
            {
                auto k = arm_compute::support::cpp14::make_unique<NEHarrisScoreFP16Kernel<3>>();
                k->configure(&_gx, &_gy, &_score, norm_factor, threshold, sensitivity, border_mode == BorderMode::UNDEFINED);
                _harris_score = std::move(k);
            }
            break;
            case 5:
            {
                auto k = arm_compute::support::cpp14::make_unique<NEHarrisScoreFP16Kernel<5>>();
                k->configure(&_gx, &_gy, &_score, norm_factor, threshold, sensitivity, border_mode == BorderMode::UNDEFINED);
                _harris_score = std::move(k);
            }
            break;
            case 7:
            {
                auto k = arm_compute::support::cpp14::make_unique<NEHarrisScoreFP16Kernel<7>>();
                k->configure(&_gx, &_gy, &_score, norm_factor, threshold, sensitivity, border_mode == BorderMode::UNDEFINED);
                _harris_score = std::move(k);
            }
            default:
                break;
        }
    }
    else
    {
        // Set/init Harris Score kernel accordingly with block_size
        switch(block_size)
        {
            case 3:
            {
                auto k = arm_compute::support::cpp14::make_unique<NEHarrisScoreKernel<3>>();
                k->configure(&_gx, &_gy, &_score, norm_factor, threshold, sensitivity, border_mode == BorderMode::UNDEFINED);
                _harris_score = std::move(k);
            }
            break;
            case 5:
            {
                auto k = arm_compute::support::cpp14::make_unique<NEHarrisScoreKernel<5>>();
                k->configure(&_gx, &_gy, &_score, norm_factor, threshold, sensitivity, border_mode == BorderMode::UNDEFINED);
                _harris_score = std::move(k);
            }
            break;
            case 7:
            {
                auto k = arm_compute::support::cpp14::make_unique<NEHarrisScoreKernel<7>>();
                k->configure(&_gx, &_gy, &_score, norm_factor, threshold, sensitivity, border_mode == BorderMode::UNDEFINED);
                _harris_score = std::move(k);
            }
            default:
                break;
        }
    }

    // Configure border filling before harris score
    _border_gx.configure(&_gx, _harris_score->border_size(), border_mode, constant_border_value);
    _border_gy.configure(&_gy, _harris_score->border_size(), border_mode, constant_border_value);

    // Allocate once all the configure methods have been called
    _gx.allocator()->allocate();
    _gy.allocator()->allocate();

    // Manage intermediate buffers
    _memory_group.manage(&_nonmax);

    // Init non-maxima suppression function
    _non_max_suppr.configure(&_score, &_nonmax, border_mode);

    // Allocate once all the configure methods have been called
    _score.allocator()->allocate();

    // Init corner candidates kernel
    _candidates.configure(&_nonmax, _corners_list.get(), &_num_corner_candidates);

    // Allocate once all the configure methods have been called
    _nonmax.allocator()->allocate();

    // Init euclidean distance
    _sort_euclidean.configure(_corners_list.get(), corners, &_num_corner_candidates, min_dist);
}

void NEHarrisCorners::run()
{
    ARM_COMPUTE_ERROR_ON_MSG(_sobel == nullptr, "Unconfigured function");

    _memory_group.acquire();

    // Init to 0 number of corner candidates
    _num_corner_candidates = 0;

    // Run Sobel kernel
    _sobel->run();

    // Fill border before harris score kernel
    NEScheduler::get().schedule(&_border_gx, Window::DimZ);
    NEScheduler::get().schedule(&_border_gy, Window::DimZ);

    // Run harris score kernel
    NEScheduler::get().schedule(_harris_score.get(), Window::DimY);

    // Run non-maxima suppression
    _non_max_suppr.run();

    // Run corner candidate kernel
    NEScheduler::get().schedule(&_candidates, Window::DimY);

    // Run sort & euclidean distance
    NEScheduler::get().schedule(&_sort_euclidean, Window::DimY);

    _memory_group.release();
}
