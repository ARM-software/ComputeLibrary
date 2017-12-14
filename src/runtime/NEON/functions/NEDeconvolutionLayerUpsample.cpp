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
#include "arm_compute/runtime/NEON/functions/NEDeconvolutionLayerUpsample.h"

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/kernels/NEDeconvolutionLayerUpsampleKernel.h"
#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "support/ToolchainSupport.h"

#include <cmath>
#include <cstddef>
#include <utility>

using namespace arm_compute;

namespace
{
inline void precompute_offsets(ITensor *offsets, float wr, size_t input_element_size, const std::pair<unsigned int, unsigned int> &a,
                               const std::pair<unsigned int, unsigned int> &iz, const PadStrideInfo &info)
{
    ARM_COMPUTE_ERROR_ON(nullptr == offsets);
    Window    win;
    const int padx          = info.pad().first;
    const int pady          = info.pad().second;
    const int ax            = a.first;
    const int ay            = a.second;
    const int offset_width  = offsets->info()->dimension(0);
    const int offset_height = offsets->info()->dimension(1);
    // The values of ax and ay denote the number of ZEROS to be added on the top and right inner border of the image.
    // Step value along the XY axis will depend on the number of zeros to be inserted between samples (number of zeros + 1).
    // Pre-compute the X offset, Y's stride is unknown at this point so we can't precompute Y's offsets
    for(int yi = ay; yi < (offset_height - pady); yi += (1 + iz.second))
    {
        for(int xi = padx; xi < (offset_width - ax); xi += (1 + iz.first))
        {
            int         *ptr                  = reinterpret_cast<int *>(offsets->ptr_to_element(Coordinates(xi, yi)));
            const size_t in_xi                = (xi + 0.5f) * wr;
            *reinterpret_cast<int32_t *>(ptr) = in_xi * input_element_size;
        }
    }
}
} // namespace

NEDeconvolutionLayerUpsample::NEDeconvolutionLayerUpsample(std::shared_ptr<IMemoryManager> memory_manager) // NOLINT
    : _memory_group(std::move(memory_manager)),
      _offsets(),
      _border_handler(),
      _upsample()
{
}

void NEDeconvolutionLayerUpsample::configure(ITensor *input, ITensor *output, const std::pair<unsigned int, unsigned int> &a,
                                             const std::pair<unsigned int, unsigned int> &iz, const PadStrideInfo &info)
{
    ARM_COMPUTE_ERROR_ON(nullptr == input);
    ARM_COMPUTE_ERROR_ON(nullptr == output);

    for(size_t i = 2; i < Coordinates::num_max_dimensions; ++i)
    {
        ARM_COMPUTE_ERROR_ON(input->info()->dimension(i) != output->info()->dimension(i));
    }

    // Get the tensor shape
    const TensorShape shape(output->info()->dimension(0), output->info()->dimension(1));

    // Compute the ratio between source width/height and destination width/height
    const auto wr = static_cast<float>(input->info()->dimension(0)) / static_cast<float>(output->info()->dimension(0));
    const auto hr = static_cast<float>(input->info()->dimension(1)) / static_cast<float>(output->info()->dimension(1));
    ARM_COMPUTE_UNUSED(hr);
    // Get the element size of the input image
    const size_t input_element_size = input->info()->element_size();

    TensorInfo tensor_info_offsets(shape, Format::S32);
    _offsets.allocator()->init(tensor_info_offsets);

    _upsample.configure(input, &_offsets, output);

    // Allocate once the configure methods have been called
    _offsets.allocator()->allocate();
    // Pre-compute offsets for nearest interpolation
    std::fill_n(reinterpret_cast<int32_t *>(_offsets.buffer()), _offsets.info()->total_size() / sizeof(int32_t), -1 * input_element_size);
    precompute_offsets(&_offsets, wr, input_element_size, a, iz, info);

    _border_handler.configure(input, _upsample.border_size(), BorderMode::CONSTANT, PixelValue(0.f));
}

void NEDeconvolutionLayerUpsample::run()
{
    NEScheduler::get().schedule(&_border_handler, Window::DimZ);
    _memory_group.acquire();
    NEScheduler::get().schedule(&_upsample, Window::DimY);
    _memory_group.release();
}
