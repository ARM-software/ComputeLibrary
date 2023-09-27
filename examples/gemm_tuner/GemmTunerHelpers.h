/*
 * Copyright (c) 2020-2021 Arm Limited.
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
#ifndef EXAMPLES_GEMMTUNERHELPERS_H
#define EXAMPLES_GEMMTUNERHELPERS_H

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"

namespace examples
{
namespace gemm_tuner_helpers
{
bool update_padding_for_cl_image(arm_compute::ITensorInfo *tensor)
{
    constexpr unsigned int num_floats_per_pixel = 4;

    const unsigned int stride_y_in_elements = tensor->strides_in_bytes()[1] / tensor->element_size();
    const unsigned int pixel_aligment =
        arm_compute::get_cl_image_pitch_alignment(arm_compute::CLKernelLibrary::get().get_device());
    if (pixel_aligment == 0)
    {
        return false;
    }
    const unsigned int row_pitch_alignment = pixel_aligment * num_floats_per_pixel;
    const unsigned int round_up_width =
        ((stride_y_in_elements + row_pitch_alignment - 1) / row_pitch_alignment) * row_pitch_alignment;
    const unsigned int padding = round_up_width - stride_y_in_elements;

    tensor->extend_padding(arm_compute::PaddingSize(0, padding, 0, 0));
    return true;
}
} // namespace gemm_tuner_helpers
} // namespace examples

#endif /* EXAMPLES_GEMMTUNERHELPERS_H */
