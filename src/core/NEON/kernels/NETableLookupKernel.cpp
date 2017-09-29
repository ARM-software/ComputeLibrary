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
#include "arm_compute/core/NEON/kernels/NETableLookupKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ILut.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"

#include <cstddef>
#include <cstdint>

using namespace arm_compute;

namespace arm_compute
{
class Coordinates;

constexpr unsigned int num_num_elems_processed_per_iteration = 16;
} // namespace arm_compute

NETableLookupKernel::NETableLookupKernel()
    : _func(nullptr), _lut(nullptr)
{
}

template <class T>
void NETableLookupKernel::tableLookup(const Window &window)
{
    uint32_t     offset = _lut->index_offset();
    size_t       count  = _lut->num_elements();
    const auto   lut    = reinterpret_cast<const T *>(_lut->buffer());
    unsigned int step   = num_num_elems_processed_per_iteration;

    ARM_COMPUTE_ERROR_ON(lut == nullptr);

    Iterator input  = Iterator(_input, window);
    Iterator output = Iterator(_output, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        auto input_ptr  = reinterpret_cast<const T *>(input.ptr());
        auto output_ptr = reinterpret_cast<T *>(output.ptr());

        for(unsigned int i = 0; i < step; ++i, ++input_ptr, ++output_ptr)
        {
            const int32_t index = offset + *input_ptr;

            if(0 <= index && index < static_cast<int32_t>(count))
            {
                *output_ptr = lut[index];
            }
        }
    },
    input, output);
}

namespace arm_compute
{
template <>
void NETableLookupKernel::tableLookup<uint8_t>(const Window &window)
{
    const uint8_t *const lut  = _lut->buffer();
    unsigned int         step = num_num_elems_processed_per_iteration;

    ARM_COMPUTE_ERROR_ON(lut == nullptr);

    Iterator input  = Iterator(_input, window);
    Iterator output = Iterator(_output, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const uint8_t *input_ptr  = input.ptr();
        uint8_t       *output_ptr = output.ptr();

        for(unsigned int i = 0; i < step; ++i)
        {
            *output_ptr++ = lut[*input_ptr++];
        }
    },
    input, output);
}
} // namespace arm_compute

void NETableLookupKernel::configure(const ITensor *input, const ILut *lut, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON(input == nullptr);
    ARM_COMPUTE_ERROR_ON(lut == nullptr);
    ARM_COMPUTE_ERROR_ON(output == nullptr);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8, DataType::S16);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8, DataType::S16);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);

    _lut = lut;

    if(input->info()->data_type() == DataType::U8 && output->info()->data_type() == DataType::U8)
    {
        _func = &NETableLookupKernel::tableLookup<uint8_t>;
    }
    else if(input->info()->data_type() == DataType::S16 && output->info()->data_type() == DataType::S16)
    {
        _func = &NETableLookupKernel::tableLookup<int16_t>;
    }
    else
    {
        ARM_COMPUTE_ERROR("Unsupported combination of input and output DataType.");
    }

    INESimpleKernel::configure(input, output, num_num_elems_processed_per_iteration);
}

void NETableLookupKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INESimpleKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);
    (this->*_func)(window);
}
