/*
 * Copyright (c) 2021-2022 Arm Limited.
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

#pragma once
#ifdef CYCLE_PROFILING
#include "profiler.hpp"
#endif

namespace arm_conv
{
namespace pooling
{
enum class PoolingType
{
    AVERAGE,
    MAX,
};

enum class PoolingMethod
{
    DEFAULT,
    DEPTHFIRST,
    PLANAR,
};

struct PoolingWindow
{
    unsigned int rows, cols;
};

struct PoolingStride
{
    unsigned int rows, cols;
};

struct PaddingValues
{
    unsigned int left, top, right, bottom;
};

class IPoolingCommon
{
public:
    virtual ~IPoolingCommon() = default;

    // Determine the amount of working space required.
    virtual size_t get_working_size(unsigned int num_threads) const = 0;
    virtual size_t get_working_size(unsigned int num_threads, unsigned int n_channels) const = 0;

    // Execute pooling over the specified area of memory.
    virtual void execute(
        const void *const input,
        void *const       output,
        void             *working_space,
        unsigned int      thread_id,
        unsigned int      num_threads) const = 0;

    virtual void execute(
        const void *const input,
        size_t            ld_input_col,
        size_t            ld_input_row,
        size_t            ld_input_batch,
        void *const       output,
        size_t            ld_output_col,
        size_t            ld_output_row,
        size_t            ld_output_batch,
        void             *working_space,
        unsigned int      thread_id,
        unsigned int      num_threads) const = 0;

    virtual void execute(
        unsigned int      batches,
        unsigned int      height,
        unsigned int      width,
        unsigned int      channels,
        const void *const input,
        size_t            ld_input_col,
        size_t            ld_input_row,
        size_t            ld_input_batch,
        const PaddingValues &,
        unsigned int output_height,
        unsigned int output_width,
        void *const  output,
        size_t       ld_output_col,
        size_t       ld_output_row,
        size_t       ld_output_batch,
        void        *working_space,
        unsigned int thread_id,
        unsigned int num_threads) const = 0;
};

} // namespace pooling
} // namespace arm_conv
