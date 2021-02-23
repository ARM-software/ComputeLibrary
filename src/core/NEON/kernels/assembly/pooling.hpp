/*
 * Copyright (c) 2021 Arm Limited.
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

#include "arm_gemm_local.hpp"
#include "pool_common.hpp"

#include <memory>

namespace arm_conv
{
namespace pooling
{
struct PoolingConfig
{
    PoolingMethod method = PoolingMethod::DEFAULT;
    std::string   filter = "";

    PoolingConfig(PoolingMethod method)
        : method(method) {};
    PoolingConfig() {};
};

struct PoolingArgs
{
    const CPUInfo *cpu_info;

    PoolingType   pool_type;
    PoolingWindow pool_window;
    PoolingStride pool_stride;
    bool          exclude_padding;

    unsigned int n_batches, input_rows, input_cols, n_channels;
    unsigned int output_rows, output_cols;

    PaddingValues padding;

    const PoolingConfig *config;

    PoolingArgs(
        const CPUInfo       *cpu_info,
        PoolingType          pool_type,
        const PoolingWindow &window,
        const PoolingStride &stride,
        bool                 exclude_padding,
        unsigned int         n_batches,
        unsigned int         input_rows,
        unsigned int         input_cols,
        unsigned int         n_channels,
        unsigned int         output_rows,
        unsigned int         output_cols,
        const PaddingValues &padding,
        const PoolingConfig *cfg)
        : cpu_info(cpu_info), pool_type(pool_type), pool_window(window), pool_stride(stride), exclude_padding(exclude_padding), n_batches(n_batches), input_rows(input_rows), input_cols(input_cols),
          n_channels(n_channels), output_rows(output_rows), output_cols(output_cols), padding(padding), config(cfg)
    {
        // If either of the pooling window dimensions are set to zero, meaning
        // "pool everything", then replace with the corresponding input dimension.
        if(pool_window.rows == 0)
        {
            pool_window.rows = input_rows;
        }
        if(pool_window.cols == 0)
        {
            pool_window.cols = input_cols;
        }
    }
};

struct Requantize32
{
    int32_t input_offset  = 0;
    int32_t output_offset = 0;

    int32_t per_layer_left_shift  = 0;
    int32_t per_layer_right_shift = 0;
    int32_t per_layer_mul         = 0;

    Requantize32(int32_t input_offset, int32_t output_offset,
                 int32_t per_layer_left_shift, int32_t per_layer_right_shift,
                 int32_t per_layer_mul)
        : input_offset(input_offset), output_offset(output_offset), per_layer_left_shift(per_layer_left_shift), per_layer_right_shift(per_layer_right_shift), per_layer_mul(per_layer_mul)
    {
    }
};

template <typename TInput, typename TOutput, class OutputStage = Nothing>
using UniquePoolingCommon = std::unique_ptr<PoolingCommon<TInput, TOutput, OutputStage>>;

// Get a pooling engine
template <typename TInput, typename TOutput = TInput, class OutputStage = Nothing>
UniquePoolingCommon<TInput, TOutput, OutputStage> pooling(const PoolingArgs &, const OutputStage & = {});

} // namespace pooling
} // namespace arm_conv
