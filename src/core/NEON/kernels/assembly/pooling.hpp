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

#include "arm_gemm_local.hpp"
#include "pool_common.hpp"

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

struct Nothing
{
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

template <typename TInput, typename TOutput>
class PoolingCommon : public IPoolingCommon
{
protected:
    const PoolingArgs m_args;

public:
    PoolingCommon(const PoolingArgs &args)
        : m_args(args)
    {
    }
    PoolingCommon(PoolingCommon &) = delete;
    PoolingCommon &operator=(PoolingCommon &) = delete;

    size_t get_working_size(unsigned int, unsigned int) const override = 0;
    size_t get_working_size(unsigned int n_threads) const override
    {
        return this->get_working_size(n_threads, m_args.n_channels);
    }

    // Execute pooling over the specified area of memory.
    void execute(
        const void *const input,
        void *const       output,
        void             *working_space,
        unsigned int      thread_id,
        unsigned int      num_threads) const override
    {
        this->execute(
            input,
            m_args.n_channels,
            m_args.n_channels * m_args.input_cols,
            m_args.n_channels * m_args.input_cols * m_args.input_rows,
            output,
            m_args.n_channels,
            m_args.n_channels * m_args.output_cols,
            m_args.n_channels * m_args.output_cols * m_args.output_rows,
            working_space,
            thread_id, num_threads);
    }

    void execute(
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
        unsigned int      num_threads) const override
    {
        this->execute(
            m_args.n_batches, m_args.input_rows, m_args.input_cols, m_args.n_channels,
            input, ld_input_col, ld_input_row, ld_input_batch,
            m_args.padding, m_args.output_rows, m_args.output_cols,
            output, ld_output_col, ld_output_row, ld_output_batch,
            working_space, thread_id, num_threads);
    }

    void execute(
        unsigned int         batches,
        unsigned int         height,
        unsigned int         width,
        unsigned int         channels,
        const void *const    input,
        size_t               ld_input_col,
        size_t               ld_input_row,
        size_t               ld_input_batch,
        const PaddingValues &padding,
        unsigned int         output_height,
        unsigned int         output_width,
        void *const          output,
        size_t               ld_output_col,
        size_t               ld_output_row,
        size_t               ld_output_batch,
        void                *working_space,
        unsigned int         thread_id,
        unsigned int         num_threads) const override
    {
        this->execute_internal(
            batches, height, width, channels, padding,
            input, ld_input_col, ld_input_row, ld_input_batch,
            output_height, output_width,
            output, ld_output_col, ld_output_row, ld_output_batch,
            working_space, thread_id, num_threads);
    }

protected:
    virtual void execute_internal(
        unsigned int batches,
        unsigned int height,
        unsigned int width,
        unsigned int channels,
        const PaddingValues &,
        const void *const input,
        size_t            ld_input_col,
        size_t            ld_input_row,
        size_t            ld_input_batch,
        unsigned int      output_height,
        unsigned int      output_width,
        void *const       output,
        size_t            ld_output_col,
        size_t            ld_output_row,
        size_t            ld_output_batch,
        void             *working_space,
        unsigned int      thread_id,
        unsigned int      num_threads) const = 0;
};

template <typename TInput, typename TOutput>
using UniquePoolingCommon = std::unique_ptr<PoolingCommon<TInput, TOutput>>;

// Get a pooling engine
template <typename TInput, typename TOutput = TInput, class OutputStage = Nothing>
UniquePoolingCommon<TInput, TOutput> pooling(const PoolingArgs &, const OutputStage & = {});

} // namespace pooling
} // namespace arm_conv
