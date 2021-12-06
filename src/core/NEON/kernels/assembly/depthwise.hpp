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

#include "arm_gemm.hpp"
#include "arm_gemm_local.hpp"
#include "depthwise_common.hpp"

namespace arm_conv
{
namespace depthwise
{
struct DepthwiseConfig
{
    DepthwiseMethod method = DepthwiseMethod::DEFAULT;
    std::string     filter = "";

    DepthwiseConfig(DepthwiseMethod method)
        : method(method) {};
    DepthwiseConfig() {};
};

struct DepthwiseArgs
{
    const CPUInfo *cpu_info;

    unsigned int kernel_rows, kernel_cols;
    unsigned int stride_rows, stride_cols;

    unsigned int n_batches, input_rows, input_cols, input_channels;
    unsigned int output_rows, output_cols;
    unsigned int channel_multiplier;

    PaddingValues padding;

    arm_gemm::Activation activation;

    const DepthwiseConfig *config;

    DepthwiseArgs(
        const CPUInfo *cpu_info,
        unsigned int kernel_rows, unsigned int kernel_cols,
        unsigned int stride_rows, unsigned int stride_cols,
        unsigned int n_batches, unsigned int input_rows, unsigned int input_cols,
        unsigned int input_channels,
        unsigned int output_rows, unsigned int output_cols,
        unsigned int  channel_multiplier,
        PaddingValues padding, arm_gemm::Activation activation,
        const DepthwiseConfig *config)
        : cpu_info(cpu_info), kernel_rows(kernel_rows), kernel_cols(kernel_cols), stride_rows(stride_rows), stride_cols(stride_cols), n_batches(n_batches), input_rows(input_rows), input_cols(input_cols),
          input_channels(input_channels), output_rows(output_rows), output_cols(output_cols), channel_multiplier(channel_multiplier), padding(padding), activation(activation), config(config)
    {
    }
};

template <typename TInput, typename TWeight, typename TOutput>
class DepthwiseCommon : public IDepthwiseCommon
{
private:
    std::string _name{};

protected:
    const DepthwiseArgs m_args; // Copy of arguments
public:
    std::string name() const
    {
        return _name;
    }
    void set_name(const std::string &n)
    {
        _name = n;
    }
    DepthwiseCommon(const DepthwiseArgs &args)
        : m_args(args) {};
    DepthwiseCommon(DepthwiseCommon &) = delete;
    DepthwiseCommon &operator=(DepthwiseCommon &) = delete;

    void execute(
        const void *const  input,
        const void *const  parameters,
        void *const        output,
        void *const        working_space,
        const unsigned int thread_id,
        const unsigned int n_threads) const override
    {
        const size_t ld_input_col    = m_args.input_channels;
        const size_t ld_input_row    = ld_input_col * m_args.input_cols;
        const size_t ld_input_batch  = ld_input_row * m_args.input_rows;
        const size_t ld_output_col   = m_args.input_channels * m_args.channel_multiplier;
        const size_t ld_output_row   = ld_output_col * m_args.output_cols;
        const size_t ld_output_batch = ld_output_row * m_args.output_rows;

        execute(
            input, ld_input_col, ld_input_row, ld_input_batch,
            parameters, output, ld_output_col, ld_output_row, ld_output_batch,
            working_space, thread_id, n_threads);
    }

    void execute(
        const void *const  input,
        size_t             ld_input_col,
        size_t             ld_input_row,
        size_t             ld_input_batch,
        const void *const  parameters,
        void *const        output,
        size_t             ld_output_col,
        size_t             ld_output_row,
        size_t             ld_output_batch,
        void *const        working_space,
        const unsigned int thread_id,
        const unsigned int n_threads) const override
    {
        execute(
            m_args.n_batches, m_args.input_rows, m_args.input_cols,
            m_args.input_channels, m_args.padding,
            input, ld_input_col, ld_input_row, ld_input_batch,
            parameters,
            m_args.output_rows, m_args.output_cols,
            output, ld_output_col, ld_output_row, ld_output_batch,
            working_space, thread_id, n_threads);
    }

    virtual void execute(
        unsigned int batches,
        unsigned int input_height,
        unsigned int input_width,
        unsigned int channels,
        const PaddingValues &,
        const void *input,
        size_t       ld_input_col,
        size_t       ld_input_row,
        size_t       ld_input_batch,
        const void *parameters,
        unsigned int output_height,
        unsigned int output_width,
        void        *output,
        size_t       ld_output_col,
        size_t       ld_output_row,
        size_t       ld_output_batch,
        void        *working_space,
        unsigned int thread_id,
        unsigned int n_threads) const override = 0;
};

template <typename TInput, typename TWeight = TInput, typename TOutput = TInput>
using UniqueDepthwiseCommon = std::unique_ptr<DepthwiseCommon<TInput, TWeight, TOutput>>;

template <typename TInput, typename TWeight = TInput, typename TOutput = TInput, class OutputStage = Nothing>
KernelDescription get_depthwise_method(const DepthwiseArgs &, const OutputStage & = {});

template <typename TInput, typename TWeight = TInput, typename TOutput = TInput, class OutputStage = Nothing>
UniqueDepthwiseCommon<TInput, TWeight, TOutput> depthwise(const DepthwiseArgs &, const OutputStage & = {});

template <typename TInput, typename TWeight = TInput, typename TOutput = TInput, class OutputStage = Nothing>
std::vector<KernelDescription> get_compatible_kernels(const DepthwiseArgs &, const OutputStage & = {});

} // namespace depthwise
} // namespace arm_conv
