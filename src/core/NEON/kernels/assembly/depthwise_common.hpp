/*
 * Copyright (c) 2021-2023 Arm Limited.
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
#include "common.hpp"
#include <cstddef>
#include <tuple>

namespace arm_conv
{
namespace depthwise
{
using arm_gemm::Nothing;

enum class DepthwiseMethod
{
    DEFAULT,
    DEPTHFIRST,
    PLANAR,
};

struct KernelDescription
{
    DepthwiseMethod method         = DepthwiseMethod::DEFAULT;
    std::string     name           = "";
    bool            is_default     = false;
    uint64_t        cycle_estimate = 0;

    KernelDescription(
        DepthwiseMethod method,
        std::string     name,
        bool            is_default,
        uint64_t        cycle_estimate)
        : method(method), name(name), is_default(is_default), cycle_estimate(cycle_estimate)
    {
    }

    KernelDescription() noexcept {};
};

class IDepthwiseCommon
{
public:
    virtual ~IDepthwiseCommon() = default;

    // Get the name of the depthwise implementation
    virtual std::string name() const = 0;

    // Determine the amount of storage space required for the rearranged weights
    // and bias.
    virtual size_t get_storage_size(void) const = 0;

    // Rearrange the weights and biases into a storage buffer.
    // Accepts a pointer to a buffer into which to store the packed parameters, a
    // pointer the bias vector (which may be nullptr in the case of no bias) and
    // a pointer to the array of weights (stored in HWIO order).
    virtual void pack_parameters(
        void       *buffer,
        const void *biases,
        const void *weights,
        size_t      ld_weight_col = 0,
        size_t      ld_weight_row = 0) = 0;

    // Determine the amount of working space required
    virtual size_t get_working_size(unsigned int n_threads, unsigned int n_input_channels) const = 0;

    // Execute the convolution over the specified area of memory.
    virtual void execute(
        const void *input,       // Pointer to input tensor
        const void *parameters,  // Packed parameters buffer
        void        *output,
        void        *working_space,
        unsigned int thread_id,
        unsigned int n_threads) const = 0;

    virtual void execute(
        const void *input,
        size_t       ld_input_col,
        size_t       ld_input_row,
        size_t       ld_input_batch,
        const void *parameters,
        void        *output,
        size_t       ld_output_col,
        size_t       ld_output_row,
        size_t       ld_output_batch,
        void        *working_space,
        unsigned int thread_id,
        unsigned int n_threads) const = 0;

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
        unsigned int n_threads) const = 0;
};

// To handle a dilation factor of D execute the kernel once for each d in
// [0..D). Each `d` corresponds to a portion or "view" of the input and output
// tensors. The output view corresponds to every Dth pixel starting from `d`;
// this function computes how many pixels are covered. The input view consists
// of an amount of before padding, every Dth pixel starting from an offset, and
// some after padding.  This function computes the start padding, input offset,
// number of valid input pixels, and the after padding.
//
// Returns
// - Number of valid output pixels corresponding to `d`
// - Number of valid input pixels corresponding to `d`
// - Offset of the first pixel corresponding to `d`
// - Amount of padding in the view for `d`
std::tuple<size_t, size_t, size_t, size_t, size_t>
get_reduced_view_for_dilation(
    size_t out_size, size_t in_size,
    size_t d, size_t dilation_factor,
    size_t kernel_size, size_t stride,
    size_t pad_before);

} // namespace depthwise
} // namespace arm_conv
