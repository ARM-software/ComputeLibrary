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
#ifndef __ARM_COMPUTE_NEGEMMWINOGRADLAYERKERNEL_H__
#define __ARM_COMPUTE_NEGEMMWINOGRADLAYERKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/NEON/kernels/winograd/tensor.hpp"

namespace arm_compute
{
class ITensor;
class NEWinogradLayerKernel;
class Winograd3x3F32
{
public:
    friend class NEWinogradLayerKernel;
    Winograd3x3F32(const KernelShape &kernel_shape, const Tensor4DShape input_shape, const PaddingType padding_type, void *kernel_storage);
    ~Winograd3x3F32();
    std::pair<void *, void *> get_nhwc_ptrs(const Tensor4DShape &input_shape, const PaddingType padding_type, void *working_space);
    void transform_weights(const void *const kernel, void *transform_working_space);
    void reshape_input(const Tensor4DShape &input_shape, const PaddingType padding_type, const void *const input, void *working_space);
    void reshape_output(const Tensor4DShape &input_shape, const PaddingType padding_type, void *const output);
    void nchw2nhwc(const Tensor4DShape &input_shape, const PaddingType padding_type, void *working_space, const void *const input);
    void nhwc2nchw(const Tensor4DShape &input_shape, const PaddingType padding_type, void *working_space, void *const output);

private:
    class Private;
    std::unique_ptr<Private> _pimpl;
};

class NEWinogradLayerKernel : public INEKernel
{
public:
    /** Constructor */
    NEWinogradLayerKernel();

    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEWinogradLayerKernel(const NEWinogradLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEWinogradLayerKernel &operator=(const NEWinogradLayerKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEWinogradLayerKernel(NEWinogradLayerKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEWinogradLayerKernel &operator=(NEWinogradLayerKernel &&) = default;

    virtual ~NEWinogradLayerKernel() = default;

    /** Initialise the kernel
     *
     * @param[in,out] output    Output tensor to store the result of matrix multiplication. Data type supported: F32.
     * @param[in]     convolver A pointer to the winograd convolver, this object must have been configured and is ready to execute 16 GEMMS .
     */
    void configure(ITensor *output, Winograd3x3F32 *convolver);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

    /* Get the memory required to instantiate a new Winograd operator.
       */
    static size_t get_kernel_storage_size(const KernelShape &shape);

    /* Get the memory required to apply a Winograd operator to some input.
       */
    static size_t get_working_space_size(const Tensor4DShape &input_shape, const KernelShape &k_shape, const PaddingType padding);

    /* Get the memory required to transform the kernel.
       */
    static size_t get_kernel_transform_working_size(const KernelShape &shape);

protected:
    Winograd3x3F32 *_convolver;
    ITensor        *_output;
};

} // namespace arm_compute
#endif /*__ARM_COMPUTE_NEGEMMWINOGRADLAYERKERNEL_H__*/
