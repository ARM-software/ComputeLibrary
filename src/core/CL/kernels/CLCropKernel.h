/*
 * Copyright (c) 2019-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_CLCROPKERNEL_H
#define ARM_COMPUTE_CLCROPKERNEL_H

#include "arm_compute/core/Types.h"
#include "src/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** OpenCL kernel to perform a copy between two tensors */
class CLCropKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLCropKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers). */
    CLCropKernel(const CLCropKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers). */
    CLCropKernel &operator=(const CLCropKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLCropKernel(CLCropKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLCropKernel &operator=(CLCropKernel &&) = default;
    /** Configure kernel
     *
     * @note Supported tensor rank: up to 4
     *
     * @param[in]  input               Source tensor. Data type supported: All. Data layouts supported: NHWC.
     * @param[out] output              Destination tensor. Data type supported: F32
     * @param[in]  start               Coordinates of where to start cropping the image.
     * @param[in]  end                 Coordinates of where to end cropping the image.
     * @param[in]  batch_index         Fourth dimension index of the 3D image to crop in @p input.
     * @param[in]  extrapolation_value Value to be used for values outside of the image. Default is 0.
     * @param[in]  output_window       Output window to be used in case cropped image is being copied into a tensor. Default is nullptr.
     */
    void configure(const ICLTensor *input, ICLTensor *output, Coordinates2D start, Coordinates2D end, uint32_t batch_index, float extrapolation_value = 0, Window *output_window = nullptr);
    /** Configure kernel
     *
     * @note Supported tensor rank: up to 4
     *
     * @param[in]  compile_context     The compile context to be used.
     * @param[in]  input               Source tensor. Data type supported: All. Data layouts supported: NHWC.
     * @param[out] output              Destination tensor. Data type supported: F32
     * @param[in]  start               Coordinates of where to start cropping the image.
     * @param[in]  end                 Coordinates of where to end cropping the image.
     * @param[in]  batch_index         Fourth dimension index of the 3D image to crop in @p input.
     * @param[in]  extrapolation_value Value to be used for values outside of the image. Default is 0.
     * @param[in]  output_window       Output window to be used in case cropped image is being copied into a tensor. Default is nullptr.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output, Coordinates2D start, Coordinates2D end, uint32_t batch_index, float extrapolation_value = 0,
                   Window *output_window = nullptr);

    /** Static function to check if given info will lead to a valid configuration of @ref CLStridedSliceKernel
     *
     * @note Supported tensor rank: up to 4
     *
     * @param[in] input               Source tensor info. Data type supported: All. Data layouts supported: NHWC.
     * @param[in] output              Destination tensor info. Data type supported: F32
     * @param[in] start               Coordinates of where to start cropping the image.
     * @param[in] end                 Coordinates of where to end cropping the image.
     * @param[in] batch_index         Fourth dimension index of the 3D image to crop in @p input.
     * @param[in] extrapolation_value Value to be used for values outside of the image. Default is 0.
     * @param[in] output_window       Output window to be used in case cropped image is being copied into a tensor. Default is nullptr.
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, Coordinates2D start, Coordinates2D end, uint32_t batch_index, float extrapolation_value = 0,
                           Window *output_window = nullptr);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_input;
    ICLTensor       *_output;
    Coordinates2D    _start;
    uint32_t         _batch_index;
    float            _extrapolation_value;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_CLCROPKERNEL_H */
