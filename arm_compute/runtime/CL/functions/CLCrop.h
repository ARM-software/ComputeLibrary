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
#ifndef ARM_COMPUTE_CLCROP_H
#define ARM_COMPUTE_CLCROP_H

#include "arm_compute/core/Types.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/runtime/IFunction.h"
#include <memory>

namespace arm_compute
{
class CLCompileContext;
class ICLTensor;
class ITensorInfo;

/** Basic function to run @ref opencl::kernels::ClCropKernel */
class CLCrop : public IFunction
{
public:
    /** Constructor */
    CLCrop();
    /** Destructor */
    ~CLCrop();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLCrop(const CLCrop &) = delete;
    /** Default move constructor */
    CLCrop(CLCrop &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLCrop &operator=(const CLCrop &) = delete;
    /** Default move assignment operator */
    CLCrop &operator=(CLCrop &&);
    /** Configure function
     *
     * @note Supported tensor rank: up to 4
     *
     * Valid data layouts:
     * - NHWC
     *
     * Valid data type configurations:
     * |src            |dst            |
     * |:--------------|:--------------|
     * |All            |F32            |
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
    /** Configure function
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
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_CLCROP_H */
