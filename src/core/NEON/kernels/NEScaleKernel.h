/*
 * Copyright (c) 2016-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_NESCALEKERNEL_H
#define ARM_COMPUTE_NESCALEKERNEL_H

#include "arm_compute/core/KernelDescriptors.h"
#include "src/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** Neon kernel to perform scaling on a tensor */
class NEScaleKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEScaleKernel";
    }
    /** Default constructor */
    NEScaleKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEScaleKernel(const NEScaleKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEScaleKernel &operator=(const NEScaleKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEScaleKernel(NEScaleKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEScaleKernel &operator=(NEScaleKernel &&) = default;
    /** Default destructor */
    ~NEScaleKernel() = default;

    /** Initialise the kernel's inputs, output and interpolation policy
     *
     * @note dx, dy and offsets have the same dimensions (width and height) of the output tensor
     * @note Using @p policy Area only supports data layout NCHW and input data type U8.
     *
     * @param[in]  input   Source tensor. Data types supported: QASYMM8/QASYMM8_SIGNED/U8/S16/F16/F32.
     * @param[in]  dx      Pixel's distance between the X real coordinate and the smallest X following integer. Data type supported: F32
     * @param[in]  dy      Pixel's distance between the Y real coordinate and the smallest Y following integer. Data type supported: F32
     * @param[in]  offsets Offset to access the pixel with NEAREST interpolation or the top-left pixel with BILINEAR interpolation in the input tensor. Data type supported: S32.
     * @param[out] output  Destination tensor. Data types supported: Same as @p input. All but the lowest two dimensions must be the same size as in the input tensor, i.e. scaling is only performed within the XY-plane.
     * @param[in]  info    @ref ScaleKernelInfo to use for configuration
     */
    void configure(const ITensor *input, const ITensor *dx, const ITensor *dy, const ITensor *offsets, ITensor *output,
                   const ScaleKernelInfo &info);
    /** Static function to check if given info will lead to a valid configuration of @ref NEScaleKernel
     *
     * @note dx, dy and offsets have the same dimensions (width and height) of the output tensor
     * @note Using @p policy Area only supports data layout NCHW and input data type U8.
     *
     * @param[in] input   Source tensor. Data types supported: QASYMM8/QASYMM8_SIGNED/U8/S16/F16/F32.
     * @param[in] dx      Pixel's distance between the X real coordinate and the smallest X following integer. Data type supported: F32
     * @param[in] dy      Pixel's distance between the Y real coordinate and the smallest Y following integer. Data type supported: F32
     * @param[in] offsets Offset to access the pixel with NEAREST interpolation or the top-left pixel with BILINEAR interpolation in the input tensor. Data type supported: S32.
     * @param[in] output  Destination tensor. Data types supported: Same as @p input. All but the lowest two dimensions must be the same size as in the input tensor, i.e. scaling is only performed within the XY-plane.
     * @param[in] info    @ref ScaleKernelInfo to use for validation
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *dx, const ITensorInfo *dy, const ITensorInfo *offsets, ITensorInfo *output,
                           const ScaleKernelInfo &info);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    /** function to perform scale using area interpolation on the given window
     *
     *  @note Used only in case down-sampling.
     */
    void scale_area_nchw_u8(const Window &window);

    /** function to perform scale using bilinear interpolation on the given window */
    template <typename T>
    void scale_bilinear_nchw(const Window &window);
    /** function to perform scale using bilinear interpolation on the given window */
    template <typename T>
    void scale_bilinear_qasymm(const Window &window);

    /** function to perform scale using nearest neighbour on the given window */
    template <typename T>
    void scale_nearest_nchw(const Window &window);

    /** Scale function to use for the particular function to use */
    using ScaleFunctionPtr = void (NEScaleKernel::*)(const Window &window);

    ScaleFunctionPtr    _func;
    const ITensor      *_offsets;
    const ITensor      *_dx;
    const ITensor      *_dy;
    const ITensor      *_input;
    ITensor            *_output;
    InterpolationPolicy _policy;
    BorderMode          _border_mode;
    PixelValue          _constant_border_value;
    float               _sampling_offset;
    bool                _align_corners;
    DataLayout          _data_layout;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_NESCALEKERNEL_H */
