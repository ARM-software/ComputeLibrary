/*
 * Copyright (c) 2016-2022 Arm Limited.
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
#ifndef ARM_COMPUTE_CPU_SCALEKERNEL_H
#define ARM_COMPUTE_CPU_SCALEKERNEL_H

#include "arm_compute/core/KernelDescriptors.h"
#include "src/core/common/Macros.h"
#include "src/cpu/ICpuKernel.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
/** Arm(R) Neon(TM) kernel to perform scaling on a tensor */
class CpuScaleKernel : public ICpuKernel<CpuScaleKernel>
{
private:
    /** Scale function to use for the particular function to use */
    using ScaleFunctionPtr = void (CpuScaleKernel::*)(const ITensor *, ITensor *, const ITensor *, const ITensor *, const ITensor *, const Window &window);
    using ScaleKernelPtr   = std::add_pointer<void(const ITensor *, ITensor *, const ITensor *, const ITensor *, const ITensor *,
                                                   InterpolationPolicy, BorderMode, PixelValue, float, bool, const Window &)>::type;

public:
    CpuScaleKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuScaleKernel);
    /** Initialise the kernel's inputs, output and interpolation policy
     *
     * @note dx, dy and offsets have the same dimensions (width and height) of the output tensor
     * @note Using @p policy Area only supports data layout NCHW and input data type U8.
     * @note Using S8 data type only supports NHWC, @p border_mode Replicate, and @p policy Bilinear
     *
     * @param[in]  src     Source tensor info. Data types supported: QASYMM8/QASYMM8_SIGNED/U8/S8/S16/F16/F32.
     * @param[in]  dx      Distance x tensor info. Pixel's distance between the X real coordinate and the smallest X following integer. Data type supported: F32
     * @param[in]  dy      Distance y tensor info. Pixel's distance between the Y real coordinate and the smallest Y following integer. Data type supported: F32
     * @param[in]  offsets Offset tensor info. Offset to access the pixel with NEAREST interpolation or the top-left pixel with BILINEAR interpolation in the input tensor. Data type supported: S32.
     * @param[out] dst     Destination tensor info. Data types supported: Same as @p input. All but the lowest two dimensions must be the same size as in the input tensor, i.e. scaling is only performed within the XY-plane.
     * @param[in]  info    @ref ScaleKernelInfo to use for configuration
     */
    void configure(const ITensorInfo *src, const ITensorInfo *dx, const ITensorInfo *dy, const ITensorInfo *offsets, ITensorInfo *dst,
                   const ScaleKernelInfo &info);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to CpuScaleKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *dx, const ITensorInfo *dy, const ITensorInfo *offsets, ITensorInfo *dst,
                           const ScaleKernelInfo &info);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;
    const char *name() const override;

    struct ScaleKernel
    {
        const char                                 *name;
        const ScaleKernelDataTypeISASelectorDataPtr is_selected;
        ScaleKernelPtr                              ukernel;
    };

    static const std::vector<ScaleKernel> &get_available_kernels();

private:
#ifdef ENABLE_NCHW_KERNELS
    /** function to perform scale using area interpolation on the given window
     *
     *  @note Used only in case down-sampling.
     */
    void scale_area_nchw_u8(const ITensor *src, ITensor *dst, const ITensor *dx, const ITensor *dy, const ITensor *offsets, const Window &window);

    /** function to perform scale using bilinear interpolation on the given window */
    template <typename T>
    void scale_bilinear_nchw(const ITensor *src, ITensor *dst, const ITensor *dx, const ITensor *dy, const ITensor *offsets, const Window &window);
    /** function to perform scale using bilinear interpolation on the given window */
    template <typename T>
    void scale_bilinear_qasymm(const ITensor *src, ITensor *dst, const ITensor *dx, const ITensor *dy, const ITensor *offsets, const Window &window);

    /** function to perform scale using nearest neighbour on the given window */
    template <typename T>
    void scale_nearest_nchw(const ITensor *src, ITensor *dst, const ITensor *dx, const ITensor *dy, const ITensor *offsets, const Window &window);
#endif // ENABLE_NCHW_KERNELS

    ScaleFunctionPtr    _func{ nullptr };
    InterpolationPolicy _policy{};
    BorderMode          _border_mode{};
    PixelValue          _constant_border_value{};
    float               _sampling_offset{ 0 };
    bool                _align_corners{ false };
    DataLayout          _data_layout{ DataLayout::UNKNOWN };
    ScaleKernelPtr      _run_method{ nullptr };
    std::string         _name{};
};
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_SCALEKERNEL_H */
