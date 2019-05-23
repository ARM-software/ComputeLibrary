/*
 * Copyright (c) 2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEON_CROP_KERNEL_H__
#define __ARM_COMPUTE_NEON_CROP_KERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Types.h"

#include <cstdint>
#include <map>

namespace arm_compute
{
// Forward declarations
class ITensor;

/** Interface for the kernel to perform tensor cropping */
class NECropKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NECropKernel";
    }
    /** Default constructor */
    NECropKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NECropKernel(const NECropKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NECropKernel &operator=(const NECropKernel &) = delete;
    /** Allow instances of this class to be moved */
    NECropKernel(NECropKernel &&) = default;
    /** Allow instances of this class to be moved */
    NECropKernel &operator=(NECropKernel &&) = default;
    /** Default destructor */
    ~NECropKernel() = default;
    /** Configure kernel
     *
     * @note Supported tensor rank: up to 4
     * @note Padding not supported.
     *
     * @param[in]  input               Source tensor. Data type supported: U16/S16/U32/S32/F16/F32. Data layouts supported: NHWC.
     * @param[in]  crop_boxes          Tensor containing all possible boxes used to crop the image, each represented by 4 normalized values.
     *                                 Data type supported: F32
     * @param[in]  box_ind             One dimensional tensor mapping the @p crop_box_ind to the index of the 3D image in @p input.
     *                                 Data type supported: F32
     * @param[out] output              Destination tensor. Data type supported: F32
     * @param[in]  crop_box_ind        Index of the crop box to be used from @p crop_boxes. Default is 0.
     * @param[in]  extrapolation_value Value to be used for values outside of the image. Default is 0.
     */
    void configure(const ITensor *input, const ITensor *crop_boxes, const ITensor *box_ind, ITensor *output, uint32_t crop_box_ind = 0, float extrapolation_value = 0);

    /** Static function to check if given info will lead to a valid configuration of @ref CLStridedSliceKernel
     *
     * @note Supported tensor rank: up to 4
     * @note Padding not supported.
     *
     * @param[in] input               Source tensor info. Data type supported: U16/S16/U32/S32/F16/F32. Data layouts supported: NHWC.
     * @param[in] crop_boxes          Tensor info for tensor containing all possible boxes used to crop the image. Data type supported: F32
     * @param[in] box_ind             Tensor info for the one dimensional tensor mapping the @p crop_box_ind to the index of the 3D image
     *                                in @p input. Data type supported: F32
     * @param[in] output              Destination tensor. Data type supported: F32
     * @param[in] crop_box_ind        Index of the crop box to be used from @p crop_boxes. Default is 0.
     * @param[in] extrapolation_value Value to be used for values outside of the image. Default is 0.
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *crop_boxes, const ITensorInfo *box_ind, const ITensorInfo *output, uint32_t crop_box_ind = 0, float extrapolation_value = 0);

    /** Configure output tensor's shape as this can only be determined at runtime. */
    void configure_output_shape();

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

    /** Function to use for in bounds crop for the particular tensor types passed to configure() */
    using InBoundsCropFunction = void(const ITensor *, const ITensor *, float *, Coordinates, int32_t, int32_t, int32_t);

private:
    const ITensor *_input;
    const ITensor *_crop_boxes;
    const ITensor *_box_ind;
    ITensor       *_output;

    Coordinates _start;
    Coordinates _end;
    uint32_t    _crop_box_ind;
    float       _extrapolation_value;
    /** The number of rows out of bounds at the start and end of output. */
    std::array<uint32_t, 2> _rows_out_of_bounds;
    /** The number of columns out of bounds at the start and end of output. */
    std::array<uint32_t, 2> _cols_out_of_bounds;

    std::pair<NECropKernel::InBoundsCropFunction *, NECropKernel::InBoundsCropFunction *> _in_bounds_crop_functions;
    NECropKernel::InBoundsCropFunction *_in_bounds_crop_function;

    using CropFunction = void(const ITensor *, const ITensor *, Coordinates, float, const std::array<uint32_t, 2> &, const std::array<uint32_t, 2> &,
                              NECropKernel::InBoundsCropFunction *);

    NECropKernel::CropFunction *_crop_function;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NEON_CROP_KERNEL_H__ */
