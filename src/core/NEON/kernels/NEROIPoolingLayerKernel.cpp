/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "src/core/NEON/kernels/NEROIPoolingLayerKernel.h"

#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include "src/core/CPP/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/ToolchainSupport.h"

#include <cfloat>

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo         *input,
                          const ITensorInfo         *rois,
                          const ITensorInfo         *output,
                          const ROIPoolingLayerInfo &pool_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output, rois);

    //Validate arguments
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(rois, DataType::U16);
    ARM_COMPUTE_RETURN_ERROR_ON(rois->dimension(0) != 5);
    ARM_COMPUTE_RETURN_ERROR_ON(rois->num_dimensions() > 2);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(input, DataType::F32, DataType::QASYMM8);
    ARM_COMPUTE_RETURN_ERROR_ON((pool_info.pooled_width() == 0) || (pool_info.pooled_height() == 0));

    if (output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON((output->dimension(0) != pool_info.pooled_width()) ||
                                    (output->dimension(1) != pool_info.pooled_height()));
        ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(2) != output->dimension(2));
        ARM_COMPUTE_RETURN_ERROR_ON(rois->dimension(1) != output->dimension(3));
    }

    return Status{};
}

/** Evaluate number needing to be stored in output tensor as quantized format.
 *
 * @param[in]  input          Source tensor. Data types supported: QASYMM8
 * @param[out] output         Destination tensor. Where output value will be stored, same datatype as input
 * @param[in]  region_start_x Beginning region of x coordinate of pooling region
 * @param[in]  region_start_y Beginning region of y coordinate of pooling region
 * @param[in]  region_end_x   End of pooling region, x coordinate
 * @param[in]  region_end_y   End of pooling region, y coordinate
 * @param[in]  fm             Channel index of coordinate in output Tensor to store value
 * @param[in]  px             Width index of coodinate in output Tensor to store value
 * @param[in]  py             Height index of coordinate in output Tensor to store value
 * @param[in]  roi_batch      Index of image to perform Pooling on in input Tensor
 * @param[in]  roi_indx       Index of image of coordinate in output Tensor to store value
 */
template <typename T>
void template_eval(const ITensor *input,
                   const ITensor *output,
                   int            region_start_x,
                   int            region_start_y,
                   int            region_end_x,
                   int            region_end_y,
                   int            fm,
                   int            px,
                   int            py,
                   int            roi_batch,
                   int            roi_indx)
{
    if ((region_end_x <= region_start_x) || (region_end_y <= region_start_y))
    {
        *reinterpret_cast<T *>(output->ptr_to_element(Coordinates(px, py, fm, roi_indx))) = 0;
    }
    else
    {
        T curr_max = std::numeric_limits<T>::lowest(); // Min value of typename T
        for (int j = region_start_y; j < region_end_y; ++j)
        {
            for (int i = region_start_x; i < region_end_x; ++i)
            {
                const auto val = *reinterpret_cast<const T *>(input->ptr_to_element(Coordinates(i, j, fm, roi_batch)));
                curr_max       = std::max(val, curr_max);
            }
        }

        // if quantized datatype, requantize then store in output tensor
        if (is_data_type_quantized(input->info()->data_type()))
        {
            // covert qasymm to new output quantization scale and offset
            UniformQuantizationInfo uqinfo = compute_requantization_scale_offset(
                input->info()->quantization_info().uniform(), output->info()->quantization_info().uniform());
            *reinterpret_cast<T *>(output->ptr_to_element(Coordinates(px, py, fm, roi_indx))) =
                quantize_qasymm8(curr_max, uqinfo);
        }
        else
        {
            *reinterpret_cast<T *>(output->ptr_to_element(Coordinates(px, py, fm, roi_indx))) = curr_max;
        }
    }
}
} // namespace

NEROIPoolingLayerKernel::NEROIPoolingLayerKernel()
    : _input(nullptr), _rois(nullptr), _output(nullptr), _pool_info(0, 0, 0.f)
{
}

Status NEROIPoolingLayerKernel::validate(const ITensorInfo         *input,
                                         const ITensorInfo         *rois,
                                         const ITensorInfo         *output,
                                         const ROIPoolingLayerInfo &pool_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, rois, output, pool_info));
    return Status{};
}

void NEROIPoolingLayerKernel::configure(const ITensor             *input,
                                        const ITensor             *rois,
                                        const ITensor             *output,
                                        const ROIPoolingLayerInfo &pool_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output, rois);

    //Validate arguments
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), rois->info(), output->info(), pool_info));

    // Output auto initialization if not yet initialized
    TensorShape output_shape(pool_info.pooled_width(), pool_info.pooled_height(), input->info()->dimension(2),
                             rois->info()->dimension(1));

    auto_init_if_empty(*output->info(), output_shape, 1, input->info()->data_type(),
                       output->info()->quantization_info());

    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_ERROR_ON((output->info()->dimension(0) != pool_info.pooled_width()) ||
                         (output->info()->dimension(1) != pool_info.pooled_height()));

    // Set instance variables
    _input     = input;
    _rois      = rois;
    _output    = output;
    _pool_info = pool_info;

    // Configure kernel window
    Window window;
    window.set(Window::DimX, Window::Dimension(0, rois->info()->dimension(1)));
    window.set(Window::DimY, Window::Dimension(0, 1));

    INEKernel::configure(window);
}

void NEROIPoolingLayerKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    const size_t values_per_roi = _rois->info()->dimension(0);

    const int   roi_list_start = window.x().start();
    const int   roi_list_end   = window.x().end();
    const int   width          = _input->info()->dimension(Window::DimX);
    const int   height         = _input->info()->dimension(Window::DimY);
    const int   fms            = _input->info()->dimension(Window::DimZ);
    const int   pooled_w       = _pool_info.pooled_width();
    const int   pooled_h       = _pool_info.pooled_height();
    const float spatial_scale  = _pool_info.spatial_scale();

    const auto *rois_ptr  = reinterpret_cast<const uint16_t *>(_rois->buffer());
    const auto  data_type = _input->info()->data_type();

    for (int roi_indx = roi_list_start; roi_indx < roi_list_end; ++roi_indx)
    {
        const unsigned int roi_batch = rois_ptr[values_per_roi * roi_indx];
        const auto         x1        = rois_ptr[values_per_roi * roi_indx + 1];
        const auto         y1        = rois_ptr[values_per_roi * roi_indx + 2];
        const auto         x2        = rois_ptr[values_per_roi * roi_indx + 3];
        const auto         y2        = rois_ptr[values_per_roi * roi_indx + 4];

        // Scale ROI
        const int roi_anchor_x = support::cpp11::round(x1 * spatial_scale);
        const int roi_anchor_y = support::cpp11::round(y1 * spatial_scale);
        const int roi_width    = std::max(support::cpp11::round((x2 - x1) * spatial_scale), 1.f);
        const int roi_height   = std::max(support::cpp11::round((y2 - y1) * spatial_scale), 1.f);

        // Iterate through all feature maps
        for (int fm = 0; fm < fms; ++fm)
        {
            // Iterate through all output pixels
            for (int py = 0; py < pooled_h; ++py)
            {
                for (int px = 0; px < pooled_w; ++px)
                {
                    auto region_start_x = static_cast<int>(std::floor((static_cast<float>(px) / pooled_w) * roi_width));
                    auto region_end_x =
                        static_cast<int>(std::floor((static_cast<float>(px + 1) / pooled_w) * roi_width));
                    auto region_start_y =
                        static_cast<int>(std::floor((static_cast<float>(py) / pooled_h) * roi_height));
                    auto region_end_y =
                        static_cast<int>(std::floor((static_cast<float>(py + 1) / pooled_h) * roi_height));

                    region_start_x = std::min(std::max(region_start_x + roi_anchor_x, 0), width);
                    region_end_x   = std::min(std::max(region_end_x + roi_anchor_x, 0), width);
                    region_start_y = std::min(std::max(region_start_y + roi_anchor_y, 0), height);
                    region_end_y   = std::min(std::max(region_end_y + roi_anchor_y, 0), height);

                    switch (data_type)
                    {
                        case DataType::F32:
                            template_eval<float>(_input, _output, region_start_x, region_start_y, region_end_x,
                                                 region_end_y, fm, px, py, roi_batch, roi_indx);
                            break;
                        case DataType::QASYMM8:
                            template_eval<qasymm8_t>(_input, _output, region_start_x, region_start_y, region_end_x,
                                                     region_end_y, fm, px, py, roi_batch, roi_indx);
                            break;
                        default:
                            ARM_COMPUTE_ERROR("DataType not Supported");
                            break;
                    }
                }
            }
        }
    }
}
} // namespace arm_compute
