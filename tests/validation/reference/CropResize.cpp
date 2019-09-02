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
#include "CropResize.h"
#include "Utils.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
namespace
{
SimpleTensor<float> scale_image(const SimpleTensor<float> &in, const TensorShape &out_shape, InterpolationPolicy policy, float extrapolation_value)
{
    ARM_COMPUTE_ERROR_ON(in.data_layout() != DataLayout::NHWC);

    SimpleTensor<float> out{ out_shape, DataType::F32, 1, QuantizationInfo(), DataLayout::NHWC };
    // Compute the ratio between source width/height and destination width/height
    const auto wr = static_cast<float>(in.shape()[1]) / static_cast<float>(out_shape[1]);
    const auto hr = static_cast<float>(in.shape()[2]) / static_cast<float>(out_shape[2]);

    const auto width  = static_cast<int>(in.shape().y());
    const auto height = static_cast<int>(in.shape().z());

    Window win;
    win.use_tensor_dimensions(out_shape);
    execute_window_loop(win, [&](const Coordinates & out_id)
    {
        Coordinates in_id(out_id);
        int         idw = in_id.y();
        int         idh = in_id.z();

        switch(policy)
        {
            case InterpolationPolicy::NEAREST_NEIGHBOR:
            {
                //Calculate the source coords without -0.5f is equivalent to round the x_scr/y_src coords
                float x_src = std::floor(idw * wr);
                float y_src = std::floor(idh * hr);
                in_id.set(1, x_src);
                in_id.set(2, y_src);

                // If coordinates in range of tensor's width or height
                if(is_valid_pixel_index(x_src, y_src, width, height, 0))
                {
                    *reinterpret_cast<float *>(out(out_id)) = tensor_elem_at(in, in_id, BorderMode::CONSTANT, extrapolation_value);
                }
                else
                {
                    *reinterpret_cast<float *>(out(out_id)) = extrapolation_value;
                }
                break;
            }
            case InterpolationPolicy::BILINEAR:
            {
                float x_src = idw * wr;
                float y_src = idh * hr;
                in_id.set(1, std::floor(x_src));
                in_id.set(2, std::floor(y_src));
                if(is_valid_pixel_index(x_src, y_src, width, height, 0))
                {
                    const int id_w = in_id[1];
                    const int id_h = in_id[2];

                    const float dx   = x_src - id_w;
                    const float dy   = y_src - id_h;
                    const float dx_1 = 1.0f - dx;
                    const float dy_1 = 1.0f - dy;

                    in_id.set(1, id_w);
                    in_id.set(2, id_h);
                    const float tl = tensor_elem_at(in, in_id, BorderMode::CONSTANT, extrapolation_value);
                    in_id.set(1, id_w + 1);
                    in_id.set(2, id_h);
                    const float tr = tensor_elem_at(in, in_id, BorderMode::CONSTANT, extrapolation_value);
                    in_id.set(1, id_w);
                    in_id.set(2, id_h + 1);
                    const float bl = tensor_elem_at(in, in_id, BorderMode::CONSTANT, extrapolation_value);
                    in_id.set(1, id_w + 1);
                    in_id.set(2, id_h + 1);
                    const float br = tensor_elem_at(in, in_id, BorderMode::CONSTANT, extrapolation_value);

                    *reinterpret_cast<float *>(out(out_id)) = tl * (dx_1 * dy_1) + tr * (dx * dy_1) + bl * (dx_1 * dy) + br * (dx * dy);
                }
                else
                {
                    *reinterpret_cast<float *>(out(out_id)) = extrapolation_value;
                }
                break;
            }
            default:
                ARM_COMPUTE_ERROR("Unsupported interpolation mode");
        }
    });

    return out;
}

template <typename T>
SimpleTensor<float> crop_image(const SimpleTensor<T> &src, Coordinates start, Coordinates end, int32_t batch_index, float extrapolation_value)
{
    TensorShape out_shape(src.shape()[0], abs(end[0] - start[0]) + 1, abs(end[1] - start[1]) + 1);

    SimpleTensor<float> out{ out_shape, DataType::F32, 1, QuantizationInfo(), DataLayout::NHWC };

    Window win;
    win.use_tensor_dimensions(out_shape);
    execute_window_loop(win, [&](const Coordinates & id)
    {
        bool        out_of_bounds = false;
        Coordinates offset(id[0], 0, 0, batch_index);
        for(uint32_t i = 1; i < 3; ++i)
        {
            offset.set(i, end[i - 1] < start[i - 1] ? start[i - 1] - id[i] : start[i - 1] + id[i]);
            if(offset[i] < 0 || static_cast<uint32_t>(offset[i]) > src.shape()[i] - 1)
            {
                out_of_bounds = true;
                break;
            }
        }
        if(!out_of_bounds)
        {
            *reinterpret_cast<float *>(out(id)) = static_cast<float>(*reinterpret_cast<const T *>(src(offset)));
        }
        else
        {
            *reinterpret_cast<float *>(out(id)) = extrapolation_value;
        }
    });
    return out;
}

} // namespace

template <typename T>
SimpleTensor<float> crop_and_resize(const SimpleTensor<T> &src, const SimpleTensor<float> &boxes, SimpleTensor<int32_t> box_ind,
                                    Coordinates2D crop_size, InterpolationPolicy method, float extrapolation_value)
{
    ARM_COMPUTE_ERROR_ON(src.shape().num_dimensions() > 4);
    ARM_COMPUTE_ERROR_ON(src.data_layout() != DataLayout::NHWC);

    const TensorShape   out_shape(src.shape()[0], crop_size.x, crop_size.y, boxes.shape()[1]);
    SimpleTensor<float> out{ out_shape, DataType::F32, 1, QuantizationInfo(), DataLayout::NHWC };

    const TensorShape scaled_image_shape(src.shape()[0], crop_size.x, crop_size.y);

    for(uint32_t i = 0; i < boxes.shape()[1]; ++i)
    {
        Coordinates start = Coordinates(std::floor((*reinterpret_cast<const float *>(boxes(Coordinates(1, i)))) * (src.shape()[1] - 1) + 0.5f),
                                        std::floor((*reinterpret_cast<const float *>(boxes(Coordinates(0, i)))) * (src.shape()[2] - 1) + 0.5f));
        Coordinates end = Coordinates(std::floor((*reinterpret_cast<const float *>(boxes(Coordinates(3, i)))) * (src.shape()[1] - 1) + 0.5f),
                                      std::floor((*reinterpret_cast<const float *>(boxes(Coordinates(2, i)))) * (src.shape()[2] - 1) + 0.5f));
        SimpleTensor<float> cropped = crop_image(src, start, end, box_ind[i], extrapolation_value);
        SimpleTensor<float> scaled  = scale_image(cropped, scaled_image_shape, method, extrapolation_value);
        std::copy_n(reinterpret_cast<float *>(scaled.data()), scaled.num_elements(), reinterpret_cast<float *>(out(Coordinates(0, 0, 0, i))));
    }
    return out;
}

template SimpleTensor<float> crop_and_resize(const SimpleTensor<float> &src, const SimpleTensor<float> &boxes, SimpleTensor<int32_t> box_ind,
                                             Coordinates2D crop_size, InterpolationPolicy method, float extrapolation_value);
template SimpleTensor<float> crop_and_resize(const SimpleTensor<uint16_t> &src, const SimpleTensor<float> &boxes, SimpleTensor<int32_t> box_ind,
                                             Coordinates2D crop_size, InterpolationPolicy method, float extrapolation_value);
template SimpleTensor<float> crop_and_resize(const SimpleTensor<uint32_t> &src, const SimpleTensor<float> &boxes, SimpleTensor<int32_t> box_ind,
                                             Coordinates2D crop_size, InterpolationPolicy method, float extrapolation_value);
template SimpleTensor<float> crop_and_resize(const SimpleTensor<int16_t> &src, const SimpleTensor<float> &boxes, SimpleTensor<int32_t> box_ind,
                                             Coordinates2D crop_size, InterpolationPolicy method, float extrapolation_value);
template SimpleTensor<float> crop_and_resize(const SimpleTensor<int32_t> &src, const SimpleTensor<float> &boxes, SimpleTensor<int32_t> box_ind,
                                             Coordinates2D crop_size, InterpolationPolicy method, float extrapolation_value);
template SimpleTensor<float> crop_and_resize(const SimpleTensor<half> &src, const SimpleTensor<float> &boxes, SimpleTensor<int32_t> box_ind,
                                             Coordinates2D crop_size, InterpolationPolicy method, float extrapolation_value);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
