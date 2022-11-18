/*
* Copyright (c) 2020-2022 Arm Limited.
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
#ifndef SRC_CORE_HELPERS_WINDOWHELPERS_H
#define SRC_CORE_HELPERS_WINDOWHELPERS_H

#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/Steps.h"
#include "arm_compute/core/Window.h"

namespace arm_compute
{
/** Update window and padding size for each of the access patterns.
 *
 * First the window size is reduced based on all access patterns that are not
 * allowed to modify the padding of the underlying tensor. Then the padding of
 * the remaining tensors is increased to match the window.
 *
 * @param[in] win      Window that is used by the kernel.
 * @param[in] patterns Access patterns used to calculate the final window and padding.
 *
 * @return True if the window has been changed. Changes to the padding do not
 *         influence the returned value.
 */
template <typename... Ts>
bool update_window_and_padding(Window &win, Ts &&... patterns)
{
    bool window_changed = false;

    utility::for_each([&](const IAccessWindow & w)
    {
        window_changed |= w.update_window_if_needed(win);
    },
    patterns...);

    utility::for_each([&](IAccessWindow & w)
    {
        w.update_padding_if_needed(win);
    },
    patterns...);

    return window_changed;
}

/** Intersect multiple valid regions.
 *
 * @param[in] regions Valid regions.
 *
 * @return Intersection of all regions.
 */
template <typename... Ts>
ValidRegion intersect_valid_regions(const Ts &... regions)
{
    auto intersect = [](const ValidRegion & r1, const ValidRegion & r2) -> ValidRegion
    {
        ValidRegion region;

        for(size_t d = 0; d < std::min(r1.anchor.num_dimensions(), r2.anchor.num_dimensions()); ++d)
        {
            region.anchor.set(d, std::max(r1.anchor[d], r2.anchor[d]));
        }

        for(size_t d = 0; d < std::min(r1.shape.num_dimensions(), r2.shape.num_dimensions()); ++d)
        {
            region.shape.set(d, std::min(r1.shape[d], r2.shape[d]));
        }

        return region;
    };

    return utility::foldl(intersect, regions...);
}

#ifndef DOXYGEN_SKIP_THIS
/** Calculate the maximum window for a given tensor shape and border setting
 *
 * @param[in] valid_region Valid region object defining the shape of the tensor space for which the window is created.
 * @param[in] steps        (Optional) Number of elements processed for each step.
 * @param[in] skip_border  (Optional) If true exclude the border region from the window.
 * @param[in] border_size  (Optional) Border size.
 *
 * @return The maximum window the kernel can be executed on.
 */
Window calculate_max_window(const ValidRegion &valid_region, const Steps &steps = Steps(), bool skip_border = false, BorderSize border_size = BorderSize());

/** Calculate the maximum window for a given tensor shape and border setting
 *
 * @param[in] shape       Shape of the tensor space
 * @param[in] steps       (Optional) Number of elements processed for each step.
 * @param[in] skip_border (Optional) If true exclude the border region from the window.
 * @param[in] border_size (Optional) Border size.
 *
 * @return The maximum window the kernel can be executed on.
 */
Window calculate_max_window(const TensorShape &shape, const Steps &steps = Steps(), bool skip_border = false, BorderSize border_size = BorderSize());

/** Calculate the maximum window for a given tensor shape and border setting
 *
 * @param[in] info        Tensor info object defining the shape of the object for which the window is created.
 * @param[in] steps       (Optional) Number of elements processed for each step.
 * @param[in] skip_border (Optional) If true exclude the border region from the window.
 * @param[in] border_size (Optional) Border size.
 *
 * @return The maximum window the kernel can be executed on.
 */
inline Window calculate_max_window(const ITensorInfo &info, const Steps &steps = Steps(), bool skip_border = false, BorderSize border_size = BorderSize())
{
    return calculate_max_window(info.tensor_shape(), steps, skip_border, border_size);
}

/** Calculate the maximum window used by a horizontal kernel for a given tensor shape and border setting
 *
 * @param[in] valid_region Valid region object defining the shape of the tensor space for which the window is created.
 * @param[in] steps        (Optional) Number of elements processed for each step.
 * @param[in] skip_border  (Optional) If true exclude the border region from the window.
 * @param[in] border_size  (Optional) Border size. The border region will be excluded from the window.
 *
 * @return The maximum window the kernel can be executed on.
 */
Window calculate_max_window_horizontal(const ValidRegion &valid_region, const Steps &steps = Steps(), bool skip_border = false, BorderSize border_size = BorderSize());

/** Calculate the maximum window used by a horizontal kernel for a given tensor shape and border setting
 *
 * @param[in] info        Tensor info object defining the shape of the object for which the window is created.
 * @param[in] steps       (Optional) Number of elements processed for each step.
 * @param[in] skip_border (Optional) If true exclude the border region from the window.
 * @param[in] border_size (Optional) Border size.
 *
 * @return The maximum window the kernel can be executed on.
 */
inline Window calculate_max_window_horizontal(const ITensorInfo &info, const Steps &steps = Steps(), bool skip_border = false, BorderSize border_size = BorderSize())
{
    return calculate_max_window_horizontal(info.valid_region(), steps, skip_border, border_size);
}

/** Calculate the maximum window for a given tensor shape and border setting. The window will also includes the border.
 *
 * @param[in] valid_region Valid region object defining the shape of the tensor space for which the window is created.
 * @param[in] steps        (Optional) Number of elements processed for each step.
 * @param[in] border_size  (Optional) Border size. The border region will be included in the window.
 *
 * @return The maximum window the kernel can be executed on.
 */
Window calculate_max_enlarged_window(const ValidRegion &valid_region, const Steps &steps = Steps(), BorderSize border_size = BorderSize());

/** Calculate the maximum window for a given tensor shape and border setting. The window will also includes the border.
 *
 * @param[in] info        Tensor info object defining the shape of the object for which the window is created.
 * @param[in] steps       (Optional) Number of elements processed for each step.
 * @param[in] border_size (Optional) Border size. The border region will be included in the window.
 *
 * @return The maximum window the kernel can be executed on.
 */
inline Window calculate_max_enlarged_window(const ITensorInfo &info, const Steps &steps = Steps(), BorderSize border_size = BorderSize())
{
    return calculate_max_enlarged_window(info.valid_region(), steps, border_size);
}

/** Calculate the squashed or maximum window for the given tensor shape.
 *
 * If the tensor data resides continuously in the memory, the tensor can be interpreted
 * as 1D array and all the dimensions can be squashed together into the x-dimension.
 * Otherwise, generate the max window for the given tensor shape.
 *
 * @param[in] src Tensor info object defining the shape of the input tensor.
 *
 * @return The maximum window the kernel can be executed on and the preferred split dimension.
 */
std::pair<Window, size_t> calculate_squashed_or_max_window(const ITensorInfo &src);

/** Calculate the squashed or maximum window for the given tensor shapes.
 *
 * If the tensor data resides continuously in the memory, the tensor can be interpreted
 * as 1D array and all the dimensions can be squashed together into the x-dimension.
 * Otherwise, generate the max window for the given tensor shapes.
 *
 * @param[in] src0 Tensor info object defining the shape of the first input tensor.
 * @param[in] src1 Tensor info object defining the shape of the second input tensor.
 *
 * @return The squashed or maximum window the kernel can be executed on and the preferred split dimension.
 */
std::pair<Window, size_t> calculate_squashed_or_max_window(const ITensorInfo &src0, const ITensorInfo &src1);

/** Function to compute the shape of output and window for the given inputs
 *
 * @param[in] infos Input tensor informations
 *
 * @return A pair of the shape and window
 */
template <typename... Shapes>
std::pair<TensorShape, Window> compute_output_shape_and_window(const Shapes &... shapes)
{
    const TensorShape out_shape = TensorShape::broadcast_shape(shapes...);
    return std::make_pair(out_shape, calculate_max_window(out_shape));
}
#endif /* DOXYGEN_SKIP_THIS */
} // namespace arm_compute

#endif /* SRC_CORE_HELPERS_WINDOWHELPERS_H */
