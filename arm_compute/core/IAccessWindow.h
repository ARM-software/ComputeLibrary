/*
 * Copyright (c) 2017-2019 Arm Limited.
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
#ifndef ARM_COMPUTE_IACCESS_WINDOW_H
#define ARM_COMPUTE_IACCESS_WINDOW_H

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"

#include <array>

namespace arm_compute
{
class Window;
class ITensorInfo;

/** Decrease @p required in steps of @p step until it's less than @p available.
 *
 * @param[in] required  Number of required bytes.
 * @param[in] available Number of available bytes.
 * @param[in] step      Step size used to decrease required bytes.
 *
 * @return Largest value smaller than @p available that is a multiple of @p step
 *
 **/
inline int adjust_down(int required, int available, int step)
{
    ARM_COMPUTE_ERROR_ON(step <= 0);

    return required - step * ((required - available + step - 1) / step);
}

/** Increase @p required in steps of @p step until it's greater than @p available.
 *
 * @param[in] required  Number of required bytes.
 * @param[in] available Number of available bytes.
 * @param[in] step      Step size used to increase required bytes.
 *
 * @return Largest value smaller than @p available that is a multiple of @p step
 *
 **/
inline int adjust_up(int required, int available, int step)
{
    ARM_COMPUTE_ERROR_ON(step <= 0);

    return required + step * ((available - required + step - 1) / step);
}

/** Interface describing methods to update access window and padding based on kernel parameters. */
class IAccessWindow
{
public:
    /** Default virtual destructor */
    virtual ~IAccessWindow() = default;
    /** Shrink the window if padding is not large enough.
     *
     * @param[in] window Window used by the kernel.
     *
     * @return True if the window has been changed.
     *
     */
    virtual bool update_window_if_needed(Window &window) const = 0;
    /** Increase the padding to be large enough for the window.
     *
     * @param[in] window Window used by the kernel.
     *
     * @return True if the padding has been changed.
     */
    virtual bool update_padding_if_needed(const Window &window) = 0;
    /** Compute the valid region based on access pattern and valid region of the inputs.
     *
     * @note This method assumes that there is no border.
     *
     * @param[in] window             Execution window of the kernel.
     * @param[in] input_valid_region Combined valid region of all inputs.
     * @param[in] border_undefined   Undefined borders are excluded from the valid region.
     * @param[in] border_size        Size of the border around the XY-plane of the tensor.
     *
     * @return a valid region.
     *
     */
    virtual ValidRegion compute_valid_region(const Window &window, ValidRegion input_valid_region, bool border_undefined, BorderSize border_size) const = 0;
};

/** Implementation of a rectangular access pattern. */
class AccessWindowRectangle : public IAccessWindow
{
public:
    /** Constructor for a rectangular access pattern.
     *
     * @note Width and height have to be non-negative.
     *
     * @param[in,out] info   Tensor info of the accessed kernel.
     * @param[in]     x      Offset of the access in X direction.
     * @param[in]     y      Offset of the access in Y direction.
     * @param[in]     width  Number of elements that are accessed in X direction.
     * @param[in]     height Number of elements that are accessed in Y direction.
     */
    AccessWindowRectangle(ITensorInfo *info, int x, int y, int width, int height)
        : AccessWindowRectangle(info, x, y, width, height, 1.f, 1.f)
    {
    }

    /** Constructor for a rectangular access pattern.
     *
     * @note Width, height and scale have to be non-negative.
     *
     * @param[in,out] info    Tensor info of the accessed kernel.
     * @param[in]     x       Offset of the access in X direction.
     * @param[in]     y       Offset of the access in Y direction.
     * @param[in]     width   Number of elements that are accessed in X direction.
     * @param[in]     height  Number of elements that are accessed in Y direction.
     * @param[in]     scale_x Ratio along the X direction between the window used by the execute_window_loop and the rectangular access pattern defined
     * @param[in]     scale_y Ratio along the Y direction between the window used by the execute_window_loop and the rectangular access pattern defined
     */
    AccessWindowRectangle(ITensorInfo *info, int x, int y, int width, int height, float scale_x, float scale_y)
        : _info(info), _x(x), _y(y), _width(width), _height(height), _scale_x(scale_x), _scale_y(scale_y)
    {
        ARM_COMPUTE_ERROR_ON(width < 0);
        ARM_COMPUTE_ERROR_ON(height < 0);
        ARM_COMPUTE_ERROR_ON(scale_x < 0);
        ARM_COMPUTE_ERROR_ON(scale_y < 0);
    }

    /** Prevent instances of this class from being copied (As this class contains pointers) */
    AccessWindowRectangle(const AccessWindowRectangle &) = delete;
    /** Allow instances of this class to be move constructed */
    AccessWindowRectangle(AccessWindowRectangle &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    AccessWindowRectangle &operator=(const AccessWindowRectangle &) = delete;
    /** Allow instances of this class to be moved */
    AccessWindowRectangle &operator=(AccessWindowRectangle &&) = default;
    /** Default destructor */
    ~AccessWindowRectangle() = default;

    /** Set the valid region based on access pattern, valid region of the inputs and border mode.
     *
     * @param[in] window             Execution window of the kernel.
     * @param[in] input_valid_region Combined valid region of all inputs.
     * @param[in] border_undefined   (Optional) Undefined borders are excluded from the valid region.
     * @param[in] border_size        (Optional) Size of the border around the XY-plane of the tensor.
     */
    void set_valid_region(const Window &window, const ValidRegion &input_valid_region, bool border_undefined = false, const BorderSize &border_size = BorderSize(0));

    /** Compute the valid region based on access pattern, valid region of the inputs and border mode.
     *
     * @note This method assumes that there is no border.
     *
     * @param[in] window             Execution window of the kernel.
     * @param[in] input_valid_region Combined valid region of all inputs.
     *
     * @return a valid region.
     *
     */
    ValidRegion compute_valid_region(const Window &window, const ValidRegion &input_valid_region) const;

    // Inherited methods overridden:

    /** Compute the valid region based on access pattern and valid region of the inputs.
     *
     * @note This method assumes that all elements written by the kernel are valid.
     *
     * @param[in] window             Execution window of the kernel.
     * @param[in] input_valid_region Combined valid region of all inputs.
     * @param[in] border_undefined   Undefined borders are excluded from the valid region.
     * @param[in] border_size        Size of the border around the XY-plane of the tensor.
     *
     * @return a valid region.
     *
     */
    ValidRegion compute_valid_region(const Window &window, ValidRegion input_valid_region, bool border_undefined, BorderSize border_size) const override;

    bool update_window_if_needed(Window &window) const override;
    bool update_padding_if_needed(const Window &window) override;

protected:
    PaddingSize get_needed_padding(const Window &window) const;

protected:
    ITensorInfo *_info;
    int          _x;
    int          _y;
    int          _width;
    int          _height;
    float        _scale_x;
    float        _scale_y;
};

/** Implementation of a column access pattern. */
class AccessWindowVertical : public AccessWindowRectangle
{
public:
    /** Constructor for a column access pattern.
     *
     * @note Height has to be non-negative.
     *
     * @param[in,out] info    Tensor info of the accessed kernel.
     * @param[in]     y       Offset of the access in Y direction.
     * @param[in]     height  Number of elements that are accessed in Y direction.
     * @param[in]     scale_y Ratio along the Y direction between the window used by the execute_window_loop and the rectangular access pattern defined
     */
    AccessWindowVertical(ITensorInfo *info, int y, int height, float scale_y = 1.f)
        : AccessWindowRectangle(info, 0, y, 1, height, 1.f, scale_y)
    {
        ARM_COMPUTE_ERROR_ON(height < 0);
        ARM_COMPUTE_ERROR_ON(scale_y < 0);
    }
};

/** Implementation of a row access pattern. */
class AccessWindowHorizontal : public AccessWindowRectangle
{
public:
    /** Constructor for a row access pattern.
     *
     * @note Width has to be non-negative.
     *
     * @param[in,out] info    Tensor info of the accessed kernel.
     * @param[in]     x       Offset of the access in X direction.
     * @param[in]     width   Number of elements that are accessed in X direction.
     * @param[in]     scale_x Ratio along the X direction between the window used by the execute_window_loop and the rectangular access pattern defined
     */
    AccessWindowHorizontal(ITensorInfo *info, int x, int width, float scale_x = 1.f)
        : AccessWindowRectangle(info, x, 0, width, 1, scale_x, 1.f)
    {
        ARM_COMPUTE_ERROR_ON(width < 0);
        ARM_COMPUTE_ERROR_ON(scale_x < 0);
    }
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_IACCESS_WINDOW_H*/
