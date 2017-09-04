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
#ifndef __ARM_COMPUTE_IACCESS_WINDOW_STATIC_H__
#define __ARM_COMPUTE_IACCESS_WINDOW_STATIC_H__

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"

#include <array>

namespace arm_compute
{
class Window;
class ITensorInfo;

/** Implementation of a static rectangular access pattern.
 *
 * In this implementation the access offsets and sizes are not relative to the
 * current element. Instead they are considered to be absolute coordinates
 * within the accessed tensor's shape.
 *
 * */
class AccessWindowStatic : public IAccessWindow
{
public:
    /** Constructor for a static access pattern.
     *
     * @param[in,out] info    Tensor info of the accessed kernel.
     * @param[in]     start_x Start of the access in X direction.
     * @param[in]     start_y Start of the access in Y direction.
     * @param[in]     end_x   End of the access in X direction.
     * @param[in]     end_y   End of the access in Y direction.
     */
    AccessWindowStatic(ITensorInfo *info, int start_x, int start_y, int end_x, int end_y);

    AccessWindowStatic(const AccessWindowStatic &) = delete;
    AccessWindowStatic &operator=(const AccessWindowStatic &) = delete;
    AccessWindowStatic(AccessWindowStatic &&)                 = default;
    AccessWindowStatic &operator=(AccessWindowStatic &&) = default;
    ~AccessWindowStatic()                                = default;

    /** Set the valid region based on the static access pattern and valid
     *  region of the inputs.
     *
     * @param[in] window             Execution window of the kernel.
     * @param[in] input_valid_region Combined valid region of all inputs.
     */
    void set_valid_region(const Window &window, const ValidRegion &input_valid_region);

    /** Compute the valid region based on the static access pattern and valid region of the inputs.
     *
     * @param[in] window             Execution window of the kernel.
     * @param[in] input_valid_region Combined valid region of all inputs.
     */
    ValidRegion compute_valid_region(const Window &window, ValidRegion input_valid_region) const;

    // Inherited methods overriden:
    bool update_window_if_needed(Window &window) const override;
    bool update_padding_if_needed(const Window &window) const override;
    ValidRegion compute_valid_region(const Window &window, ValidRegion input_valid_region, bool border_undefined, BorderSize border_size) const override;

    ITensorInfo *_info;
    int          _start_x;
    int          _start_y;
    int          _end_x;
    int          _end_y;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_IACCESS_WINDOW_STATIC_H__*/
