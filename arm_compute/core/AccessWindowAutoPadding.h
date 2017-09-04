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
#ifndef __ARM_COMPUTE_ACCESS_WINDOW_AUTO_PADDING_H__
#define __ARM_COMPUTE_ACCESS_WINDOW_AUTO_PADDING_H__

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
class Window;
class ITensorInfo;

/** Dummy access window.
 *
 * This implementation always uses the auto padding of the tensor info and
 * never updates the window. The valid region is always set to cover the entire
 * tensor.
 *
 * @note This access window is only used during the migration to the new
 *       padding system. It will be removed once all kernels have been ported.
 *
 * */
class AccessWindowAutoPadding : public IAccessWindow
{
public:
    /** Default constructor.
     *
     * @param[in,out] info Tensor info of the accessed kernel.
     */
    AccessWindowAutoPadding(ITensorInfo *info);
    AccessWindowAutoPadding(const AccessWindowAutoPadding &) = delete;
    AccessWindowAutoPadding &operator=(const AccessWindowAutoPadding &) = delete;
    AccessWindowAutoPadding(AccessWindowAutoPadding &&)                 = default;
    AccessWindowAutoPadding &operator=(AccessWindowAutoPadding &&) = default;
    ~AccessWindowAutoPadding()                                     = default;

    /** Set the valid region to match the entire tensor. */
    void set_valid_region();

    /** Return a valid region that spans across the entire tensor. */
    ValidRegion compute_valid_region() const;

    // Inherited methods overridden:
    bool update_window_if_needed(Window &window) const override;
    bool update_padding_if_needed(const Window &window) const override;
    ValidRegion compute_valid_region(const Window &window, ValidRegion input_valid_region, bool border_undefined, BorderSize border_size) const override;

private:
    ITensorInfo *_info;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_ACCESS_WINDOW_AUTO_PADDING_H__*/
