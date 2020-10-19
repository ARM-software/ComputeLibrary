/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_IACCESS_WINDOW_TRANSPOSE_H
#define ARM_COMPUTE_IACCESS_WINDOW_TRANSPOSE_H

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
class Window;
class ITensorInfo;

/** Implementation of a XY-transpose access pattern. */
class AccessWindowTranspose : public AccessWindowRectangle
{
public:
    using AccessWindowRectangle::AccessWindowRectangle;
    bool update_window_if_needed(Window &window) const override;
    bool update_padding_if_needed(const Window &window) override;
    using AccessWindowRectangle::compute_valid_region;
    ValidRegion compute_valid_region(const Window &window, ValidRegion input_valid_region, bool border_undefined, BorderSize border_size) const override;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_IACCESS_WINDOW_TRANSPOSE_H*/
