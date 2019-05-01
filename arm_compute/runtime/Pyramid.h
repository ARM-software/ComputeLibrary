/*
 * Copyright (c) 2016-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_PYRAMID_H__
#define __ARM_COMPUTE_PYRAMID_H__

#include "arm_compute/core/IPyramid.h"
#include "arm_compute/core/PyramidInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/Tensor.h"

#include <cstddef>
#include <memory>

namespace arm_compute
{
class Tensor;

/** Basic implementation of the pyramid interface */
class Pyramid : public IPyramid
{
public:
    /** Initialize pyramid data-object using the given Pyramid's metadata
     *
     * @param[in] info Pyramid's metadata
     */
    void init(const PyramidInfo &info);

    /** Initialize pyramid data-object using the given Pyramid's metadata
     *
     * @note Uses conservative padding strategy which fits all kernels.
     *
     * @param[in] info Pyramid's metadata
     */
    void init_auto_padding(const PyramidInfo &info);

    /** Allocate the planes in the pyramid */
    void allocate();

    // Inherited method overridden
    const PyramidInfo *info() const override;
    Tensor *get_pyramid_level(size_t index) const override;

private:
    /** Initialize pyramid data-object using the given Pyramid's metadata
     *
     * @param[in] info         Pyramid's metadata
     * @param[in] auto_padding Specifies whether the image in the pyramid use auto padding
     */
    void internal_init(const PyramidInfo &info, bool auto_padding);

    PyramidInfo                 _info{};
    mutable std::vector<Tensor> _pyramid{};
};
}
#endif /*__ARM_COMPUTE_PYRAMID_H__ */
