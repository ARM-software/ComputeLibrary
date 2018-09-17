/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_HOG_H__
#define __ARM_COMPUTE_HOG_H__

#include "arm_compute/core/HOGInfo.h"
#include "arm_compute/core/IHOG.h"
#include "arm_compute/core/Types.h"

#include <memory>

namespace arm_compute
{
/** CPU implementation of HOG data-object */
class HOG : public IHOG
{
public:
    /** Default constructor */
    HOG();
    /** Allocate the HOG descriptor using the given HOG's metadata
     *
     * @param[in] input HOG's metadata used to allocate the HOG descriptor
     */
    void init(const HOGInfo &input);

    // Inherited method overridden:
    const HOGInfo *info() const override;
    float         *descriptor() const override;

private:
    HOGInfo                  _info;
    std::unique_ptr<float[]> _descriptor;
};
}
#endif /* __ARM_COMPUTE_HOG_H__ */
