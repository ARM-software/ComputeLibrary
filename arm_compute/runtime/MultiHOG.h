/*
 * Copyright (c) 2016-2019 Arm Limited.
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
#ifndef ARM_COMPUTE_MULTIHOG_H
#define ARM_COMPUTE_MULTIHOG_H

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IMultiHOG.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/HOG.h"

#include <memory>

namespace arm_compute
{
/** CPU implementation of multi HOG data-object */
class MultiHOG : public IMultiHOG
{
public:
    /** Constructor
     *
     * @param[in] num_models Number of HOG data objects to contain
     *
     */
    MultiHOG(size_t num_models);

    // Inherited methods overridden:
    size_t num_models() const override;
    IHOG *model(size_t index) override;
    const IHOG *model(size_t index) const override;

private:
    size_t           _num_models;
    std::vector<HOG> _model;
};
}

#endif /* ARM_COMPUTE_MULTIHOG_H */
