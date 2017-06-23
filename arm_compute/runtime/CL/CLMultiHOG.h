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
#ifndef __ARM_COMPUTE_CLMULTIHOG_H__
#define __ARM_COMPUTE_CLMULTIHOG_H__

#include "arm_compute/core/CL/ICLMultiHOG.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLHOG.h"

#include <memory>

namespace arm_compute
{
/** Basic implementation of the CL multi HOG data-objects */
class CLMultiHOG : public ICLMultiHOG
{
public:
    /** Constructor
     *
     * @param[in] num_models Number of HOG data objects to contain
     *
     */
    CLMultiHOG(size_t num_models);

    // Inherited methods overridden:
    size_t  num_models() const override;
    ICLHOG *cl_model(size_t index) override;
    const ICLHOG *cl_model(size_t index) const override;

private:
    size_t                   _num_models;
    std::unique_ptr<CLHOG[]> _model;
};
}
#endif /*__ARM_COMPUTE_CLMULTIHOG_H__ */
