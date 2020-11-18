/*
 * Copyright (c) 2020-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CLTUNING_PARAMS_H
#define ARM_COMPUTE_CLTUNING_PARAMS_H

#include "arm_compute/core/CL/OpenCL.h"

namespace arm_compute
{
/**< OpenCL tuner parameters */
class CLTuningParams
{
public:
    CLTuningParams(const CLTuningParams &) = default;

    CLTuningParams(unsigned int lws_x = 0, unsigned int lws_y = 0, unsigned int lws_z = 0)
        : _lws(lws_x, lws_y, lws_z)
    {
    }
    CLTuningParams(cl::NDRange lws)
        : _lws(lws)
    {
    }
    void set_lws(cl::NDRange &lws)
    {
        _lws = lws;
    }

    cl::NDRange get_lws()
    {
        return _lws;
    }

private:
    cl::NDRange _lws;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_CLTUNING_PARAMS_H */
