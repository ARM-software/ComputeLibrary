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
#include "arm_compute/runtime/CL/CLMultiHOG.h"

#include "arm_compute/core/CL/ICLHOG.h"
#include "arm_compute/core/Error.h"

using namespace arm_compute;

CLMultiHOG::CLMultiHOG(size_t num_models)
    : _num_models(num_models), _model()
{
    _model.resize(_num_models);
}

size_t CLMultiHOG::num_models() const
{
    return _num_models;
}

ICLHOG *CLMultiHOG::cl_model(size_t index)
{
    ARM_COMPUTE_ERROR_ON(index >= _num_models);
    return (&_model[index]);
}

const ICLHOG *CLMultiHOG::cl_model(size_t index) const
{
    ARM_COMPUTE_ERROR_ON(index >= _num_models);
    return (&_model[index]);
}
