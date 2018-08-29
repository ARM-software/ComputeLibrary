/*
 * Copyright (c) 2018 ARM Limited.
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
#include "arm_compute/runtime/CL/functions/CLConcatenateLayer.h"

#include "arm_compute/runtime/CL/functions/CLDepthConcatenateLayer.h"
#include "arm_compute/runtime/CL/functions/CLWidthConcatenateLayer.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "support/ToolchainSupport.h"

namespace arm_compute
{
CLConcatenateLayer::CLConcatenateLayer()
    : _concat_function(nullptr)
{
}

void CLConcatenateLayer::configure(const std::vector<ICLTensor *> &inputs_vector, ICLTensor *output, DataLayoutDimension axis)
{
    ARM_COMPUTE_ERROR_ON(output == nullptr);

    switch(get_data_layout_dimension_index(output->info()->data_layout(), axis))
    {
        case 0:
        {
            auto func = support::cpp14::make_unique<CLWidthConcatenateLayer>();
            func->configure(inputs_vector, output);
            _concat_function = std::move(func);
            break;
        }
        case 2:
        {
            auto func = support::cpp14::make_unique<CLDepthConcatenateLayer>();
            func->configure(inputs_vector, output);
            _concat_function = std::move(func);
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Concatenation is supported across width and depth only!");
    }
}

Status CLConcatenateLayer::validate(const std::vector<ITensorInfo *> &inputs_vector, const ITensorInfo *output, DataLayoutDimension axis)
{
    ARM_COMPUTE_RETURN_ERROR_ON(output == nullptr);

    switch(get_data_layout_dimension_index(output->data_layout(), axis))
    {
        case 0:
            ARM_COMPUTE_RETURN_ON_ERROR(CLWidthConcatenateLayer::validate(inputs_vector, output));
            break;
        case 2:
            ARM_COMPUTE_RETURN_ON_ERROR(CLDepthConcatenateLayer::validate(inputs_vector, output));
            break;
        default:
            ARM_COMPUTE_RETURN_ERROR_MSG("Concatenation is supported across width and depth only!");
    }
    return Status{};
}

void CLConcatenateLayer::run()
{
    ARM_COMPUTE_ERROR_ON(_concat_function == nullptr);
    _concat_function->run();
}
} // namespace arm_compute
