/*
 * Copyright (c) 2023 Arm Limited.
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

#include "src/TensorUtils.h"
#include "ckw/Error.h"
#include "ckw/TensorInfo.h"
#include "ckw/types/TensorComponentType.h"

namespace ckw
{
TensorComponentType get_tensor_dimension(TensorDataLayout layout, TensorDataLayoutComponent component)
{
    switch(layout)
    {
        case TensorDataLayout::Nhwc:
            switch(component)
            {
                case TensorDataLayoutComponent::C:
                    return TensorComponentType::Dim0;
                case TensorDataLayoutComponent::W:
                    return TensorComponentType::Dim1;
                case TensorDataLayoutComponent::H:
                    return TensorComponentType::Dim2;
                case TensorDataLayoutComponent::N:
                    return TensorComponentType::Dim3;
                default:
                    COMPUTE_KERNEL_WRITER_ERROR_ON_MSG("Unsupported tensor component for NHWC");
                    return TensorComponentType::Unknown;
            }
        case TensorDataLayout::Ndhwc:
            switch(component)
            {
                case TensorDataLayoutComponent::C:
                    return TensorComponentType::Dim0;
                case TensorDataLayoutComponent::W:
                    return TensorComponentType::Dim1;
                case TensorDataLayoutComponent::H:
                    return TensorComponentType::Dim2;
                case TensorDataLayoutComponent::D:
                    return TensorComponentType::Dim3;
                case TensorDataLayoutComponent::N:
                    return TensorComponentType::Dim4;
                default:
                    COMPUTE_KERNEL_WRITER_ERROR_ON_MSG("Unsupported tensor component for NDHWC");
                    return TensorComponentType::Unknown;
            }
        default:
            COMPUTE_KERNEL_WRITER_ERROR_ON_MSG("Unsupported tensor data layout");
            return TensorComponentType::Unknown;
    }
}

TensorComponentType get_tensor_stride(TensorDataLayout layout, TensorDataLayoutComponent component)
{
    switch(layout)
    {
        case TensorDataLayout::Nhwc:
            switch(component)
            {
                case TensorDataLayoutComponent::C:
                    return TensorComponentType::Stride0;
                case TensorDataLayoutComponent::W:
                    return TensorComponentType::Stride1;
                case TensorDataLayoutComponent::H:
                    return TensorComponentType::Stride2;
                case TensorDataLayoutComponent::N:
                    return TensorComponentType::Stride3;
                default:
                    COMPUTE_KERNEL_WRITER_ERROR_ON_MSG("Unsupported tensor component for NHWC");
                    return TensorComponentType::Unknown;
            }
        case TensorDataLayout::Ndhwc:
            switch(component)
            {
                case TensorDataLayoutComponent::C:
                    return TensorComponentType::Stride0;
                case TensorDataLayoutComponent::W:
                    return TensorComponentType::Stride1;
                case TensorDataLayoutComponent::H:
                    return TensorComponentType::Stride2;
                case TensorDataLayoutComponent::D:
                    return TensorComponentType::Stride3;
                case TensorDataLayoutComponent::N:
                    return TensorComponentType::Stride4;
                default:
                    COMPUTE_KERNEL_WRITER_ERROR_ON_MSG("Unsupported tensor component for NDHWC");
                    return TensorComponentType::Unknown;
            }
        default:
            COMPUTE_KERNEL_WRITER_ERROR_ON_MSG("Unsupported tensor data layout");
            return TensorComponentType::Unknown;
    }
}
} // namespace ckw
