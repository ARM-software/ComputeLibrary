/*
 * Copyright (c) 2020 Arm Limited.
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
#include "arm_compute/core/NEON/kernels/NEMaxUnpoolingLayerKernel.h"

#include "arm_compute/core/CPP/Validate.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include "support/ToolchainSupport.h"

namespace arm_compute
{
using namespace misc::shape_calculator;

namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const PoolingLayerInfo &pool_info, const ITensorInfo *indices)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output, indices);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(indices, 1, DataType::U32);

    int                 pool_stride_x   = 0;
    int                 pool_stride_y   = 0;
    PoolingType         pool_type       = pool_info.pool_type;
    const PadStrideInfo pad_stride_info = pool_info.pad_stride_info;
    std::tie(pool_stride_x, pool_stride_y) = pad_stride_info.stride();
    const int    pool_size_x = pool_info.pool_size.width;
    const int    pool_size_y = pool_info.pool_size.height;
    const Size2D pool_size(pool_size_x, pool_size_y);

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(pool_type != PoolingType::MAX, "Pooling indices only supported for MAX pooling method");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((pool_size != Size2D(2, 2)), "Pooling indices only supported for pool size 2x2");
    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(input, output);
    }

    return Status{};
}
} // namespace

NEMaxUnpoolingLayerKernel::NEMaxUnpoolingLayerKernel()
    : _func(nullptr), _input(nullptr), _output(nullptr), _indices(nullptr)
{
}

void NEMaxUnpoolingLayerKernel::configure(const ITensor *input, const ITensor *indices, ITensor *output, const PoolingLayerInfo &pool_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), pool_info, indices->info()));

    _input   = input;
    _output  = output;
    _indices = indices;

    switch(input->info()->data_type())
    {
        case DataType::F32:
            _func = &NEMaxUnpoolingLayerKernel::unpooling2<float>;
            break;
        case DataType::QASYMM8:
            _func = &NEMaxUnpoolingLayerKernel::unpooling2<uint8_t>;
            break;
        case DataType::QASYMM8_SIGNED:
            _func = &NEMaxUnpoolingLayerKernel::unpooling2<int8_t>;
            break;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F16:
            _func = &NEMaxUnpoolingLayerKernel::unpooling2<float16_t>;
            break;
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
        default:
            break;
    }
    const TensorShape output_shape = compute_unpool_shape(*input->info(), pool_info);
    auto_init_if_empty(*output->info(), input->info()->clone()->set_tensor_shape(output_shape));

    auto window = calculate_max_window(*input->info(), Steps());
    INEKernel::configure(window);
}
template <typename T>
void NEMaxUnpoolingLayerKernel::unpooling2(const Window &window)
{
    Iterator  input(_input, window);
    Iterator  indices(_indices, window);
    auto      out_ptr      = reinterpret_cast<T *>(_output->buffer());
    const int out_stride_w = static_cast<int>(_output->info()->strides_in_bytes()[3]);
    execute_window_loop(window, [&](const Coordinates & id)
    {
        auto vindices                                         = reinterpret_cast<uint32_t *>(indices.ptr());
        auto vinput                                           = reinterpret_cast<T *>(input.ptr());
        out_ptr[id[3] * out_stride_w / sizeof(T) + *vindices] = *vinput;
    },
    input, indices);
}

Status NEMaxUnpoolingLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *indices, const ITensorInfo *output, const PoolingLayerInfo &pool_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, indices, output);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, pool_info, indices));
    return Status{};
}

void NEMaxUnpoolingLayerKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);
    // Run function
    (this->*_func)(window);
}
} // namespace arm_compute
