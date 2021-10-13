/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "src/cpu/kernels/CpuPool2dKernel.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "src/core/AccessWindowStatic.h"
#include "src/core/CPP/Validate.h"
#include "src/core/NEON/NEAsymm.h"
#include "src/core/NEON/NEFixedPoint.h"
#include "src/core/NEON/NEMath.h"
#include "src/core/common/Registrars.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/cpu/kernels/pool2d/neon/list.h"
#include "support/ToolchainSupport.h"

#include "src/core/NEON/wrapper/wrapper.h"
#include <arm_neon.h>

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace
{
using namespace misc::shape_calculator;

struct PoolingSelectorData
{
    DataType   dt;
    DataLayout dl;
    int        pool_stride_x;
    Size2D     pool_size;
};

using PoolingSelectorPtr = std::add_pointer<bool(const PoolingSelectorData &data)>::type;
using PoolingKernelPtr   = std::add_pointer<void(const ITensor *, ITensor *, ITensor *, PoolingLayerInfo &, const Window &, const Window &)>::type;
struct PoolingKernel
{
    const char              *name;
    const PoolingSelectorPtr is_selected;
    PoolingKernelPtr         ukernel;
};

static const PoolingKernel available_kernels[] =
{
    {
        "neon_qu8_nhwc_poolMxN",
        [](const PoolingSelectorData & data) { return ((data.dl == DataLayout::NHWC) && (data.dt == DataType::QASYMM8)); },
        REGISTER_QASYMM8_NEON(arm_compute::cpu::poolingMxN_qasymm8_neon_nhwc)
    },
    {
        "neon_qs8_nhwc_poolMxN",
        [](const PoolingSelectorData & data) { return ((data.dl == DataLayout::NHWC) && (data.dt == DataType::QASYMM8_SIGNED)); },
        REGISTER_QASYMM8_SIGNED_NEON(arm_compute::cpu::poolingMxN_qasymm8_signed_neon_nhwc)
    },
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    {
        "neon_f16_nhwc_poolMxN",
        [](const PoolingSelectorData & data) { return ((data.dl == DataLayout::NHWC) && (data.dt == DataType::F16)); },
        REGISTER_FP16_NEON(arm_compute::cpu::poolingMxN_fp16_neon_nhwc)
    },
#endif /* defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) */
    {
        "neon_fp32_nhwc_poolMxN",
        [](const PoolingSelectorData & data) { return ((data.dl == DataLayout::NHWC) && (data.dt == DataType::F32)); },
        REGISTER_FP32_NEON(arm_compute::cpu::poolingMxN_fp32_neon_nhwc)
    },
#if defined(ENABLE_NCHW_KERNELS)
    {
        "neon_qu8_nchw_pool2",
        [](const PoolingSelectorData & data) { return ((data.dl == DataLayout::NCHW) && (data.dt == DataType::QASYMM8) && (data.pool_size.x() == data.pool_size.y()) && (data.pool_size.x() == 2) && (data.pool_stride_x < 3)); },
        REGISTER_QASYMM8_NEON(arm_compute::cpu::pooling2_quantized_neon_nchw<uint8_t>)
    },
    {
        "neon_qu8_nchw_pool3",
        [](const PoolingSelectorData & data) { return ((data.dl == DataLayout::NCHW) && (data.dt == DataType::QASYMM8) && (data.pool_size.x() == data.pool_size.y()) && (data.pool_size.x() == 3) && (data.pool_stride_x < 3)); },
        REGISTER_QASYMM8_NEON(arm_compute::cpu::pooling3_quantized_neon_nchw<uint8_t>)
    },
    {
        "neon_qu8_nchw_poolMxN",
        [](const PoolingSelectorData & data) { return ((data.dl == DataLayout::NCHW) && (data.dt == DataType::QASYMM8)); },
        REGISTER_QASYMM8_NEON(arm_compute::cpu::poolingMxN_quantized_neon_nchw<uint8_t>)
    },
    {
        "neon_qs8_nchw_pool2",
        [](const PoolingSelectorData & data) { return ((data.dl == DataLayout::NCHW) && (data.dt == DataType::QASYMM8_SIGNED) && (data.pool_size.x() == data.pool_size.y()) && (data.pool_size.x() == 2) && (data.pool_stride_x < 3)); },
        REGISTER_QASYMM8_SIGNED_NEON(arm_compute::cpu::pooling2_quantized_neon_nchw<int8_t>)
    },
    {
        "neon_qs8_nchw_pool3",
        [](const PoolingSelectorData & data) { return ((data.dl == DataLayout::NCHW) && (data.dt == DataType::QASYMM8_SIGNED) && (data.pool_size.x() == data.pool_size.y()) && (data.pool_size.x() == 3) && (data.pool_stride_x < 3)); },
        REGISTER_QASYMM8_SIGNED_NEON(arm_compute::cpu::pooling3_quantized_neon_nchw<int8_t>)
    },
    {
        "neon_qs8_nchw_poolMxN",
        [](const PoolingSelectorData & data) { return ((data.dl == DataLayout::NCHW) && (data.dt == DataType::QASYMM8_SIGNED)); },
        REGISTER_QASYMM8_SIGNED_NEON(arm_compute::cpu::poolingMxN_quantized_neon_nchw<int8_t>)
    },
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    {
        "neon_fp16_nchw_pool2",
        [](const PoolingSelectorData & data) { return ((data.dl == DataLayout::NCHW) && (data.dt == DataType::F16) && (data.pool_size.x() == data.pool_size.y()) && (data.pool_size.x() == 2)); },
        REGISTER_FP16_NEON(arm_compute::cpu::pooling2_fp16_neon_nchw)
    },
    {
        "neon_fp16_nchw_pool3",
        [](const PoolingSelectorData & data) { return ((data.dl == DataLayout::NCHW) && (data.dt == DataType::F16) && (data.pool_size.x() == data.pool_size.y()) && (data.pool_size.x() == 3)); },
        REGISTER_FP16_NEON(arm_compute::cpu::pooling3_fp16_neon_nchw)
    },
    {
        "neon_fp16_nchw_poolMxN",
        [](const PoolingSelectorData & data) { return ((data.dl == DataLayout::NCHW) && (data.dt == DataType::F16)); },
        REGISTER_FP16_NEON(arm_compute::cpu::poolingMxN_fp16_neon_nchw)
    },
#endif /* defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) */
    {
        "neon_fp32_nchw_pool2",
        [](const PoolingSelectorData & data) { return ((data.dl == DataLayout::NCHW) && (data.dt == DataType::F32) && (data.pool_size.x() == data.pool_size.y()) && (data.pool_size.x() == 2)); },
        REGISTER_FP32_NEON(arm_compute::cpu::pooling2_fp32_neon_nchw)
    },
    {
        "neon_fp32_nchw_pool3",
        [](const PoolingSelectorData & data) { return ((data.dl == DataLayout::NCHW) && (data.dt == DataType::F32) && (data.pool_size.x() == data.pool_size.y()) && (data.pool_size.x() == 3)); },
        REGISTER_FP32_NEON(arm_compute::cpu::pooling3_fp32_neon_nchw)
    },
    {
        "neon_fp32_nchw_pool7",
        [](const PoolingSelectorData & data) { return ((data.dl == DataLayout::NCHW) && (data.dt == DataType::F32) && (data.pool_size.x() == data.pool_size.y()) && (data.pool_size.x() == 7)); },
        REGISTER_FP32_NEON(arm_compute::cpu::pooling7_fp32_neon_nchw)
    },
    {
        "neon_fp32_nchw_poolMxN",
        [](const PoolingSelectorData & data) { return ((data.dl == DataLayout::NCHW) && (data.dt == DataType::F32)); },
        REGISTER_FP32_NEON(arm_compute::cpu::poolingMxN_fp32_neon_nchw)
    },
#endif /* defined(ENABLE_NCHW_KERNELS) */
};

/** Micro-kernel selector
 *
 * @param[in] data Selection data passed to help pick the appropriate micro-kernel
 *
 * @return A matching micro-kernel else nullptr
 */
const PoolingKernel *get_implementation(DataType dt, DataLayout dl, int pool_stride_x, Size2D pool_size)
{
    for(const auto &uk : available_kernels)
    {
        if(uk.is_selected({ dt, dl, pool_stride_x, pool_size }))
        {
            return &uk;
        }
    }
    return nullptr;
}

Status validate_arguments(const ITensorInfo *src, const ITensorInfo *dst, const PoolingLayerInfo &pool_info,
                          const ITensorInfo *indices, Size2D pool_size)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_RETURN_ERROR_ON(pool_size.x() == 0);
    ARM_COMPUTE_RETURN_ERROR_ON(pool_size.y() == 0);

    int                 pool_stride_x   = 0;
    int                 pool_stride_y   = 0;
    int                 output_width    = 0;
    int                 output_height   = 0;
    PoolingType         pool_type       = pool_info.pool_type;
    const PadStrideInfo pad_stride_info = pool_info.pad_stride_info;
    const auto          data_layout     = pool_info.data_layout == DataLayout::UNKNOWN ? src->data_layout() : pool_info.data_layout;
    const int           idx_width       = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int           idx_height      = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);

    std::tie(output_width, output_height) = scaled_dimensions_signed(src->tensor_shape()[idx_width], src->tensor_shape()[idx_height],
                                                                     pool_size.x(), pool_size.y(), pool_info.pad_stride_info);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((output_width < 1 || output_height < 1), "Calculated output dimension size is invalid");

    TensorInfo out_info(TensorInfo(compute_pool_shape(*src, pool_info), 1, dst->data_type()));
    std::tie(pool_stride_x, pool_stride_y) = pad_stride_info.stride();

    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(src);
    if(indices)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::F32, DataType::F16);
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(indices, 1, DataType::U32);
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(pool_type != PoolingType::MAX, "Pooling indices only supported for MAX pooling method");
    }
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON(pool_type == PoolingType::L2 && is_data_type_quantized(src->data_type()));
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(is_data_type_quantized(src->data_type()) && !pool_info.exclude_padding && (pool_info.pool_type == PoolingType::AVG) && pool_info.pad_stride_info.has_padding()
                                    && (src->data_layout() == DataLayout::NHWC),
                                    "exclude_padding equal false is not supported for AVG Pooling with padding on quantized types");

    if(dst->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(src, dst);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(dst, &out_info);
        if(indices)
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MSG((pool_size != Size2D(2, 2)), "Pooling indices only supported for pool size 2x2");
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(indices, &out_info);
        }
    }

    const auto *uk = get_implementation(src->data_type(), src->data_layout(), pool_stride_x, pool_size);
    ARM_COMPUTE_RETURN_ERROR_ON(uk == nullptr || uk->ukernel == nullptr);

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *src, ITensorInfo *dst, ITensorInfo *indices, const PoolingLayerInfo &pool_info,
                                                        unsigned int &num_elems_processed_per_iteration,
                                                        int pool_size_x, int pool_size_y)
{
    // dst auto inizialitation if not yet initialized
    auto_init_if_empty(*dst, src->clone()->set_tensor_shape(compute_pool_shape(*src, pool_info)));
    if(indices)
    {
        // Indices auto inizialitation if not yet initialized
        auto_init_if_empty(*indices, (src->clone()->set_tensor_shape(compute_pool_shape(*src,
                                                                                        pool_info)))
                           .set_data_type(DataType::U32) /* we store the offset to the element */);
    }
    const auto data_layout = pool_info.data_layout == DataLayout::UNKNOWN ? src->data_layout() : pool_info.data_layout;

    int                 pool_stride_x   = 0;
    int                 pool_stride_y   = 0;
    const int           idx_width       = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int           idx_height      = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const PadStrideInfo pad_stride_info = pool_info.pad_stride_info;

    std::tie(pool_stride_x, pool_stride_y) = pad_stride_info.stride();
    const bool         is_square = pool_size_x == pool_size_y;
    const unsigned int pooled_w  = dst->dimension(idx_width);
    const unsigned int pooled_h  = dst->dimension(idx_height);

    //If it's not squared and optimized will be executed the MxN
    num_elems_processed_per_iteration = 1;

    if(is_square)
    {
        switch(src->data_type())
        {
            case DataType::QASYMM8:
            case DataType::QASYMM8_SIGNED:
                switch(pool_size_x)
                {
                    case 2:
                        num_elems_processed_per_iteration = (pool_stride_x == 2) ? 8 : 15;
                        break;
                    case 3:
                        num_elems_processed_per_iteration = (pool_stride_x == 2) ? 7 : 14;
                        break;
                    default:
                        break;
                }
                break;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            case DataType::F16:
                num_elems_processed_per_iteration = 1;
                break;
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
            case DataType::F32:
                num_elems_processed_per_iteration = 1;
                break;
            default:
                ARM_COMPUTE_ERROR("Element size not supported");
                break;
        }
    }

    bool   window_changed = false;
    Window win{};
    // Upper limit for the number of right/bottom border elements that are accessed
    TensorShape dst_shape{ src->tensor_shape() };
    dst_shape.set(0, pooled_w);
    dst_shape.set(1, pooled_h);
    TensorInfo dst_info(src->clone()->set_tensor_shape(dst_shape));
    win = calculate_max_window(dst_info, Steps(num_elems_processed_per_iteration));

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

void CpuPool2dKernel::configure(ITensorInfo *src, ITensorInfo *dst, const PoolingLayerInfo &pool_info, ITensorInfo *indices)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);
    const PadStrideInfo pad_stride_info   = pool_info.pad_stride_info;
    const bool          is_global_pooling = pool_info.is_global_pooling;

    // Get data layout
    const auto data_layout = pool_info.data_layout == DataLayout::UNKNOWN ? src->data_layout() : pool_info.data_layout;
    const int  idx_width   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int  idx_height  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);

    // Update pool size in case of global pooling
    const Size2D pool_size(
        is_global_pooling ? src->dimension(idx_width) : pool_info.pool_size.width,
        is_global_pooling ? src->dimension(idx_height) : pool_info.pool_size.height);

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, dst, pool_info, indices, pool_size));

    const auto *uk = get_implementation(src->data_type(), src->data_layout(), pad_stride_info.stride().first, pool_size);
    ARM_COMPUTE_ERROR_ON(uk == nullptr);

    // Set instance variables
    _pool_info     = pool_info;
    _data_layout   = src->data_layout();
    _pool_size     = pool_size;
    _pool_stride_x = pad_stride_info.stride().first;
    _run_method    = uk->ukernel;
    _name          = std::string("CpuPool2dKernel").append("/").append(uk->name);

    if(_data_layout == DataLayout::NHWC)
    {
        // Configure kernel window
        Window win = calculate_max_window(*dst, Steps());
        ICpuKernel::configure(win);
    }
    else
    {
        // Configure kernel window
        auto win_config = validate_and_configure_window(src, dst, indices, pool_info, _num_elems_processed_per_iteration,
                                                        pool_size.x(), pool_size.y());
        ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
        ICpuKernel::configure(win_config.second);
    }
}

Status CpuPool2dKernel::validate(const ITensorInfo *src, const ITensorInfo *dst, const PoolingLayerInfo &pool_info, const ITensorInfo *indices)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src);

    unsigned int num_elems_processed_per_iteration = 0;

    const bool is_global_pooling = pool_info.is_global_pooling;

    // Get data layout
    const auto data_layout = pool_info.data_layout == DataLayout::UNKNOWN ? src->data_layout() : pool_info.data_layout;
    const int  idx_width   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int  idx_height  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);

    unsigned int pool_size_x = is_global_pooling ? src->dimension(idx_width) : pool_info.pool_size.width;
    unsigned int pool_size_y = is_global_pooling ? src->dimension(idx_height) : pool_info.pool_size.height;

    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, dst, pool_info, indices, Size2D(pool_size_x, pool_size_y)));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(src->clone().get(), dst->clone().get(),
                                                              (indices) ? indices->clone().get() : nullptr, pool_info, num_elems_processed_per_iteration,
                                                              pool_size_x, pool_size_y)
                                .first);

    return Status{};
}

void CpuPool2dKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICpuKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_run_method == nullptr);

    const ITensor *src     = tensors.get_const_tensor(TensorType::ACL_SRC_0);
    ITensor       *dst     = tensors.get_tensor(TensorType::ACL_DST_0);
    ITensor       *indices = tensors.get_tensor(TensorType::ACL_DST_1);

    const unsigned int pool_stride_x = _pool_info.pad_stride_info.stride().first;
    const unsigned int pool_stride_y = _pool_info.pad_stride_info.stride().second;
    const unsigned int pool_size     = _pool_info.pool_size.width;

    Window window_src(window);
    if(_data_layout == DataLayout::NCHW)
    {
        // Set step for src in x and y direction for the src
        unsigned int window_x_inc = 0;
        switch(src->info()->data_type())
        {
            case DataType::QASYMM8:
            case DataType::QASYMM8_SIGNED:
            {
                window_x_inc = pool_stride_x;
                if((pool_size == 2 || pool_size == 3) && pool_stride_x < 3)
                {
                    window_x_inc = (pool_stride_x == 2) ? _num_elems_processed_per_iteration * 2 : _num_elems_processed_per_iteration;
                }
                break;
            }

            case DataType::F16:
            case DataType::F32:
            {
                window_x_inc = pool_stride_x;
                break;
            }
            default:
            {
                ARM_COMPUTE_ERROR("Not supported");
            }
        }
        window_src.set(Window::DimX, Window::Dimension(window.x().start() * pool_stride_x, window.x().end() * pool_stride_x, window_x_inc));
        window_src.set(Window::DimY, Window::Dimension(window.y().start() * pool_stride_y, window.y().end() * pool_stride_y, pool_stride_y));
    }
    else
    {
        window_src.set(Window::DimX, Window::Dimension(0, 1, 1));
        window_src.set(Window::DimY, Window::Dimension(0, src->info()->dimension(1), pool_stride_x));
        window_src.set(Window::DimZ, Window::Dimension(0, src->info()->dimension(2), pool_stride_y));
    }
    _run_method(src, dst, indices, _pool_info, window_src, window);
}

const char *CpuPool2dKernel::name() const
{
    return _name.c_str();
}
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
