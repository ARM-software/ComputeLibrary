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
#include "src/core/cpu/kernels/CpuPoolingKernel.h"

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
#include "src/core/cpu/kernels/pooling/neon/list.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
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
        "poolingMxN_qasymm8_neon_nhwc",
        [](const PoolingSelectorData & data) { return ((data.dl == DataLayout::NHWC) && (data.dt == DataType::QASYMM8)); },
        REGISTER_QASYMM8_NEON(arm_compute::cpu::poolingMxN_qasymm8_neon_nhwc)
    },
    {
        "poolingMxN_qasymm8_signed_neon_nhwc",
        [](const PoolingSelectorData & data) { return ((data.dl == DataLayout::NHWC) && (data.dt == DataType::QASYMM8_SIGNED)); },
        REGISTER_QASYMM8_SIGNED_NEON(arm_compute::cpu::poolingMxN_qasymm8_signed_neon_nhwc)
    },
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    {
        "poolingMxN_fp16_neon_nhwc",
        [](const PoolingSelectorData & data) { return ((data.dl == DataLayout::NHWC) && (data.dt == DataType::F16)); },
        REGISTER_FP16_NEON(arm_compute::cpu::poolingMxN_fp16_neon_nhwc)
    },
#endif /* defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) */
    {
        "poolingMxN_fp32_neon_nhwc",
        [](const PoolingSelectorData & data) { return ((data.dl == DataLayout::NHWC) && (data.dt == DataType::F32)); },
        REGISTER_FP32_NEON(arm_compute::cpu::poolingMxN_fp32_neon_nhwc)
    },
#if defined(ENABLE_NCHW_KERNELS)
    {
        "pooling2_qasymm8_neon_nchw",
        [](const PoolingSelectorData & data) { return ((data.dl == DataLayout::NCHW) && (data.dt == DataType::QASYMM8) && (data.pool_size.x() == data.pool_size.y()) && (data.pool_size.x() == 2) && (data.pool_stride_x < 3)); },
        REGISTER_QASYMM8_NEON(arm_compute::cpu::pooling2_quantized_neon_nchw<uint8_t>)
    },
    {
        "pooling3_qasymm8_neon_nchw",
        [](const PoolingSelectorData & data) { return ((data.dl == DataLayout::NCHW) && (data.dt == DataType::QASYMM8) && (data.pool_size.x() == data.pool_size.y()) && (data.pool_size.x() == 3) && (data.pool_stride_x < 3)); },
        REGISTER_QASYMM8_NEON(arm_compute::cpu::pooling3_quantized_neon_nchw<uint8_t>)
    },
    {
        "poolingMxN_qasymm8_neon_nchw",
        [](const PoolingSelectorData & data) { return ((data.dl == DataLayout::NCHW) && (data.dt == DataType::QASYMM8)); },
        REGISTER_QASYMM8_NEON(arm_compute::cpu::poolingMxN_quantized_neon_nchw<uint8_t>)
    },
    {
        "pooling2_qasymm8_signed_neon_nchw",
        [](const PoolingSelectorData & data) { return ((data.dl == DataLayout::NCHW) && (data.dt == DataType::QASYMM8_SIGNED) && (data.pool_size.x() == data.pool_size.y()) && (data.pool_size.x() == 2) && (data.pool_stride_x < 3)); },
        REGISTER_QASYMM8_SIGNED_NEON(arm_compute::cpu::pooling2_quantized_neon_nchw<int8_t>)
    },
    {
        "pooling3_qasymm8_signed_neon_nchw",
        [](const PoolingSelectorData & data) { return ((data.dl == DataLayout::NCHW) && (data.dt == DataType::QASYMM8_SIGNED) && (data.pool_size.x() == data.pool_size.y()) && (data.pool_size.x() == 3) && (data.pool_stride_x < 3)); },
        REGISTER_QASYMM8_SIGNED_NEON(arm_compute::cpu::pooling3_quantized_neon_nchw<int8_t>)
    },
    {
        "poolingMxN_qasymm8_signed_neon_nchw",
        [](const PoolingSelectorData & data) { return ((data.dl == DataLayout::NCHW) && (data.dt == DataType::QASYMM8_SIGNED)); },
        REGISTER_QASYMM8_SIGNED_NEON(arm_compute::cpu::poolingMxN_quantized_neon_nchw<int8_t>)
    },
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    {
        "pooling2_fp16_neon_nchw",
        [](const PoolingSelectorData & data) { return ((data.dl == DataLayout::NCHW) && (data.dt == DataType::F16) && (data.pool_size.x() == data.pool_size.y()) && (data.pool_size.x() == 2)); },
        REGISTER_FP16_NEON(arm_compute::cpu::pooling2_fp16_neon_nchw)
    },
    {
        "pooling3_fp16_neon_nchw",
        [](const PoolingSelectorData & data) { return ((data.dl == DataLayout::NCHW) && (data.dt == DataType::F16) && (data.pool_size.x() == data.pool_size.y()) && (data.pool_size.x() == 3)); },
        REGISTER_FP16_NEON(arm_compute::cpu::pooling3_fp16_neon_nchw)
    },
    {
        "poolingMxN_fp16_neon_nchw",
        [](const PoolingSelectorData & data) { return ((data.dl == DataLayout::NCHW) && (data.dt == DataType::F16)); },
        REGISTER_FP16_NEON(arm_compute::cpu::poolingMxN_fp16_neon_nchw)
    },
#endif /* defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) */
    {
        "pooling2_fp32_neon_nchw",
        [](const PoolingSelectorData & data) { return ((data.dl == DataLayout::NCHW) && (data.dt == DataType::F32) && (data.pool_size.x() == data.pool_size.y()) && (data.pool_size.x() == 2)); },
        REGISTER_FP32_NEON(arm_compute::cpu::pooling2_fp32_neon_nchw)
    },
    {
        "pooling3_fp32_neon_nchw",
        [](const PoolingSelectorData & data) { return ((data.dl == DataLayout::NCHW) && (data.dt == DataType::F32) && (data.pool_size.x() == data.pool_size.y()) && (data.pool_size.x() == 3)); },
        REGISTER_FP32_NEON(arm_compute::cpu::pooling3_fp32_neon_nchw)
    },
    {
        "pooling7_fp32_neon_nchw",
        [](const PoolingSelectorData & data) { return ((data.dl == DataLayout::NCHW) && (data.dt == DataType::F32) && (data.pool_size.x() == data.pool_size.y()) && (data.pool_size.x() == 7)); },
        REGISTER_FP32_NEON(arm_compute::cpu::pooling7_fp32_neon_nchw)
    },
    {
        "poolingMxN_fp32_neon_nchw",
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
                          unsigned int &pooled_w, unsigned int pooled_h, const ITensorInfo *indices, Size2D pool_size)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);

    int                 pool_stride_x   = 0;
    int                 pool_stride_y   = 0;
    PoolingType         pool_type       = pool_info.pool_type;
    const PadStrideInfo pad_stride_info = pool_info.pad_stride_info;
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
        ARM_COMPUTE_RETURN_ERROR_ON((dst->dimension(get_data_layout_dimension_index(src->data_layout(), DataLayoutDimension::WIDTH)) != pooled_w)
                                    || (dst->dimension(get_data_layout_dimension_index(src->data_layout(), DataLayoutDimension::HEIGHT)) != pooled_h));

        if(indices)
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MSG((pool_size != Size2D(2, 2)), "Pooling indices only supported for pool size 2x2");
            ARM_COMPUTE_RETURN_ERROR_ON((indices->dimension(get_data_layout_dimension_index(indices->data_layout(), DataLayoutDimension::WIDTH)) != pooled_w)
                                        || (indices->dimension(get_data_layout_dimension_index(indices->data_layout(), DataLayoutDimension::HEIGHT)) != pooled_h));
        }
    }

    const auto *uk = get_implementation(src->data_type(), src->data_layout(), pool_stride_x, pool_size);
    ARM_COMPUTE_RETURN_ERROR_ON(uk == nullptr || uk->ukernel == nullptr);

    return Status{};
}

Status validate_arguments_pool_info(const unsigned int pool_size_x, const unsigned int pool_size_y)
{
    ARM_COMPUTE_RETURN_ERROR_ON(pool_size_x == 0);
    ARM_COMPUTE_RETURN_ERROR_ON(pool_size_y == 0);

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *src, ITensorInfo *dst, ITensorInfo *indices, const PoolingLayerInfo &pool_info,
                                                        unsigned int &num_elems_processed_per_iteration,
                                                        BorderSize   &border_size,
                                                        unsigned int pooled_w, unsigned int pooled_h, int pool_size_x, int pool_size_y)
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
    const auto          data_layout                  = pool_info.data_layout == DataLayout::UNKNOWN ? src->data_layout() : pool_info.data_layout;
    unsigned int        num_elems_read_per_iteration = 0;
    unsigned int        num_elems_horizontal_window  = 0;
    int                 pool_stride_x                = 0;
    int                 pool_stride_y                = 0;
    const int           idx_width                    = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int           idx_height                   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const int           src_width                    = src->dimension(idx_width);
    const int           src_height                   = src->dimension(idx_height);
    const PadStrideInfo pad_stride_info              = pool_info.pad_stride_info;
    std::tie(pool_stride_x, pool_stride_y) = pad_stride_info.stride();
    const int  pool_pad_right  = pad_stride_info.pad_right();
    const int  pool_pad_top    = pad_stride_info.pad_top();
    const int  pool_pad_left   = pad_stride_info.pad_left();
    const int  pool_pad_bottom = pad_stride_info.pad_bottom();
    const bool is_square       = pool_size_x == pool_size_y;

    // Check dst dimensions
    std::tie(pooled_w, pooled_h) = scaled_dimensions(src->dimension(idx_width),
                                                     src->dimension(idx_height),
                                                     pool_size_x,
                                                     pool_size_y,
                                                     pad_stride_info);

    //If it's not squared and optimized will be executed the MxN
    num_elems_read_per_iteration      = 1;
    num_elems_processed_per_iteration = 1;
    num_elems_horizontal_window       = 1;

    if(is_square)
    {
        switch(src->data_type())
        {
            case DataType::QASYMM8:
            case DataType::QASYMM8_SIGNED:
                switch(pool_size_x)
                {
                    case 2:
                        num_elems_read_per_iteration      = 16;
                        num_elems_processed_per_iteration = (pool_stride_x == 2) ? 8 : 15;
                        num_elems_horizontal_window       = (pool_stride_x == 2) ? 8 : 16;
                        break;
                    case 3:
                        num_elems_read_per_iteration      = 16;
                        num_elems_processed_per_iteration = (pool_stride_x == 2) ? 7 : 14;
                        num_elems_horizontal_window       = (pool_stride_x == 2) ? 8 : 16;
                        break;
                    default:
                        break;
                }
                break;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            case DataType::F16:
                switch(pool_size_x)
                {
                    case 2:
                    case 3:
                        num_elems_read_per_iteration      = 4;
                        num_elems_processed_per_iteration = 1;
                        num_elems_horizontal_window       = 1;
                        break;
                    default:
                        break;
                }
                break;
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
            case DataType::F32:
                switch(pool_size_x)
                {
                    case 2:
                        num_elems_read_per_iteration = 2;
                        break;
                    case 3:
                        num_elems_read_per_iteration = 4; // We use vload4 for pooling3
                        break;
                    case 7:
                        num_elems_read_per_iteration = 8; // We use vload8 for pooling7
                        break;
                    default:
                        break;
                }
                num_elems_processed_per_iteration = 1;
                num_elems_horizontal_window       = 1;
                break;
            default:
                ARM_COMPUTE_ERROR("Element size not supported");
                break;
        }
    }

    bool   window_changed = false;
    Window win{};
    if(data_layout == DataLayout::NCHW)
    {
        // Number of iterations in X dimension
        const int num_iterations_x = (pooled_w + num_elems_processed_per_iteration - 1) / num_elems_processed_per_iteration;
        // Upper limit for the number of right/bottom border elements that are accessed
        const int upper_bound_w = ((num_iterations_x - 1) * num_elems_processed_per_iteration * pool_stride_x - pool_pad_left + num_elems_read_per_iteration) - src_width;
        const int upper_bound_h = ((pooled_h - 1) * pool_stride_y - pool_pad_top + pool_size_y) - src_height;
        border_size             = BorderSize(pool_pad_top, pool_pad_right, pool_pad_bottom, pool_pad_left);
        border_size.right       = std::max(upper_bound_w, pool_pad_right);
        border_size.bottom      = std::max(upper_bound_h, pool_pad_bottom);
        TensorShape dst_shape{ src->tensor_shape() };
        dst_shape.set(0, pooled_w);
        dst_shape.set(1, pooled_h);
        TensorInfo dst_info(src->clone()->set_tensor_shape(dst_shape));
        win = calculate_max_window(dst_info, Steps(num_elems_processed_per_iteration));
        AccessWindowStatic     src_access(src, -pool_pad_left, -pool_pad_top, ceil_to_multiple(src_width + border_size.right, pool_size_x), src_height + border_size.bottom);
        AccessWindowHorizontal dst_access(dst, 0, num_elems_horizontal_window);
        if(indices)
        {
            AccessWindowHorizontal indices_access(indices, 0, num_elems_horizontal_window);
            window_changed = update_window_and_padding(win, src_access, dst_access, indices_access);
        }
        else
        {
            window_changed = update_window_and_padding(win, src_access, dst_access);
        }
        dst_access.set_valid_region(win, ValidRegion(Coordinates(), dst->tensor_shape()));

        border_size = src->padding();
    }

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

BorderSize CpuPoolingKernel::border_size() const
{
    return _border_size;
}

void CpuPoolingKernel::configure(ITensorInfo *src, ITensorInfo *dst, const PoolingLayerInfo &pool_info, ITensorInfo *indices)
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

    // Validate pool info before calling scaled_dimensions
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments_pool_info(pool_size.x(), pool_size.y()));

    // Check dst dimensions
    unsigned int pooled_w;
    unsigned int pooled_h;
    std::tie(pooled_w, pooled_h) = scaled_dimensions(src->dimension(idx_width),
                                                     src->dimension(idx_height),
                                                     pool_size.x(),
                                                     pool_size.y(),
                                                     pad_stride_info);

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, dst, pool_info, pooled_w, pooled_h, indices, pool_size));

    // Set instance variables
    _pool_info     = pool_info;
    _data_layout   = src->data_layout();
    _pool_size     = pool_size;
    _pool_stride_x = pad_stride_info.stride().first;

    if(_data_layout == DataLayout::NHWC)
    {
        // Configure kernel window
        Window      win = calculate_max_window(*dst, Steps());
        Coordinates coord;
        coord.set_num_dimensions(dst->num_dimensions());
        dst->set_valid_region(ValidRegion(coord, dst->tensor_shape()));
        ICpuKernel::configure(win);
    }
    else
    {
        // Configure kernel window
        auto win_config = validate_and_configure_window(src, dst, indices, pool_info, _num_elems_processed_per_iteration,
                                                        _border_size, pooled_w, pooled_h, pool_size.x(), pool_size.y());
        ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
        ICpuKernel::configure(win_config.second);
    }
}

Status CpuPoolingKernel::validate(const ITensorInfo *src, const ITensorInfo *dst, const PoolingLayerInfo &pool_info, const ITensorInfo *indices)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src);

    unsigned int pooled_w                          = 0;
    unsigned int pooled_h                          = 0;
    unsigned int num_elems_processed_per_iteration = 0;
    BorderSize   border_size(0);

    const bool   is_global_pooling = pool_info.is_global_pooling;
    unsigned int pool_size_x       = 0;
    unsigned int pool_size_y       = 0;

    // Get data layout
    const auto data_layout = pool_info.data_layout == DataLayout::UNKNOWN ? src->data_layout() : pool_info.data_layout;
    const int  idx_width   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int  idx_height  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);

    pool_size_x = is_global_pooling ? src->dimension(idx_width) : pool_info.pool_size.width;
    pool_size_y = is_global_pooling ? src->dimension(idx_height) : pool_info.pool_size.height;

    // Validate pool info before calling scaled_dimensions
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments_pool_info(pool_size_x, pool_size_y));

    // Check dst dimensions
    std::tie(pooled_w, pooled_h) = scaled_dimensions(src->dimension(idx_width),
                                                     src->dimension(idx_height),
                                                     pool_size_x,
                                                     pool_size_y,
                                                     pool_info.pad_stride_info);

    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, dst, pool_info, pooled_w, pooled_h, indices, Size2D(pool_size_x, pool_size_y)));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(src->clone().get(), dst->clone().get(),
                                                              (indices) ? indices->clone().get() : nullptr, pool_info, num_elems_processed_per_iteration, border_size, pooled_w, pooled_h,
                                                              pool_size_x, pool_size_y)
                                .first);

    return Status{};
}

void CpuPoolingKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICpuKernel::window(), window);

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

    const auto *uk = get_implementation(src->info()->data_type(), _data_layout, _pool_stride_x, _pool_size);
    ARM_COMPUTE_ERROR_ON(uk == nullptr || uk->ukernel == nullptr);

    uk->ukernel(src, dst, indices, _pool_info, window_src, window);
}

const char *CpuPoolingKernel::name() const
{
    return "CpuPoolingKernel";
}
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
