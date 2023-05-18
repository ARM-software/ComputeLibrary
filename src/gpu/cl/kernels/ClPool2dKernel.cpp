/*
 * Copyright (c) 2017-2021, 2023 Arm Limited.
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
#include "src/gpu/cl/kernels/ClPool2dKernel.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "src/core/CL/CLValidate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/Cast.h"

namespace arm_compute
{
namespace opencl
{
namespace kernels
{
using namespace arm_compute::misc::shape_calculator;

namespace
{
Status validate_arguments(const ITensorInfo *src, const ITensorInfo *dst, const PoolingLayerInfo &pool_info, const ITensorInfo *indices)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(src);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((is_data_type_quantized_asymmetric(src->data_type()) && pool_info.pool_type == PoolingType::L2),
                                    "Unsupported combination of parameters!");

    const auto   data_layout       = pool_info.data_layout == DataLayout::UNKNOWN ? src->data_layout() : pool_info.data_layout;
    const int    idx_width         = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int    idx_height        = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const bool   is_global_pooling = pool_info.is_global_pooling;
    unsigned int pool_size_x       = is_global_pooling ? src->dimension(idx_width) : pool_info.pool_size.width;
    unsigned int pool_size_y       = is_global_pooling ? src->dimension(idx_height) : pool_info.pool_size.height;
    int          output_width      = 0;
    int          output_height     = 0;

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(is_pool_region_entirely_outside_input(pool_info), "Pooling region that is entirely outside input tensor is unsupported");

    std::tie(output_width, output_height) = scaled_dimensions_signed(src->tensor_shape()[idx_width], src->tensor_shape()[idx_height],
                                                                     pool_size_x, pool_size_y, pool_info.pad_stride_info);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((output_width < 1 || output_height < 1), "Calculated output dimension size is invalid");

    // Check indices
    if(indices)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::F16, DataType::F32);
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(pool_info.pool_type != PoolingType::MAX, "Pooling indices only supported for MAX pooling method");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG((pool_info.pool_size != Size2D(2, 2)), "Pooling indices only supported for pool size 2x2");

        if(indices->total_size() != 0)
        {
            TensorInfo idx_info(TensorInfo(compute_pool_shape(*src, pool_info), 1, DataType::U32));
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(indices, &idx_info);
        }
    }

    // Checks performed when dst is configured
    if(dst->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(src, dst);
        TensorInfo out_info(TensorInfo(compute_pool_shape(*src, pool_info), 1, dst->data_type()));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(dst, &out_info);
    }

    return Status{};
}
} // namespace

ClPool2dKernel::ClPool2dKernel()
{
    _type = CLKernelType::POOL;
}

void ClPool2dKernel::configure(const ClCompileContext &compile_context, ITensorInfo *src, ITensorInfo *dst, const PoolingLayerInfo &pool_info, ITensorInfo *indices)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, dst, pool_info, indices));

    auto padding_info = get_padding_info({ src, dst, indices });

    // Auto init if empty
    TensorShape out_shape = compute_pool_shape(*src, pool_info);
    auto_init_if_empty(*dst, src->clone()->set_tensor_shape(out_shape));
    if(indices)
    {
        auto_init_if_empty(*indices, src->clone()->set_tensor_shape(out_shape).set_data_type(DataType::U32));
    }

    // Set instance variables
    _pool_info                         = pool_info;
    _data_layout                       = pool_info.data_layout == DataLayout::UNKNOWN ? src->data_layout() : pool_info.data_layout;
    _num_elems_processed_per_iteration = (_data_layout == DataLayout::NCHW) ? 1 : ((dst->data_type() == DataType::F32) ? 2 : 4);
    _num_elems_processed_per_iteration = adjust_vec_size(_num_elems_processed_per_iteration, dst->dimension(0));

    int                 pool_stride_x   = 0;
    int                 pool_stride_y   = 0;
    const PoolingType   pool_type       = pool_info.pool_type;
    const int           idx_width       = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::WIDTH);
    const int           idx_height      = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::HEIGHT);
    const int           idx_channel     = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::CHANNEL);
    const int           idx_batch_size  = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::BATCHES);
    const int           pool_size_x     = pool_info.is_global_pooling ? src->dimension(idx_width) : pool_info.pool_size.width;
    const int           pool_size_y     = pool_info.is_global_pooling ? src->dimension(idx_height) : pool_info.pool_size.height;
    const PadStrideInfo pad_stride_info = pool_info.pad_stride_info;
    const bool          exclude_padding = pool_info.exclude_padding;
    std::tie(pool_stride_x, pool_stride_y) = pad_stride_info.stride();
    const int      pool_pad_top  = pad_stride_info.pad_top();
    const int      pool_pad_left = pad_stride_info.pad_left();
    const DataType data_type     = src->data_type();

    // Set build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DVEC_SIZE=" + support::cpp11::to_string(_num_elems_processed_per_iteration));
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(data_type));
    build_opts.add_option("-DPOOL_" + string_from_pooling_type(pool_type));
    build_opts.add_option("-DSTRIDE_X=" + support::cpp11::to_string(pool_stride_x));
    build_opts.add_option("-DSTRIDE_Y=" + support::cpp11::to_string(pool_stride_y));
    build_opts.add_option("-DPAD_X=" + support::cpp11::to_string(pool_pad_left));
    build_opts.add_option("-DPAD_Y=" + support::cpp11::to_string(pool_pad_top));
    build_opts.add_option("-DPOOL_SIZE_X=" + support::cpp11::to_string(pool_size_x));
    build_opts.add_option("-DPOOL_SIZE_Y=" + support::cpp11::to_string(pool_size_y));
    build_opts.add_option("-DSRC_WIDTH=" + support::cpp11::to_string(src->dimension(idx_width)));
    build_opts.add_option("-DSRC_HEIGHT=" + support::cpp11::to_string(src->dimension(idx_height)));
    build_opts.add_option("-DMAX_WIDTH=" + support::cpp11::to_string(src->dimension(idx_width) + (exclude_padding ? 0 : pool_pad_left)));
    build_opts.add_option("-DMAX_HEIGHT=" + support::cpp11::to_string(src->dimension(idx_height) + (exclude_padding ? 0 : pool_pad_top)));

    // Tensor paddings are used to calculate the indicies for MAX pooling
    if(pool_info.pool_size == Size2D(2, 2) && pool_type == PoolingType::MAX && indices && is_data_type_float(data_type))
    {
        build_opts.add_option("-DSRC_BATCH=" + support::cpp11::to_string(src->tensor_shape().total_size_lower(3)));
    }

    if(is_data_type_quantized_asymmetric(data_type))
    {
        build_opts.add_option("-DQUANTIZED");

        if(src->quantization_info() != dst->quantization_info())
        {
            const UniformQuantizationInfo iq_info = src->quantization_info().uniform();
            const UniformQuantizationInfo oq_info = dst->quantization_info().uniform();

            build_opts.add_option("-DOFFSET_IN1=" + float_to_string_with_full_precision(iq_info.offset));
            build_opts.add_option("-DOFFSET_OUT=" + float_to_string_with_full_precision(oq_info.offset));
            build_opts.add_option("-DSCALE_IN1=" + float_to_string_with_full_precision(iq_info.scale));
            build_opts.add_option("-DSCALE_OUT=" + float_to_string_with_full_precision(oq_info.scale));
        }
    }

    // Set the initial value for the pooling operation accordingly with the data type
    if(pool_type == PoolingType::MAX)
    {
        if(is_data_type_quantized(data_type))
        {
            PixelValue type_min{};
            std::tie(type_min, std::ignore) = get_min_max(data_type);
            build_opts.add_option("-DINITIAL_VALUE=" + support::cpp11::to_string(type_min.get<int32_t>()));
        }
        else
        {
            std::string initial_value = pool_info.use_inf_as_limit ? "(-INFINITY)" : float_to_string_with_full_precision(std::numeric_limits<float>::lowest());
            build_opts.add_option("-DINITIAL_VALUE=" + initial_value);
        }
    }
    else
    {
        // Pool AVG and Pool L2 initial value
        build_opts.add_option("-DINITIAL_VALUE=0");
    }

    // Create kernel
    switch(_data_layout)
    {
        case DataLayout::NCHW:
        {
            const auto use_fp_mixed_precision = (data_type == DataType::F16) && pool_info.fp_mixed_precision;
            const auto use_wider_accumulator  = use_fp_mixed_precision && (pool_type != PoolingType::MAX);
            const auto acc_data_type          = get_cl_type_from_data_type(use_wider_accumulator ? DataType::F32 : (is_data_type_quantized(data_type) ? DataType::S32 : data_type));
            build_opts.add_option("-DACC_DATA_TYPE=" + acc_data_type);
            build_opts.add_option_if(use_wider_accumulator, "-DFP_MIXED_PRECISION");

            if(pool_type != PoolingType::MAX)
            {
                build_opts.add_option_if(exclude_padding, "-DEXCLUDE_PADDING");
            }

            if(pool_info.pool_size == Size2D(2, 2) && pool_type == PoolingType::MAX && indices && is_data_type_float(data_type))
            {
                // For max pooling with pool2x2, store indicies which will be used in max unpooling
                std::string kernel_name = "pooling_layer_2_nchw_indices";
                _kernel                 = create_kernel(compile_context, kernel_name, build_opts.options());
            }
            else // Run general case
            {
                std::string kernel_name = "pooling_layer_MxN_nchw";
                _kernel                 = create_kernel(compile_context, kernel_name, build_opts.options());
            }
            break;
        }
        case DataLayout::NHWC:
        {
            // Floating point mixed precision is support on F16 only
            const auto use_fp_mixed_precision = (data_type == DataType::F16) && pool_info.fp_mixed_precision && pool_type != PoolingType::MAX;

            // Wider accumulation is required to avoid accuracy loss
            // Case 1: Floating point mixed precision (fp16 src data and fp32 accumulation)
            // Cast 2: Quantized (int8/uint8 src data and int32 accumulation )
            DataType acc_data_type = data_type;

            if(use_fp_mixed_precision)
            {
                acc_data_type = DataType::F32;
            }
            else if(is_data_type_quantized(data_type) && pool_type != PoolingType::MAX)
            {
                acc_data_type = DataType::S32;
            }

            build_opts.add_option("-DACC_DATA_TYPE=" + get_cl_type_from_data_type(acc_data_type));
            build_opts.add_option_if(use_fp_mixed_precision, "-DFP_MIXED_PRECISION");
            build_opts.add_option_if(exclude_padding, "-DEXCLUDE_PADDING");
            build_opts.add_option("-DSRC_WIDTH=" + support::cpp11::to_string(src->dimension(idx_width)));
            build_opts.add_option("-DSRC_HEIGHT=" + support::cpp11::to_string(src->dimension(idx_height)));
            build_opts.add_option("-DDST_HEIGHT=" + support::cpp11::to_string(dst->dimension(idx_height)));
            build_opts.add_option("-DDST_CHANNELS=" + support::cpp11::to_string(dst->dimension(idx_channel)));
            build_opts.add_option("-DDST_BATCH_SIZE=" + support::cpp11::to_string(dst->dimension(idx_batch_size)));
            build_opts.add_option("-DVEC_SIZE_LEFTOVER=" + support::cpp11::to_string(src->dimension(0) % _num_elems_processed_per_iteration));
            if(pool_info.pool_size == Size2D(2, 2) && is_data_type_float(data_type))
            {
                build_opts.add_option_if(indices != nullptr && pool_type == PoolingType::MAX, "-DEXTRACT_MAX_INDEX");

                std::string kernel_name = "pooling_layer_2x2_nhwc";
                _kernel                 = create_kernel(compile_context, kernel_name, build_opts.options());
            }
            else
            {
                std::string kernel_name = is_data_type_quantized_asymmetric(data_type) ? "pooling_layer_MxN_quantized_nhwc" : "pooling_layer_MxN_nhwc";
                _kernel                 = create_kernel(compile_context, kernel_name, build_opts.options());
            }
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Not implemented");
    }

    // Configure kernel window
    Window win = calculate_max_window(*dst, Steps(_num_elems_processed_per_iteration));
    ICLKernel::configure_internal(win);

    // Set config_id for enabling LWS tuning
    _config_id = "pooling_layer_";
    _config_id += lower_string(string_from_data_type(data_type));
    _config_id += "_";
    _config_id += lower_string(string_from_data_layout(_data_layout));
    _config_id += "_";
    _config_id += support::cpp11::to_string(dst->dimension(idx_width));
    _config_id += "_";
    _config_id += support::cpp11::to_string(dst->dimension(idx_height));
    _config_id += "_";
    _config_id += support::cpp11::to_string(dst->dimension(idx_channel));
    _config_id += "_";
    _config_id += lower_string(string_from_data_layout(src->data_layout()));

    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
}

Status ClPool2dKernel::validate(const ITensorInfo *src, const ITensorInfo *dst, const PoolingLayerInfo &pool_info, const ITensorInfo *indices)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, dst, pool_info, indices));
    return Status{};
}

void ClPool2dKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    unsigned int pool_stride_x = 0;
    unsigned int pool_stride_y = 0;
    std::tie(pool_stride_x, pool_stride_y) = _pool_info.pad_stride_info.stride();

    const auto src     = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC));
    auto       dst     = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST_0));
    auto       indices = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST_1));

    // Collapse window
    Window window_collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);

    switch(_data_layout)
    {
        case DataLayout::NCHW:
        {
            Window slice = window_collapsed.first_slice_window_3D();
            do
            {
                // Set srcs
                unsigned int idx = 0;
                add_3D_tensor_argument(idx, src, slice);
                add_3D_tensor_argument(idx, dst, slice);
                if(indices && is_data_type_float(src->info()->data_type()) && (_pool_info.pool_size == Size2D(2, 2)))
                {
                    add_3D_tensor_argument(idx, indices, slice);
                }
                enqueue(queue, *this, slice, lws_hint());
            }
            while(window_collapsed.slide_window_slice_3D(slice));
            break;
        }
        case DataLayout::NHWC:
        {
            const size_t batch_size = dst->info()->tensor_shape().total_size_upper(3);

            Window slice    = window_collapsed.first_slice_window_4D();
            Window in_slice = window_collapsed.first_slice_window_4D();
            in_slice.set(Window::DimX, Window::Dimension(0, src->info()->dimension(0), _num_elems_processed_per_iteration));
            in_slice.set(Window::DimY, Window::Dimension(0, src->info()->dimension(1), pool_stride_x));
            in_slice.set(Window::DimZ, Window::Dimension(0, src->info()->dimension(2), pool_stride_y));
            in_slice.set(3, Window::Dimension(0, batch_size, 1));
            do
            {
                // Set srcs
                unsigned int idx = 0;
                add_4D_tensor_argument(idx, src, in_slice);
                add_4D_tensor_argument(idx, dst, slice);
                if(indices && is_data_type_float(src->info()->data_type()) && (_pool_info.pool_type == PoolingType::MAX) && (_pool_info.pool_size == Size2D(2, 2)))
                {
                    add_4D_tensor_argument(idx, indices, slice);
                }
                enqueue(queue, *this, slice, lws_hint());
            }
            while(window.slide_window_slice_4D(slice) && window.slide_window_slice_4D(in_slice));
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Not implemented");
    }
}
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
