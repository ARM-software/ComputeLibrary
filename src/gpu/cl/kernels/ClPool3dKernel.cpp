/*
 * Copyright (c) 2022 Arm Limited.
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
#include "src/gpu/cl/kernels/ClPool3dKernel.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "src/core/CL/CLValidate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/Cast.h"
#include "utils/TypePrinter.h"

namespace arm_compute
{
namespace opencl
{
namespace kernels
{
using namespace arm_compute::misc::shape_calculator;

namespace
{
Status validate_arguments(const ITensorInfo *src, const ITensorInfo *dst, const Pooling3dLayerInfo &pool_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(src->data_layout() != DataLayout::NDHWC, "Only NDHWC layout supported");

    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(src);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((pool_info.stride.x() == 0 || pool_info.stride.y() == 0 || pool_info.stride.z() == 0), "Strides cannot be zero.");
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::F16, DataType::F32);

    const auto   data_layout       = src->data_layout();
    const int    idx_width         = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int    idx_height        = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const int    idx_depth         = get_data_layout_dimension_index(data_layout, DataLayoutDimension::DEPTH);
    const bool   is_global_pooling = pool_info.is_global_pooling;
    const unsigned int pool_size_x       = is_global_pooling ? src->dimension(idx_width) : pool_info.pool_size.width;
    const unsigned int pool_size_y       = is_global_pooling ? src->dimension(idx_height) : pool_info.pool_size.height;
    const unsigned int pool_size_z       = is_global_pooling ? src->dimension(idx_depth) : pool_info.pool_size.depth;
    int          output_width      = 0;
    int          output_height     = 0;
    int          output_depth      = 0;

    bool round_type_ceil_with_asymm_padding = (pool_info.round_type == DimensionRoundingType::CEIL) && (!is_symmetric(pool_info.padding));
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(round_type_ceil_with_asymm_padding, "Cannot use dimension round type CEIL when padding is asymmetric.");

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(is_pool_3d_region_entirely_outside_input(pool_info), "Pooling region that is entirely outside input tensor is unsupported");
    std::tie(output_width, output_height, output_depth) = scaled_3d_dimensions_signed(src->tensor_shape()[idx_width], src->tensor_shape()[idx_height],
                                                                                      src->tensor_shape()[idx_depth], pool_size_x, pool_size_y,
                                                                                      pool_size_z, pool_info);

    ARM_COMPUTE_RETURN_ERROR_ON_MSG((output_width < 1 || output_height < 1 || output_depth < 1), "Calculated output dimension size is invalid");
    // Checks performed when dst is configured
    if(dst->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(src, dst);
        TensorInfo out_info(TensorInfo(compute_pool3d_shape(src->tensor_shape(), pool_info), 1, dst->data_type()));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(dst, &out_info);
    }

    return Status{};
}
} // namespace

ClPool3dKernel::ClPool3dKernel()
{
    _type = CLKernelType::POOL;
}

void ClPool3dKernel::configure(const ClCompileContext &compile_context, const ITensorInfo *src, ITensorInfo *dst, const Pooling3dLayerInfo &pool_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, dst, pool_info));
    auto padding_info = get_padding_info({ src, dst });

    // Auto init if empty
    TensorShape out_shape = compute_pool3d_shape(src->tensor_shape(), pool_info);
    auto_init_if_empty(*dst, src->clone()->set_tensor_shape(out_shape));

    // Set instance variables
    _pool_info   = pool_info;
    _data_layout = src->data_layout();

    _num_elems_processed_per_iteration = (dst->data_type() == DataType::F32) ? 2 : 4;
    _num_elems_processed_per_iteration = adjust_vec_size(_num_elems_processed_per_iteration, dst->dimension(0));

    const PoolingType pool_type       = pool_info.pool_type;
    const int         idx_width       = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::WIDTH);
    const int         idx_height      = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::HEIGHT);
    const int         idx_depth       = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::DEPTH);
    const int         idx_channel     = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::CHANNEL);
    const int         idx_batch_size  = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::BATCHES);
    const int         pool_size_x     = pool_info.is_global_pooling ? src->dimension(idx_width) : pool_info.pool_size.width;
    const int         pool_size_y     = pool_info.is_global_pooling ? src->dimension(idx_height) : pool_info.pool_size.height;
    const int         pool_size_z     = pool_info.is_global_pooling ? src->dimension(idx_depth) : pool_info.pool_size.depth;
    const bool        exclude_padding = pool_info.exclude_padding;
    const int         pool_stride_x   = pool_info.stride.x();
    const int         pool_stride_y   = pool_info.stride.y();
    const int         pool_stride_z   = pool_info.stride.z();
    const int         pool_pad_top    = pool_info.padding.top;
    const int         pool_pad_left   = pool_info.padding.left;
    const int         pool_pad_front  = pool_info.padding.front;
    const DataType    data_type       = src->data_type();

    // Set build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DVEC_SIZE=" + support::cpp11::to_string(_num_elems_processed_per_iteration));
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(data_type));
    build_opts.add_option("-DPOOL_" + string_from_pooling_type(pool_type));
    build_opts.add_option("-DSTRIDE_X=" + support::cpp11::to_string(pool_stride_x));
    build_opts.add_option("-DSTRIDE_Y=" + support::cpp11::to_string(pool_stride_y));
    build_opts.add_option("-DSTRIDE_Z=" + support::cpp11::to_string(pool_stride_z));
    build_opts.add_option("-DPAD_X=" + support::cpp11::to_string(pool_pad_left));
    build_opts.add_option("-DPAD_Y=" + support::cpp11::to_string(pool_pad_top));
    build_opts.add_option("-DPAD_Z=" + support::cpp11::to_string(pool_pad_front));
    build_opts.add_option("-DPOOL_SIZE_X=" + support::cpp11::to_string(pool_size_x));
    build_opts.add_option("-DPOOL_SIZE_Y=" + support::cpp11::to_string(pool_size_y));
    build_opts.add_option("-DPOOL_SIZE_Z=" + support::cpp11::to_string(pool_size_z));
    build_opts.add_option("-DSRC_WIDTH=" + support::cpp11::to_string(src->dimension(idx_width)));
    build_opts.add_option("-DSRC_HEIGHT=" + support::cpp11::to_string(src->dimension(idx_height)));
    build_opts.add_option("-DSRC_DEPTH=" + support::cpp11::to_string(src->dimension(idx_depth)));

    // Set the initial value for the pooling operation accordingly with the data type
    if(pool_type == PoolingType::MAX)
    {
        build_opts.add_option("-DINITIAL_VALUE=" + float_to_string_with_full_precision(std::numeric_limits<float>::lowest()));
    }
    else
    {
        // Pool AVG and Pool L2 initial value
        build_opts.add_option("-DINITIAL_VALUE=0");
    }
    // Create kernel
    // Floating point mixed precision is support on F16 only
    const auto use_fp_mixed_precision = (data_type == DataType::F16) && pool_info.fp_mixed_precision && pool_type != PoolingType::MAX;

    // Wider accumulation is required to avoid accuracy loss
    // Case 1: Floating point mixed precision (fp16 src data and fp32 accumulation)
    DataType acc_data_type = data_type;
    if(use_fp_mixed_precision)
    {
        acc_data_type = DataType::F32;
    }
    build_opts.add_option("-DACC_DATA_TYPE=" + get_cl_type_from_data_type(acc_data_type));
    build_opts.add_option_if(use_fp_mixed_precision, "-DFP_MIXED_PRECISION");
    build_opts.add_option_if(exclude_padding, "-DEXCLUDE_PADDING");
    build_opts.add_option("-DDST_HEIGHT=" + support::cpp11::to_string(dst->dimension(idx_height)));
    build_opts.add_option("-DDST_DEPTH=" + support::cpp11::to_string(dst->dimension(idx_depth)));
    build_opts.add_option("-DDST_CHANNELS=" + support::cpp11::to_string(dst->dimension(idx_channel)));
    build_opts.add_option("-DDST_BATCH_SIZE=" + support::cpp11::to_string(dst->dimension(idx_batch_size)));
    build_opts.add_option("-DVEC_SIZE_LEFTOVER=" + support::cpp11::to_string(src->dimension(0) % _num_elems_processed_per_iteration));
    std::string kernel_name = "pooling_3d_layer_MxN_ndhwc";
    _kernel                 = create_kernel(compile_context, kernel_name, build_opts.options());

    // Configure kernel window
    Window win = calculate_max_window(*dst, Steps(_num_elems_processed_per_iteration));
    ICLKernel::configure_internal(win);

    // Set config_id for enabling LWS tuning
    _config_id = "pooling_layer_3d";
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

Status ClPool3dKernel::validate(const ITensorInfo *src, const ITensorInfo *dst, const Pooling3dLayerInfo &pool_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, dst, pool_info));
    return Status{};
}

void ClPool3dKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    const auto src = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC));
    auto       dst = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST_0));

    // Collapse 3D window
    Window window_collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);

    // Set CL kernel arguments
    unsigned int idx = 0;
    // Passing of the window not needed, as the steps are not used for the pool3d kernel
    add_5D_tensor_argument(idx, src, window);
    add_5D_tensor_argument(idx, dst, window);
    enqueue(queue, *this, window_collapsed, lws_hint());
}
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
