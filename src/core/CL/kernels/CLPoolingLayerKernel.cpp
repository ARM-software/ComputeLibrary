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
#include "src/core/CL/kernels/CLPoolingLayerKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "src/core/AccessWindowStatic.h"
#include "src/core/CL/CLValidate.h"
#include "src/core/CL/ICLKernel.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/StringSupport.h"

#include <set>
#include <string>
#include <tuple>

namespace arm_compute
{
using namespace arm_compute::misc::shape_calculator;

namespace
{
// Internal window config info
using CLPoolingConfig = std::pair<unsigned int, BorderSize>; //num_elems_processed_per_iteration, border_size

void auto_init(const ITensorInfo *input, ITensorInfo *output, ITensorInfo *indices, PoolingLayerInfo pool_info)
{
    TensorShape out_shape = compute_pool_shape(*input, pool_info);
    auto_init_if_empty(*output, input->clone()->set_tensor_shape(out_shape));
    if(indices)
    {
        auto_init_if_empty(*indices, input->clone()->set_tensor_shape(out_shape).set_data_type(DataType::U32));
    }
}

Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const PoolingLayerInfo &pool_info, const ITensorInfo *indices)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((is_data_type_quantized_asymmetric(input->data_type()) && pool_info.pool_type == PoolingType::L2),
                                    "Unsupported combination of parameters!");

    // Check indices
    if(indices)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(pool_info.pool_type != PoolingType::MAX, "Pooling indices only supported for MAX pooling method");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG((pool_info.pool_size != Size2D(2, 2)), "Pooling indices only supported for pool size 2x2");

        if(indices->total_size() != 0)
        {
            TensorInfo idx_info(TensorInfo(compute_pool_shape(*input, pool_info), 1, DataType::U32));
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(indices, &idx_info);
        }
    }

    // Checks performed when output is configured
    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(input, output);
        TensorInfo out_info(TensorInfo(compute_pool_shape(*input, pool_info), 1, output->data_type()));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(output, &out_info);
    }

    return Status{};
}

std::tuple<Status, Window, CLPoolingConfig> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output, const PoolingLayerInfo &pool_info, ITensorInfo *indices = nullptr)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    // Get data layout
    const DataLayout data_layout = pool_info.data_layout == DataLayout::UNKNOWN ? input->data_layout() : pool_info.data_layout;
    const int        idx_width   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int        idx_height  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);

    int                 pool_stride_x   = 0;
    int                 pool_stride_y   = 0;
    unsigned int        pooled_w        = 0;
    unsigned int        pooled_h        = 0;
    int                 pool_size_x     = pool_info.is_global_pooling ? input->dimension(idx_width) : pool_info.pool_size.width;
    int                 pool_size_y     = pool_info.is_global_pooling ? input->dimension(idx_height) : pool_info.pool_size.height;
    const PadStrideInfo pad_stride_info = pool_info.pad_stride_info;
    std::tie(pool_stride_x, pool_stride_y) = pad_stride_info.stride();
    const int  pool_pad_right  = pad_stride_info.pad_right();
    const int  pool_pad_top    = pad_stride_info.pad_top();
    const int  pool_pad_left   = pad_stride_info.pad_left();
    const int  pool_pad_bottom = pad_stride_info.pad_bottom();
    BorderSize border_size     = BorderSize();

    auto_init(input, output, indices, pool_info);
    pooled_w = output->tensor_shape()[idx_width];
    pooled_h = output->tensor_shape()[idx_height];

    const DataType data_type = input->data_type();

    const int input_width  = input->dimension(idx_width);
    const int input_height = input->dimension(idx_height);

    unsigned int num_elems_processed_per_iteration = 0;
    bool         window_changed                    = false;
    Window       win{};
    switch(data_layout)
    {
        case DataLayout::NCHW:
        {
            // Initialize border size
            border_size = BorderSize(pool_pad_top, pool_pad_right, pool_pad_bottom, pool_pad_left);
            // Change the number of elements processed per iteration
            // for pooling 3x3 with stride less equal than 3
            const bool can_optimize                         = (pool_size_x == 3) && (pool_size_y == 3) && (pool_stride_x <= 3) && !is_data_type_quantized(data_type);
            num_elems_processed_per_iteration               = can_optimize ? 4 : 1;
            const unsigned int num_elems_read_per_iteration = (num_elems_processed_per_iteration - 1) * pool_stride_x + pool_size_x;

            // Number of iterations in X dimension
            const int num_iterations_x = (pooled_w + num_elems_processed_per_iteration - 1) / num_elems_processed_per_iteration;

            // Upper limit for the number of right/bottom border elements that are accessed
            const int upper_bound_w = ((num_iterations_x - 1) * num_elems_processed_per_iteration * pool_stride_x - pool_pad_left + num_elems_read_per_iteration) - input_width;
            const int upper_bound_h = ((pooled_h - 1) * pool_stride_y - pool_pad_top + pool_size_y) - input_height;

            border_size.right  = std::max(upper_bound_w, pool_pad_right);
            border_size.bottom = std::max(upper_bound_h, pool_pad_bottom);

            win = calculate_max_window(*output, Steps(num_elems_processed_per_iteration));

            AccessWindowRectangle input_access(input, -pool_pad_left, -pool_pad_top, num_elems_read_per_iteration, pool_size_y,
                                               pool_stride_x, pool_stride_y);
            AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);

            // Update indices window
            if(indices)
            {
                AccessWindowHorizontal indices_access(indices, 0, num_elems_processed_per_iteration);
                window_changed = update_window_and_padding(win, input_access, output_access, indices_access);
                indices_access.set_valid_region(win, ValidRegion(Coordinates(), indices->tensor_shape()));
            }
            else
            {
                window_changed = update_window_and_padding(win, input_access, output_access);
            }

            output_access.set_valid_region(win, ValidRegion(Coordinates(), output->tensor_shape()));
            break;
        }
        case DataLayout::NHWC:
        {
            // Initialize border size
            border_size                       = BorderSize();
            num_elems_processed_per_iteration = adjust_vec_size(4, output->dimension(0));
            win                               = calculate_max_window(*output, Steps(num_elems_processed_per_iteration));

            if(indices != nullptr)
            {
                indices->set_valid_region(ValidRegion(Coordinates(), indices->tensor_shape()));
            }

            output->set_valid_region(ValidRegion(Coordinates(), output->tensor_shape()));
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Not implemented");
    }

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_tuple(err, win, CLPoolingConfig(num_elems_processed_per_iteration, border_size));
}
} // namespace

CLPoolingLayerKernel::CLPoolingLayerKernel()
    : _input(nullptr), _output(nullptr), _indices(nullptr), _pool_info(), _data_layout(DataLayout::UNKNOWN), _border_size(0), _num_elems_processed_per_iteration(1)
{
}

BorderSize CLPoolingLayerKernel::border_size() const
{
    return _border_size;
}

void CLPoolingLayerKernel::configure(const ICLTensor *input, ICLTensor *output, const PoolingLayerInfo &pool_info, ICLTensor *indices)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output, pool_info, indices);
}

void CLPoolingLayerKernel::configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output, const PoolingLayerInfo &pool_info, ICLTensor *indices)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    auto padding_info = get_padding_info({ input, output, indices });

    // Set instance variables
    _input                              = input;
    _output                             = output;
    _pool_info                          = pool_info;
    _data_layout                        = pool_info.data_layout == DataLayout::UNKNOWN ? input->info()->data_layout() : pool_info.data_layout;
    _indices                            = indices;
    int                 pool_stride_x   = 0;
    int                 pool_stride_y   = 0;
    const PoolingType   pool_type       = pool_info.pool_type;
    const int           idx_width       = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::WIDTH);
    const int           idx_height      = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::HEIGHT);
    const int           idx_channel     = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::CHANNEL);
    const int           idx_batch_size  = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::BATCHES);
    const int           pool_size_x     = pool_info.is_global_pooling ? input->info()->dimension(idx_width) : pool_info.pool_size.width;
    const int           pool_size_y     = pool_info.is_global_pooling ? input->info()->dimension(idx_height) : pool_info.pool_size.height;
    const PadStrideInfo pad_stride_info = pool_info.pad_stride_info;
    const bool          exclude_padding = pool_info.exclude_padding;
    std::tie(pool_stride_x, pool_stride_y) = pad_stride_info.stride();
    const int pool_pad_top  = pad_stride_info.pad_top();
    const int pool_pad_left = pad_stride_info.pad_left();

    // Set build options
    CLBuildOptions build_opts;
    const DataType data_type = input->info()->data_type();

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), output->info(), pool_info, (indices ? indices->info() : nullptr));

    ARM_COMPUTE_ERROR_THROW_ON(std::get<0>(win_config));
    ICLKernel::configure_internal(std::get<1>(win_config));

    CLPoolingConfig pooling_config     = std::get<2>(win_config);
    _num_elems_processed_per_iteration = pooling_config.first;
    _border_size                       = pooling_config.second;

    build_opts.add_option("-DVEC_SIZE=" + support::cpp11::to_string(_num_elems_processed_per_iteration));

    // Tensor paddings are used to calculate the indicies for MAX pooling
    if(pool_info.pool_size == Size2D(2, 2) && pool_type == PoolingType::MAX && _indices && is_data_type_float(data_type))
    {
        build_opts.add_option("-DPAD_TENSOR_LEFT=" + support::cpp11::to_string(input->info()->padding().left));
        build_opts.add_option("-DPAD_TENSOR_RIGHT=" + support::cpp11::to_string(input->info()->padding().right));
        build_opts.add_option("-DPAD_TENSOR_TOP=" + support::cpp11::to_string(input->info()->padding().top));
        build_opts.add_option("-DPAD_TENSOR_BOTTOM=" + support::cpp11::to_string(input->info()->padding().bottom));
        build_opts.add_option("-DTENSOR_CHANNEL=" + support::cpp11::to_string(input->info()->dimension(idx_channel)));
        build_opts.add_option("-DTENSOR_WIDTH=" + support::cpp11::to_string(input->info()->dimension(idx_width)));
        build_opts.add_option("-DTENSOR_HEIGHT=" + support::cpp11::to_string(input->info()->dimension(idx_height)));
    }

    if(is_data_type_quantized_asymmetric(data_type) && input->info()->quantization_info() != output->info()->quantization_info())
    {
        const UniformQuantizationInfo iq_info = input->info()->quantization_info().uniform();
        const UniformQuantizationInfo oq_info = output->info()->quantization_info().uniform();

        build_opts.add_option("-DOFFSET_IN1=" + float_to_string_with_full_precision(iq_info.offset));
        build_opts.add_option("-DOFFSET_OUT=" + float_to_string_with_full_precision(oq_info.offset));
        build_opts.add_option("-DSCALE_IN1=" + float_to_string_with_full_precision(iq_info.scale));
        build_opts.add_option("-DSCALE_OUT=" + float_to_string_with_full_precision(oq_info.scale));
    }

    // Check output dimensions
    auto_init(input->info(), output->info(), indices ? indices->info() : nullptr, pool_info);

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), pool_info, (indices) ? indices->info() : nullptr));

    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(data_type));
    build_opts.add_option("-DPOOL_" + string_from_pooling_type(pool_type));
    build_opts.add_option("-DSTRIDE_X=" + support::cpp11::to_string(pool_stride_x));
    build_opts.add_option("-DSTRIDE_Y=" + support::cpp11::to_string(pool_stride_y));
    build_opts.add_option("-DPAD_X=" + support::cpp11::to_string(pool_pad_left));
    build_opts.add_option("-DPAD_Y=" + support::cpp11::to_string(pool_pad_top));
    build_opts.add_option("-DPOOL_SIZE_X=" + support::cpp11::to_string(pool_size_x));
    build_opts.add_option("-DPOOL_SIZE_Y=" + support::cpp11::to_string(pool_size_y));

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
            build_opts.add_option("-DINITIAL_VALUE=" + float_to_string_with_full_precision(std::numeric_limits<float>::lowest()));
        }
    }
    else
    {
        // Pool AVG and Pool L2 initial value
        build_opts.add_option("-DINITIAL_VALUE=0");
    }

    build_opts.add_option("-DMAX_WIDTH=" + support::cpp11::to_string(input->info()->dimension(idx_width) + (exclude_padding ? 0 : pool_pad_left)));
    build_opts.add_option("-DMAX_HEIGHT=" + support::cpp11::to_string(input->info()->dimension(idx_height) + (exclude_padding ? 0 : pool_pad_top)));

    // Create kernel
    switch(_data_layout)
    {
        case DataLayout::NCHW:
        {
            const auto use_fp_mixed_precision = (data_type == DataType::F16) && pool_info.fp_mixed_precision;
            const auto use_wider_accumulator  = use_fp_mixed_precision && (pool_type != PoolingType::MAX);
            const auto acc_data_type          = get_cl_type_from_data_type(use_wider_accumulator ? DataType::F32 : data_type);
            build_opts.add_option("-DACC_DATA_TYPE=" + acc_data_type);
            build_opts.add_option_if(use_wider_accumulator, "-DFP_MIXED_PRECISION");

            if(pool_type != PoolingType::MAX)
            {
                build_opts.add_option_if(exclude_padding, "-DEXCLUDE_PADDING");
            }

            if((pool_size_x == 3) && (pool_size_y == 3) && !is_data_type_quantized_asymmetric(data_type))
            {
                // Check if we have pool3x3 with stride_x less equal than 3. In these cases, run an optimized OpenCL kernel where
                // each thread computes 4 output elements
                const bool is_pool3x3_stride_le3 = (pool_size_x == 3) && (pool_size_y == 3) && (pool_stride_x <= 3);

                std::string kernel_name = ((is_pool3x3_stride_le3) ? "pooling_layer_optimized_" : "pooling_layer_")
                                          + support::cpp11::to_string(pool_size_x);
                _kernel = create_kernel(compile_context, kernel_name, build_opts.options());
            }
            else if(pool_info.pool_size == Size2D(2, 2) && pool_type == PoolingType::MAX && _indices && is_data_type_float(data_type))
            {
                // For max pooling with pool2x2, store indicies which will be used in max unpooling
                if(data_type == DataType::F32)
                {
                    std::string kernel_name = "pooling_layer_2_nchw_indices_fp32";
                    _kernel                 = create_kernel(compile_context, kernel_name, build_opts.options());
                }
                else if(data_type == DataType::F16)
                {
                    std::string kernel_name = "pooling_layer_2_nchw_indices_fp16";
                    _kernel                 = create_kernel(compile_context, kernel_name, build_opts.options());
                }
            }
            else // Run general case
            {
                std::string kernel_name = is_data_type_quantized_asymmetric(data_type) ? "pooling_layer_MxN_quantized_nchw" : "pooling_layer_MxN_nchw";
                _kernel                 = create_kernel(compile_context, kernel_name, build_opts.options());
            }
            break;
        }
        case DataLayout::NHWC:
        {
            // Floating point mixed precision is support on F16 only
            const auto use_fp_mixed_precision = (data_type == DataType::F16) && pool_info.fp_mixed_precision && pool_type != PoolingType::MAX;

            // Wider accumulation is required to avoid accuracy loss
            // Case 1: Floating point mixed precision (fp16 input data and fp32 accumulation)
            // Cast 2: Quantized (int8/uint8 input data and int32 accumulation )
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
            build_opts.add_option("-DSRC_WIDTH=" + support::cpp11::to_string(input->info()->dimension(idx_width)));
            build_opts.add_option("-DSRC_HEIGHT=" + support::cpp11::to_string(input->info()->dimension(idx_height)));
            build_opts.add_option("-DDST_HEIGHT=" + support::cpp11::to_string(output->info()->dimension(idx_height)));
            build_opts.add_option("-DDST_CHANNELS=" + support::cpp11::to_string(output->info()->dimension(idx_channel)));
            build_opts.add_option("-DDST_BATCH_SIZE=" + support::cpp11::to_string(output->info()->dimension(idx_batch_size)));
            build_opts.add_option("-DVEC_SIZE_LEFTOVER=" + support::cpp11::to_string(input->info()->dimension(0) % _num_elems_processed_per_iteration));
            if(pool_info.pool_size == Size2D(2, 2) && is_data_type_float(data_type))
            {
                build_opts.add_option_if(_indices != nullptr && pool_type == PoolingType::MAX, "-DEXTRACT_MAX_INDEX");

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

    // Set config_id for enabling LWS tuning
    _config_id = "pooling_layer_";
    _config_id += lower_string(string_from_data_type(data_type));
    _config_id += "_";
    _config_id += lower_string(string_from_data_layout(_data_layout));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(idx_width));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(idx_height));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(idx_channel));
    _config_id += "_";
    _config_id += lower_string(string_from_data_layout(input->info()->data_layout()));

    ARM_COMPUTE_ERROR_ON(input->info()->data_layout() == DataLayout::NHWC && has_padding_changed(padding_info));
}

Status CLPoolingLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const PoolingLayerInfo &pool_info, const ITensorInfo *indices)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, pool_info, indices));
    ARM_COMPUTE_RETURN_ON_ERROR(std::get<0>(validate_and_configure_window(input->clone().get(), output->clone().get(), pool_info)));

    return Status{};
}

void CLPoolingLayerKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    unsigned int pool_stride_x = 0;
    unsigned int pool_stride_y = 0;
    std::tie(pool_stride_x, pool_stride_y) = _pool_info.pad_stride_info.stride();

    // Collapse window
    Window window_collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);

    switch(_data_layout)
    {
        case DataLayout::NCHW:
        {
            Window slice = window_collapsed.first_slice_window_3D();
            do
            {
                // Upsample input by pool size
                Window in_slice(slice);
                in_slice.set(Window::DimX, Window::Dimension(in_slice.x().start() - _pool_info.pad_stride_info.pad_left(),
                                                             (in_slice.x().end() - _pool_info.pad_stride_info.pad_left()) * pool_stride_x,
                                                             pool_stride_x * _num_elems_processed_per_iteration));
                in_slice.set(Window::DimY, Window::Dimension(in_slice.y().start() - _pool_info.pad_stride_info.pad_top(),
                                                             (in_slice.y().end() - _pool_info.pad_stride_info.pad_top()) * pool_stride_y,
                                                             pool_stride_y));

                // Set inputs
                unsigned int idx = 0;
                add_3D_tensor_argument(idx, _input, in_slice);
                add_3D_tensor_argument(idx, _output, slice);
                if(_indices && is_data_type_float(_input->info()->data_type()) && (_pool_info.pool_size == Size2D(2, 2)))
                {
                    add_3D_tensor_argument(idx, _indices, slice);
                }
                enqueue(queue, *this, slice, lws_hint());
            }
            while(window_collapsed.slide_window_slice_3D(slice));
            break;
        }
        case DataLayout::NHWC:
        {
            const size_t batch_size = _output->info()->tensor_shape().total_size_upper(3);

            Window slice    = window_collapsed.first_slice_window_4D();
            Window in_slice = window_collapsed.first_slice_window_4D();
            in_slice.set(Window::DimX, Window::Dimension(0, _input->info()->dimension(0), _num_elems_processed_per_iteration));
            in_slice.set(Window::DimY, Window::Dimension(0, _input->info()->dimension(1), pool_stride_x));
            in_slice.set(Window::DimZ, Window::Dimension(0, _input->info()->dimension(2), pool_stride_y));
            in_slice.set(3, Window::Dimension(0, batch_size, 1));
            do
            {
                // Set inputs
                unsigned int idx = 0;
                add_4D_tensor_argument(idx, _input, in_slice);
                add_4D_tensor_argument(idx, _output, slice);
                if(_indices && is_data_type_float(_input->info()->data_type()) && (_pool_info.pool_type == PoolingType::MAX) && (_pool_info.pool_size == Size2D(2, 2)))
                {
                    add_4D_tensor_argument(idx, _indices, slice);
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
} // namespace arm_compute
