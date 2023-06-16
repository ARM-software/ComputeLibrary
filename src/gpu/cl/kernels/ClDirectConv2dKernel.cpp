/*
 * Copyright (c) 2017-2023 Arm Limited.
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
#include "src/gpu/cl/kernels/ClDirectConv2dKernel.h"

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "src/core/AccessWindowStatic.h"
#include "src/core/CL/CLUtils.h"
#include "src/core/CL/CLValidate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/gpu/cl/kernels/gemm/ClGemmHelpers.h"
#include "support/Cast.h"
#include "support/StringSupport.h"

namespace arm_compute
{
namespace opencl
{
namespace kernels
{
namespace
{
Status validate_arguments(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *dst,
                          const PadStrideInfo &conv_info, const ActivationLayerInfo &act_info, const DirectConvComputeKernelInfo &desc)
{
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(src);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::QASYMM8_SIGNED, DataType::QASYMM8, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, weights);

    const DataLayout data_layout = src->data_layout();
    const int        width_idx   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int        height_idx  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const int        channel_idx = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(weights->dimension(channel_idx) != src->dimension(channel_idx), "Weights feature map dimension should match the respective src's one");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(weights->num_dimensions() > 4, "Weights can be at most 4 dimensional");

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(desc.export_input_to_cl_image == true, "Export to CLImage is not supported for the input tensor");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(desc.export_output_to_cl_image == true, "Export to CLImage is not supported for the output tensor");

    if(data_layout == DataLayout::NCHW)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(weights->dimension(width_idx) != weights->dimension(height_idx), "Weights should have same width and height");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG((weights->dimension(width_idx) == 1) && std::get<0>(conv_info.stride()) > 3, "Strides larger than 3 not supported for 1x1 convolution.");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG((weights->dimension(width_idx) == 3 || weights->dimension(width_idx) == 5 || weights->dimension(width_idx) == 9) && std::get<0>(conv_info.stride()) > 2,
                                        "Strides larger than 2 not supported for 3x3, 5x5, 9x9 convolution.");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(act_info.enabled(), "Fused activation is not supported for NCHW layout");

        if(is_data_type_quantized(src->data_type()))
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MSG(weights->dimension(width_idx) != 1 && weights->dimension(width_idx) != 3 && weights->dimension(width_idx) != 5 && weights->dimension(width_idx) != 9,
                                            "Kernel sizes other than 1x1, 3x3, 5x5 or 9x9 are not supported with quantized data types");
        }
        else
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MSG(weights->dimension(width_idx) != 1 && weights->dimension(width_idx) != 3 && weights->dimension(width_idx) != 5,
                                            "Kernel sizes other than 1x1, 3x3 or 5x5 are not supported with float data types");
        }
    }

    if(data_layout == DataLayout::NHWC)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(act_info.enabled() && !is_data_type_float(src->data_type()), "Fused activation in NHWC is only supported for floating point.");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(desc.m0 <= 0 || desc.m0 > 8, "M0 can only be greater than 0 and less than or equal to 8");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(desc.n0 != 1 && desc.n0 != 2 && desc.n0 != 3 && desc.n0 != 4 && desc.n0 != 8 && desc.n0 != 16,
                                        "N0 can only be: 1, 2, 3, 4, 8, and 16");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(desc.k0 != 1 && desc.k0 != 2 && desc.k0 != 3 && desc.k0 != 4 && desc.k0 != 8 && desc.k0 != 16,
                                        "K0 can only be: 1, 2, 3, 4, 8, and 16");
        if(desc.export_weights_to_cl_image)
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MSG(desc.k0 != 4 && desc.k0 != 8 && desc.k0 != 16,
                                            "K0 can only be: 4, 8, and 16");
            ARM_COMPUTE_RETURN_ERROR_ON_MSG(!export_to_cl_image(weights),
                                            "Export to CLImage is not supported for this weight configuration");
        }
    }

    if(biases != nullptr)
    {
        if(is_data_type_quantized_asymmetric(src->data_type()))
        {
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(biases, 1, DataType::S32);
        }
        else
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(weights, biases);
        }
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(biases->dimension(0) != weights->dimension(3),
                                        "Biases size and number of dst feature maps should match");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(biases->num_dimensions() > 1,
                                        "Biases should be one dimensional");
    }

    // Checks performed when dst is configured
    if(dst->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(dst->tensor_shape(),
                                                           misc::shape_calculator::compute_deep_convolution_shape(*src, *weights, conv_info));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
    }

    const auto data_type = src->data_type();
    if(is_data_type_quantized(data_type))
    {
        const UniformQuantizationInfo iqinfo = src->quantization_info().uniform();
        const UniformQuantizationInfo wqinfo = weights->quantization_info().uniform();
        const UniformQuantizationInfo oqinfo = dst->quantization_info().uniform();

        float multiplier        = iqinfo.scale * wqinfo.scale / oqinfo.scale;
        int   output_multiplier = 0;
        int   output_shift      = 0;
        ARM_COMPUTE_RETURN_ON_ERROR(quantization::calculate_quantized_multiplier(multiplier, &output_multiplier, &output_shift));
    }
    return Status{};
}
} // namespace

ClDirectConv2dKernel::ClDirectConv2dKernel()
{
    _type = CLKernelType::DIRECT;
}

void ClDirectConv2dKernel::configure(const CLCompileContext &compile_context, ITensorInfo *src, ITensorInfo *weights, ITensorInfo *biases, ITensorInfo *dst,
                                     const PadStrideInfo &conv_info, const ActivationLayerInfo &act_info, const DirectConvComputeKernelInfo &desc)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, weights, dst);

    // Perform validation
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, weights, biases, dst, conv_info, act_info, desc));

    const int conv_stride_x = std::get<0>(conv_info.stride());
    const int conv_stride_y = std::get<1>(conv_info.stride());

    _data_layout = src->data_layout();
    _conv_info   = conv_info;

    const unsigned int width_idx   = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::WIDTH);
    const unsigned int height_idx  = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::HEIGHT);
    const unsigned int channel_idx = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::CHANNEL);
    const unsigned int kernel_size = weights->dimension(width_idx);
    const DataType     data_type   = src->data_type();

    const GPUTarget gpu_target                         = get_target();
    unsigned int    _num_elems_processed_per_iteration = 0;

    // Get dst shape
    TensorShape output_shape = misc::shape_calculator::compute_deep_convolution_shape(*src, *weights, conv_info);

    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*dst, output_shape,
                       1,
                       src->data_type(),
                       src->quantization_info());

    // Configure kernel window
    Window win;
    if(_data_layout == DataLayout::NHWC)
    {
        output_shape.collapse(2U, 1U);
        const unsigned int n0 = adjust_vec_size(desc.n0, output_shape[0]);
        const unsigned int m0 = adjust_vec_size(desc.m0, output_shape[1]);

        // Create window and update padding
        win = calculate_max_window(output_shape, Steps(n0, m0));
    }
    else if(_data_layout == DataLayout::NCHW)
    {
        _num_elems_processed_per_iteration = 1u;
        win                                = calculate_max_window(*dst, Steps(_num_elems_processed_per_iteration));
    }

    ICLKernel::configure_internal(win);

    std::stringstream kernel_name;
    CLBuildOptions    build_options;

    if(_data_layout == DataLayout::NHWC)
    {
        kernel_name << "direct_convolution_nhwc";

        const unsigned int n0               = win.x().step();
        const unsigned int m0               = win.y().step();
        const unsigned int k0               = adjust_vec_size(desc.k0, src->dimension(channel_idx));
        const unsigned int partial_store_n0 = dst->dimension(channel_idx) % n0;
        const unsigned int pad_left         = conv_info.pad_left();
        const unsigned int pad_top          = conv_info.pad_top();

        _export_weights_to_cl_image = desc.export_weights_to_cl_image;
        _export_input_to_cl_image   = desc.export_input_to_cl_image;
        _export_output_to_cl_image  = desc.export_output_to_cl_image;

        // Update the padding for the weights tensor if we can export to cl_image
        if(_export_weights_to_cl_image)
        {
            gemm::update_padding_for_cl_image(weights);
        }

        if(_export_output_to_cl_image)
        {
            gemm::update_padding_for_cl_image(dst);
        }

        if(_export_input_to_cl_image)
        {
            gemm::update_padding_for_cl_image(src);
        }

        if(biases != nullptr)
        {
            build_options.add_option(std::string("-DHAS_BIAS"));
            build_options.add_option(std::string("-DBIA_DATA_TYPE=" + get_cl_type_from_data_type(biases->data_type())));
        }

        // Conditions of -cl-fast-relaxed-math causing accuracy issues can be traced from COMPMID-5324
        const auto act_function  = act_info.activation();
        const auto dst_data_type = dst->data_type();

        if((gpu_target != GPUTarget::G71 && (gpu_target & GPUTarget::GPU_ARCH_MASK) == GPUTarget::BIFROST)
           && (act_function == ActivationLayerInfo::ActivationFunction::BOUNDED_RELU || act_function == ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU)
           && (dst_data_type == DataType::F32 || dst_data_type == DataType::F16))
        {
            // -cl-fast-relaxed-math also sets -cl-finite-math-only and -cl-unsafe-math-optimizations
            // to disable -cl-finite-math-only, we only include -cl-unsafe-math-optimizations
            build_options.add_option("-cl-unsafe-math-optimizations");
        }
        else
        {
            build_options.add_option("-cl-fast-relaxed-math");
        }

        build_options.add_option_if_else(_export_input_to_cl_image, "-DSRC_TENSOR_TYPE=IMAGE", "-DSRC_TENSOR_TYPE=BUFFER");
        build_options.add_option("-DSRC_DATA_TYPE=" + get_cl_type_from_data_type(src->data_type()));
        build_options.add_option("-DSRC_CHANNELS=" + support::cpp11::to_string(src->dimension(0)));
        build_options.add_option("-DSRC_WIDTH=" + support::cpp11::to_string(src->dimension(1)));
        build_options.add_option("-DSRC_HEIGHT=" + support::cpp11::to_string(src->dimension(2)));
        build_options.add_option("-DDST_CHANNELS=" + support::cpp11::to_string(dst->dimension(0)));
        build_options.add_option("-DDST_WIDTH=" + support::cpp11::to_string(dst->dimension(1)));
        build_options.add_option("-DDST_HEIGHT=" + support::cpp11::to_string(dst->dimension(2)));
        build_options.add_option_if_else(_export_output_to_cl_image, "-DDST_TENSOR_TYPE=IMAGE", "-DDST_TENSOR_TYPE=BUFFER");
        build_options.add_option("-DDST_DATA_TYPE=" + get_cl_type_from_data_type(dst_data_type));
        build_options.add_option_if_else(_export_weights_to_cl_image, "-DWEI_TENSOR_TYPE=IMAGE", "-DWEI_TENSOR_TYPE=BUFFER");
        build_options.add_option("-DWEI_WIDTH=" + support::cpp11::to_string(weights->dimension(width_idx)));
        build_options.add_option("-DWEI_HEIGHT=" + support::cpp11::to_string(weights->dimension(height_idx)));
        build_options.add_option("-DWEI_DATA_TYPE=" + get_cl_type_from_data_type(weights->data_type()));
        build_options.add_option("-DSTRIDE_X=" + support::cpp11::to_string(conv_stride_x));
        build_options.add_option("-DSTRIDE_Y=" + support::cpp11::to_string(conv_stride_y));
        build_options.add_option("-DPAD_LEFT=" + support::cpp11::to_string(pad_left));
        build_options.add_option("-DPAD_TOP=" + support::cpp11::to_string(pad_top));
        build_options.add_option("-DN0=" + support::cpp11::to_string(n0));
        build_options.add_option("-DM0=" + support::cpp11::to_string(m0));
        build_options.add_option("-DK0=" + support::cpp11::to_string(k0));
        build_options.add_option("-DPARTIAL_N0=" + support::cpp11::to_string(partial_store_n0));
        build_options.add_option_if((src->dimension(channel_idx) % k0) != 0, "-DLEFTOVER_LOOP");
        build_options.add_option("-DACTIVATION_TYPE=" + lower_string(string_from_activation_func(act_function)));

        if(is_data_type_quantized(data_type))
        {
            const UniformQuantizationInfo iqinfo = src->quantization_info().uniform();
            const UniformQuantizationInfo wqinfo = weights->quantization_info().uniform();
            const UniformQuantizationInfo oqinfo = dst->quantization_info().uniform();

            PixelValue zero_value = PixelValue(0, src->data_type(), src->quantization_info());
            int        zero_value_s32;
            zero_value.get(zero_value_s32);

            float multiplier        = iqinfo.scale * wqinfo.scale / oqinfo.scale;
            int   output_multiplier = 0;
            int   output_shift      = 0;
            quantization::calculate_quantized_multiplier(multiplier, &output_multiplier, &output_shift);
            build_options.add_option("-DIS_QUANTIZED");
            build_options.add_option("-DDST_MULTIPLIER=" + support::cpp11::to_string(output_multiplier));
            build_options.add_option("-DDST_SHIFT=" + support::cpp11::to_string(output_shift));
            build_options.add_option("-DSRC_OFFSET=" + support::cpp11::to_string(-iqinfo.offset));
            build_options.add_option("-DWEI_OFFSET=" + support::cpp11::to_string(-wqinfo.offset));
            build_options.add_option("-DDST_OFFSET=" + support::cpp11::to_string(oqinfo.offset));
            build_options.add_option("-DZERO_VALUE=" + support::cpp11::to_string(zero_value_s32));
            build_options.add_option("-DACC_DATA_TYPE=" + get_cl_type_from_data_type(DataType::S32));
        }
        else
        {
            build_options.add_option("-DACC_DATA_TYPE=" + get_cl_type_from_data_type(data_type));
            build_options.add_option("-DZERO_VALUE=" + support::cpp11::to_string(0));
            build_options.add_option("-DSRC_OFFSET=" + support::cpp11::to_string(0));
            build_options.add_option("-DWEI_OFFSET=" + support::cpp11::to_string(0));
            build_options.add_option("-DDST_OFFSET=" + support::cpp11::to_string(0));
            build_options.add_option_if(act_info.enabled(), "-DA_VAL=" + float_to_string_with_full_precision(act_info.a()));
            build_options.add_option_if(act_info.enabled(), "-DB_VAL=" + float_to_string_with_full_precision(act_info.b()));
        }

        if(compile_context.get_ddk_version() >= 30)
        {
            build_options.add_option("-fregister-allocation=64");
        }
    }
    else
    {
        _export_weights_to_cl_image = false;

        kernel_name << "direct_convolution_nchw";
        build_options.add_option_if(biases != nullptr, std::string("-DHAS_BIAS"));
        build_options.add_option("-DSRC_WIDTH=" + support::cpp11::to_string(src->dimension(width_idx)));
        build_options.add_option("-DSRC_HEIGHT=" + support::cpp11::to_string(src->dimension(height_idx)));
        build_options.add_option("-DSRC_CHANNELS=" + support::cpp11::to_string(src->dimension(channel_idx)));
        build_options.add_option("-DPAD_LEFT=" + support::cpp11::to_string(conv_info.pad_left()));
        build_options.add_option("-DPAD_TOP=" + support::cpp11::to_string(conv_info.pad_top()));
        build_options.add_option("-DSTRIDE_X=" + support::cpp11::to_string(conv_stride_x));
        build_options.add_option("-DSTRIDE_Y=" + support::cpp11::to_string(conv_stride_y));
        build_options.add_option("-DWEI_WIDTH=" + support::cpp11::to_string(weights->dimension(width_idx)));
        build_options.add_option("-DWEI_HEIGHT=" + support::cpp11::to_string(weights->dimension(height_idx)));
        build_options.add_option(std::string("-DDATA_TYPE=" + get_cl_type_from_data_type(data_type)));
        build_options.add_option(std::string("-DDATA_SIZE=" + get_data_size_from_data_type(data_type)));
        build_options.add_option(std::string("-DWEIGHTS_DEPTH=" + support::cpp11::to_string(weights->dimension(channel_idx))));
        build_options.add_option(std::string("-DSTRIDE_X=" + support::cpp11::to_string(conv_stride_x)));
        build_options.add_option(std::string("-DDATA_TYPE_PROMOTED=" + get_cl_type_from_data_type(data_type)));
        build_options.add_option(std::string("-DVEC_SIZE=" + support::cpp11::to_string(_num_elems_processed_per_iteration)));
        build_options.add_option(std::string("-DVEC_SIZE_LEFTOVER=" + support::cpp11::to_string(src->dimension(0) % _num_elems_processed_per_iteration)));

        if(is_data_type_quantized(data_type))
        {
            const UniformQuantizationInfo iqinfo = src->quantization_info().uniform();
            const UniformQuantizationInfo wqinfo = weights->quantization_info().uniform();
            const UniformQuantizationInfo oqinfo = dst->quantization_info().uniform();

            float multiplier        = iqinfo.scale * wqinfo.scale / oqinfo.scale;
            int   output_multiplier = 0;
            int   output_shift      = 0;
            quantization::calculate_quantized_multiplier(multiplier, &output_multiplier, &output_shift);
            build_options.add_option("-DIS_QUANTIZED");
            build_options.add_option("-DOUTPUT_MULTIPLIER=" + support::cpp11::to_string(output_multiplier));
            build_options.add_option("-DOUTPUT_SHIFT=" + support::cpp11::to_string(output_shift));
            build_options.add_option("-DKERNEL_SIZE=" + support::cpp11::to_string(kernel_size));
            build_options.add_option("-DINPUT_OFFSET=" + support::cpp11::to_string(-iqinfo.offset));
            build_options.add_option("-DWEIGHTS_OFFSET=" + support::cpp11::to_string(-wqinfo.offset));
            build_options.add_option("-DOUTPUT_OFFSET=" + support::cpp11::to_string(oqinfo.offset));
        }
    }

    _kernel = create_kernel(compile_context, kernel_name.str(), build_options.options());

    // Set config_id for enabling LWS tuning
    // config_id should include the variables used to parameterize the kernel
    _config_id = kernel_name.str();
    _config_id += "_";
    _config_id += lower_string(string_from_data_type(data_type));
    _config_id += "_";
    _config_id += support::cpp11::to_string(kernel_size);
    _config_id += "_";
    _config_id += support::cpp11::to_string(border_size().left);
    _config_id += "_";
    _config_id += support::cpp11::to_string(border_size().top);
    _config_id += "_";
    _config_id += support::cpp11::to_string(border_size().right);
    _config_id += "_";
    _config_id += support::cpp11::to_string(border_size().bottom);
    _config_id += "_";
    _config_id += support::cpp11::to_string(conv_stride_x);
    _config_id += "_";
    _config_id += support::cpp11::to_string(conv_stride_y);
    // SRC_CHANNELS, SRC_WIDTH, SRC_HEIGHT
    _config_id += "_";
    _config_id += support::cpp11::to_string(src->dimension(channel_idx));
    _config_id += "_";
    _config_id += support::cpp11::to_string(src->dimension(width_idx));
    _config_id += "_";
    _config_id += support::cpp11::to_string(src->dimension(height_idx));
    _config_id += "_";
    // DST_CHANNELS, DST_WIDTH, DST_HEIGHT
    _config_id += support::cpp11::to_string(dst->dimension(channel_idx));
    _config_id += "_";
    _config_id += support::cpp11::to_string(dst->dimension(width_idx));
    _config_id += "_";
    _config_id += support::cpp11::to_string(dst->dimension(height_idx));
    _config_id += "_";
    _config_id += lower_string(string_from_data_layout(_data_layout));
}

Status ClDirectConv2dKernel::validate(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *dst,
                                      const PadStrideInfo &conv_info, const ActivationLayerInfo &act_info, const DirectConvComputeKernelInfo &desc)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, weights, biases, dst, conv_info, act_info, desc));
    return Status{};
}

void ClDirectConv2dKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    // Get initial windows
    Window slice = window.first_slice_window_3D();

    const auto src     = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_0));
    const auto weights = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_1));
    const auto biases  = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_2));
    auto       dst     = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));

    if(_data_layout == DataLayout::NHWC)
    {
        cl::Image2D weights_cl_image;
        cl::Image2D output_cl_image;
        cl::Image2D input_cl_image;

        if(_export_weights_to_cl_image)
        {
            // Export tensor to cl_image
            weights_cl_image = create_image2d_from_tensor(weights, CLImage2DType::ReadOnly);
        }

        if(_export_output_to_cl_image)
        {
            // Export tensor to cl_image
            output_cl_image = create_image2d_from_tensor(dst, CLImage2DType::WriteOnly);
        }

        if(_export_input_to_cl_image)
        {
            // Export tensor to cl_image
            input_cl_image = create_image2d_from_tensor(src, CLImage2DType::ReadOnly);
        }

        unsigned int idx = 0;
        if(_export_input_to_cl_image)
        {
            _kernel.setArg(idx++, input_cl_image);
        }
        add_4d_tensor_nhwc_argument(idx, src);
        if(_export_output_to_cl_image)
        {
            _kernel.setArg(idx++, output_cl_image);
        }
        add_4d_tensor_nhwc_argument(idx, dst);
        if(_export_weights_to_cl_image)
        {
            _kernel.setArg(idx++, weights_cl_image);
        }
        add_4d_tensor_nhwc_argument(idx, weights);
        if(biases != nullptr)
        {
            add_1D_tensor_argument(idx, biases, slice);
        }
        enqueue(queue, *this, slice, lws_hint());
    }
    else
    {
        unsigned int idx1 = 2 * num_arguments_per_3D_tensor();
        add_3D_tensor_argument(idx1, weights, slice);

        if(biases != nullptr)
        {
            Window slice_biases;
            slice_biases.use_tensor_dimensions(biases->info()->tensor_shape());
            add_1D_tensor_argument(idx1, biases, slice_biases);
        }

        _kernel.setArg(idx1++, static_cast<unsigned int>(weights->info()->strides_in_bytes()[3]));

        do
        {
            unsigned int idx = 0;
            add_3D_tensor_argument(idx, src, slice);
            add_3D_tensor_argument(idx, dst, slice);
            enqueue(queue, *this, slice, lws_hint());
        }
        while(window.slide_window_slice_3D(slice));
    }
}
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
