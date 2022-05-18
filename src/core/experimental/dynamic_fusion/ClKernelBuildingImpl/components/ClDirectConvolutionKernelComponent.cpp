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
#ifdef ENABLE_EXPERIMENTAL_DYNAMIC_FUSION

#include "src/core/experimental/dynamic_fusion/ClKernelBuildingImpl/components/ClDirectConvolutionKernelComponent.h"

#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "src/core/CL/ICLKernel.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/gpu/cl/kernels/gemm/ClGemmHelpers.h"

#include "arm_compute/runtime/CL/CLScheduler.h"
namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
ComponentType ClDirectConvolutionKernelComponent::get_component_type() const
{
    return ComponentType::Complex;
}

std::set<std::string> ClDirectConvolutionKernelComponent::get_headers_list() const
{
    return std::set<std::string> { "helpers.h", "tile_helpers.h" };
}

Window ClDirectConvolutionKernelComponent::get_window() const
{
    const auto src_info    = _blueprint->impl().get_kernel_argument_info(_src.arg_id);
    const auto weight_info = _blueprint->impl().get_kernel_argument_info(_weight.arg_id);
    auto       dst_info    = _blueprint->impl().get_kernel_argument_info(_blueprint->impl().get_dst_id());

    // Get dst shape
    PadStrideInfo pad_stride_info
    {
        static_cast<unsigned int>(_desc.conv2d.stride.x()),
        static_cast<unsigned int>(_desc.conv2d.stride.y()),
        static_cast<unsigned int>(_desc.conv2d.pad.left),
        static_cast<unsigned int>(_desc.conv2d.pad.right),
        static_cast<unsigned int>(_desc.conv2d.pad.top),
        static_cast<unsigned int>(_desc.conv2d.pad.bottom),
        DimensionRoundingType::FLOOR /*default rounding type*/
    };
    TensorShape output_shape = misc::shape_calculator::compute_deep_convolution_shape(*src_info, *weight_info, pad_stride_info);

    // Output auto initialization if not yet initialized
    auto_init_if_empty(*dst_info, output_shape,
                       1,
                       src_info->data_type(),
                       src_info->quantization_info());

    const unsigned int vec_size = std::min(static_cast<unsigned int>(dst_info->tensor_shape()[0]), 4u);
    const unsigned int num_rows = (dst_info->tensor_shape()[0] > 16) ? ((src_info->data_type() == DataType::F32) ? 2U : 4U) : 1U;
    // const unsigned int num_rows = 1;
    // const unsigned int vec_size = tile_info.tile_dims.x();
    // const unsigned int num_rows = tile_info.tile_dims.y();

    // Create and configure kernel window
    Window win = calculate_max_window(output_shape, Steps(vec_size, num_rows));

    const size_t dim_y_collapsed = ceil_to_multiple(output_shape[1] * output_shape[2], num_rows);
    win.set(Window::DimY, Window::Dimension(0, dim_y_collapsed, num_rows));
    win.set(Window::DimZ, Window::Dimension(0, output_shape.total_size_upper(3), 1));

    return win;
}

std::string ClDirectConvolutionKernelComponent::get_additional_macros() const
{
    return R"_()_"; // no macros
}

std::string ClDirectConvolutionKernelComponent::get_component_code() const
{
    const auto src_info  = _blueprint->impl().get_kernel_argument_info(_src.arg_id);
    const auto bias_info = _blueprint->impl().get_kernel_argument_info(_bias.arg_id);

    ARM_COMPUTE_ERROR_ON_MSG(src_info->data_layout() != DataLayout::NHWC, "Only NHWC data layout is supported by this component.");

    const auto channel_idx   = get_data_layout_dimension_index(src_info->data_layout(), DataLayoutDimension::CHANNEL);
    const auto k0            = adjust_vec_size(is_data_type_quantized(src_info->data_type()) ? 16u : 8u, src_info->dimension(channel_idx));
    const bool leftover_loop = (src_info->dimension(channel_idx) % k0) != 0;

    std::string code = R"_(
    //------------------ START KERNEL {{meta_kernel_id}} ---------------------
    // IN_0(src)            {{src}}
    // IN_1(wei)            {{weight}}
    )_";
    if(bias_info != nullptr)
    {
        code += R"_(
    // IN_1(bia)            {{bias}}
    )_";
    }
    code += R"_(
    // OUT(dst, accum)      {{dst}}

    // Initialize the accumulators
    TILE({{ACC_DATA_TYPE}}, M0, N0, {{dst}});
    {
        // All the tensor dimensions are passed at compile time.
        // In case of dynamic tensor support, the following dimensions should be passed as function argument.
    #define _IWEI_WIDTH {{WEI_WIDTH}}
    #define _IWEI_HEIGHT {{WEI_HEIGHT}}
    #define _ISRC_WIDTH {{src}}_w
    #define _ISRC_HEIGHT {{src}}_h
    #define _ISRC_CHANNELS {{src}}_c
    #define _IDST_WIDTH {{arg_dst}}_w
    #define _IDST_HEIGHT {{arg_dst}}_h
    #define _IDST_CHANNELS {{arg_dst}}_c
    #define _IY_MULTIPLIER (_IWEI_WIDTH * _IWEI_HEIGHT)

        // .v    = access the whole vector (OpenCL vector)
        // .s[x] = access the vector element at position x (scalar access)
        TILE(int, M0, 1, xi);
        TILE(int, M0, 1, yi);

        // Convert the linear index to coordinate
        LOOP_UNROLLING(int, i, 0, 1, M0,
        {
            xi[i].v = ((mout + i) % _IDST_WIDTH) * {{STRIDE_X}};
            yi[i].v = ((mout + i) / _IDST_WIDTH) * {{STRIDE_Y}};
            xi[i].v -= {{PAD_LEFT}};
            yi[i].v -= {{PAD_TOP}};
        })

        LOOP_UNROLLING(int, i, 0, 1, M0,
        {
            {{dst}}[i].v = 0;
        })

        for(int i = 0; i < (_IWEI_WIDTH * _IWEI_HEIGHT); ++i)
        {
            int ck = 0;
            int xk = i % _IWEI_WIDTH;
            int yk = i / _IWEI_HEIGHT;

            int k = 0;
            for(; k <= (_ISRC_CHANNELS - K0); k += K0)
            {
                TILE({{SRC_DATA_TYPE}}, M0, K0, a);
                TILE({{WEI_DATA_TYPE}}, N0, K0, b);

                LOOP_UNROLLING(int, i, 0, 1, M0,
                {
                    a[i].v = {{ZERO_VALUE}};
                })

                // Load tile from the src tensor
                T_LOAD_NHWC_INDIRECT({{SRC_DATA_TYPE}}, M0, K0, {{SRC_TENSOR_TYPE}}, {{src}}, bout, yk, xk, ck, _ISRC_WIDTH, _ISRC_HEIGHT, {{src}}_stride_y, xi, yi, a);

                // Load tile from the weights tensor
                T_LOAD({{WEI_DATA_TYPE}}, N0, K0, {{WEI_TENSOR_TYPE}}, {{weight}}, ck, cout * _IY_MULTIPLIER + i, _IY_MULTIPLIER, {{weight}}_stride_y, b);

                // Compute the matrix multiplication between two tiles
                T_MMUL({{SRC_DATA_TYPE}}, {{WEI_DATA_TYPE}}, {{ACC_DATA_TYPE}}, M0, N0, K0, NT, T, a, b, {{dst}});

                ck += K0;
            }

            // We voluntarily use SRC_CHANNELS rather than _DSRC_CHANNELS
            // This #if directive should be removed in case of dynamic tensor support
    )_";

    if(leftover_loop)
    {
        code += R"_(
            // Left-over accumulations
            for(; k < _ISRC_CHANNELS; ++k)
            {
                TILE({{SRC_DATA_TYPE}}, M0, 1, a);
                TILE({{WEI_DATA_TYPE}}, N0, 1, b);

                LOOP_UNROLLING(int, i, 0, 1, M0,
                {
                    a[i].v = {{ZERO_VALUE}};
                })

                // Load tile from the src tensor
                T_LOAD_NHWC_INDIRECT({{SRC_DATA_TYPE}}, M0, 1, {{SRC_TENSOR_TYPE}}, {{src}}, bout, yk, xk, ck, _ISRC_WIDTH, _ISRC_HEIGHT, {{src}}_stride_y, xi, yi, a);

                // Load tile from the weights tensor
                // The T_LOAD for the left-over elements can only use BUFFER because we load one element per iteration
                T_LOAD({{WEI_DATA_TYPE}}, N0, 1, BUFFER, {{weight}}, ck, cout * _IY_MULTIPLIER + i, _IY_MULTIPLIER, {{weight}}_stride_y, b);

                // Compute the matrix multiplication between two tiles
                T_MMUL({{SRC_DATA_TYPE}}, {{WEI_DATA_TYPE}}, {{ACC_DATA_TYPE}}, M0, N0, 1, NT, T, a, b, {{dst}});

                ++ck;
            }
        )_";
    }

    code += R"_(
    #undef _I_WEI_WIDTH
    #undef _I_WEI_HEIGHT
    #undef _ISRC_WIDTH
    #undef _ISRC_HEIGHT
    #undef _ISRC_CHANNELS
    #undef _IDST_WIDTH
    #undef _IDST_HEIGHT
    #undef _IDST_CHANNELS
    #undef _IY_MULTIPLIER

        }
    )_";

    if(bias_info != nullptr)
    {
        code += R"_(
            TILE({{BIA_DATA_TYPE}}, 1, N0, bias0);

            T_LOAD({{BIA_DATA_TYPE}}, 1, N0, BUFFER, {{bias}}, cout, 0, 1, 0, bias0);

            // c = c + bias[broadcasted]
            T_ADD_BROADCAST_X({{ACC_DATA_TYPE}}, M0, N0, {{dst}}, bias0, {{dst}});
        )_";
    }

    code += R"_(
    }
//------------------ END KERNEL {{meta_kernel_id}} ---------------------
    )_";
    return code.c_str();
}

bool export_to_cl_image_support(const ITensorInfo *tensor, GPUTarget gpu_target, DataLayout data_layout)
{
    if(tensor->tensor_shape()[0] % 4 || (data_layout != DataLayout::NHWC))
    {
        return false;
    }

    // If not floating point
    if(!is_data_type_float(tensor->data_type()))
    {
        return false;
    }

    if(gpu_target == GPUTarget::G71 || get_arch_from_target(gpu_target) == GPUTarget::MIDGARD)
    {
        return false;
    }

    // Check if the cl_khr_image2d_from_buffer extension is supported on the target platform
    if(!image2d_from_buffer_supported(CLKernelLibrary::get().get_device()))
    {
        return false;
    }

    // Check cl image pitch alignment
    if(get_cl_image_pitch_alignment(CLKernelLibrary::get().get_device()) == 0)
    {
        return false;
    }

    const size_t image_w     = tensor->tensor_shape()[0] / 4;
    const size_t image_h     = tensor->tensor_shape()[1] * tensor->tensor_shape()[2] * tensor->tensor_shape()[3];
    const size_t max_image_w = CLKernelLibrary::get().get_device().getInfo<CL_DEVICE_IMAGE2D_MAX_WIDTH>();
    const size_t max_image_h = CLKernelLibrary::get().get_device().getInfo<CL_DEVICE_IMAGE2D_MAX_HEIGHT>();

    if(image_w > max_image_w || image_h > max_image_h)
    {
        return false;
    }

    return true;
}

CLBuildOptions ClDirectConvolutionKernelComponent::generate_build_options() const
{
    const auto src_info    = _blueprint->impl().get_kernel_argument_info(_src.arg_id);
    auto       weight_info = _blueprint->impl().get_kernel_argument_info(_weight.arg_id);
    const auto dst_info    = _blueprint->impl().get_kernel_argument_info(_blueprint->impl().get_dst_id());
    // const auto tile_info  = _blueprint->impl().get_tile_info();

    const unsigned int channel_idx = get_data_layout_dimension_index(src_info->data_layout(), DataLayoutDimension::CHANNEL);
    const DataType     data_type   = src_info->data_type();
    const GPUTarget    gpu_target  = CLScheduler::get().target();

    const unsigned int n0                 = _blueprint->impl().get_execution_window().x().step();
    const unsigned int m0                 = _blueprint->impl().get_execution_window().y().step();
    const unsigned int k0                 = adjust_vec_size(is_data_type_quantized(data_type) ? 16u : 8u, src_info->dimension(channel_idx));
    const unsigned int partial_store_n0   = dst_info->dimension(0) % n0;
    const bool         export_to_cl_image = export_to_cl_image_support(weight_info, gpu_target, src_info->data_layout());

    // Update the padding for the weights tensor if we can export to cl_image
    if(export_to_cl_image)
    {
        arm_compute::opencl::kernels::gemm::update_padding_for_cl_image(weight_info);
    }

    CLBuildOptions build_opts{};
    build_opts.add_option("-cl-fast-relaxed-math");
    build_opts.add_option("-DIS_TILED");
    build_opts.add_option("-DN0=" + support::cpp11::to_string(n0));
    build_opts.add_option("-DM0=" + support::cpp11::to_string(m0));
    build_opts.add_option("-DK0=" + support::cpp11::to_string(k0));
    build_opts.add_option("-DPARTIAL_N0=" + support::cpp11::to_string(partial_store_n0));

    return build_opts;
}

void ClDirectConvolutionKernelComponent::allocate_shared_vars(SharedVarTable &vtable) const
{
    const auto src_info    = _blueprint->impl().get_kernel_argument_info(_src.arg_id);
    const auto weight_info = _blueprint->impl().get_kernel_argument_info(_weight.arg_id);

    vtable.add(_src, _blueprint->impl().group(_src.arg_id), ClKernelArgDescriptor(_src.arg_id, ClKernelTensorArgType::Tensor_4D_t_Buffer), "src");

    const GPUTarget             gpu_target         = CLScheduler::get().target();
    const bool                  export_to_cl_image = export_to_cl_image_support(weight_info, gpu_target, src_info->data_layout());
    const ClKernelTensorArgType weight_type        = export_to_cl_image ? ClKernelTensorArgType::Tensor_4D_t_Image : ClKernelTensorArgType::Tensor_4D_t_Buffer;
    vtable.add(_weight, _blueprint->impl().group(_weight.arg_id), ClKernelArgDescriptor(_weight.arg_id, weight_type), "weight");

    if(!_bias.is_empty()) // optional bias
    {
        vtable.add(_bias, _blueprint->impl().group(_bias.arg_id), ClKernelArgDescriptor(_bias.arg_id, ClKernelTensorArgType::Vector), "bias");
    }
    vtable.add(_dst, _blueprint->impl().group(_dst.arg_id), ClKernelArgDescriptor(_dst.arg_id, ClKernelTensorArgType::Tensor_4D_t_Buffer), "dst");
}

ClDirectConvolutionKernelComponent::TagLUT ClDirectConvolutionKernelComponent::get_tag_lut(const SharedVarTable &vtable) const
{
    TagLUT lut{};

    const auto src_info    = _blueprint->impl().get_kernel_argument_info(_src.arg_id);
    const auto weight_info = _blueprint->impl().get_kernel_argument_info(_weight.arg_id);
    const auto bias_info   = _blueprint->impl().get_kernel_argument_info(_bias.arg_id);

    // Arguments and global shared variables
    lut["src"]    = vtable.get(_src);
    lut["weight"] = vtable.get(_weight);

    if(!_bias.is_empty()) // optional bias
    {
        lut["bias"]          = vtable.get(_bias);
        lut["BIA_DATA_TYPE"] = get_cl_type_from_data_type(bias_info->data_type());
    }
    lut["dst"] = vtable.get(_dst);

    const auto dst_argument = _blueprint->impl().get_argument_shared_vars().get_dst_var();
    lut["arg_dst"]          = dst_argument.uniq_name;

    // Local build options
    lut["meta_kernel_id"] = id();
    lut["ACC_DATA_TYPE"]  = src_info->data_type();
    lut["SRC_DATA_TYPE"]  = src_info->data_type();
    lut["WEI_DATA_TYPE"]  = weight_info->data_type();

    lut["SRC_TENSOR_TYPE"] = "BUFFER";
    switch(vtable.get(_weight).desc.tensor_arg_type)
    {
        case ClKernelTensorArgType::Image_Export_To_ClImage2D:
        case ClKernelTensorArgType::Image_3D_Export_To_ClImage2D:
        case ClKernelTensorArgType::Tensor_4D_t_Image:
        {
            lut["WEI_TENSOR_TYPE"] = "IMAGE";
            break;
        }
        default:
        {
            lut["WEI_TENSOR_TYPE"] = "BUFFER";
            break;
        }
    }
    const auto width_idx  = get_data_layout_dimension_index(src_info->data_layout(), DataLayoutDimension::WIDTH);
    const auto height_idx = get_data_layout_dimension_index(src_info->data_layout(), DataLayoutDimension::HEIGHT);
    lut["WEI_WIDTH"]      = weight_info->dimension(width_idx);
    lut["WEI_HEIGHT"]     = weight_info->dimension(height_idx);

    lut["STRIDE_X"] = _desc.conv2d.stride.x();
    lut["STRIDE_Y"] = _desc.conv2d.stride.y();

    lut["PAD_LEFT"] = _desc.conv2d.pad.left;
    lut["PAD_TOP"]  = _desc.conv2d.pad.top;

    lut["ZERO_VALUE"] = 0;

    return lut;
}
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif /* ENABLE_EXPERIMENTAL_DYNAMIC_FUSION */