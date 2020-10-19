/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#include "arm_compute/runtime/CL/tuners/BifrostTuner.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernels.h"
#include "support/Cast.h"

namespace arm_compute
{
namespace tuners
{
namespace
{
/** Tunes a @ref CLDirectConvolutionLayerKernel for a bifrost target
 *
 * @param[in] k Kernels to tune
 */
void tune_direct_convolution_kernel(CLDirectConvolutionLayerKernel &k)
{
    cl::NDRange lws_hint = k.lws_hint();

    const GPUTarget    gpu_target    = k.get_target();
    const DataType     dt            = k._input->info()->data_type();
    const TensorShape  weights_shape = k._weights->info()->tensor_shape();
    const TensorShape  inputs_shape  = k._input->info()->tensor_shape();
    const size_t       kernel_size   = weights_shape.x();
    const unsigned int stride_x      = k._conv_stride_x;
    const unsigned int stride_y      = k._conv_stride_y;

    if(gpu_target_is_in(gpu_target, GPUTarget::G71, GPUTarget::G72) && (kernel_size <= 5) && (stride_x == 1) && (stride_y == 1) && (dt == DataType::F32))
    {
        // Through extensive experimentation with over 30 representative tensor
        // shapes, we found a small number of local work size configurations
        // that result in nearly optimal execution times. Selecting the right
        // lws for a given shape, however, required a complex decision tree,
        // until we constructed a simple feature as described below.
        //
        // We started from the number of multiply-accumulate operations for a
        // convolution layer, which is equal to the product of the input
        // dimensions 0..2 and the weights dimensions 0..2.  Unfortunately,
        // this resulted in ties between distinct shapes that required distinct
        // lws configurations. Replacing the width of the input with the kernel
        // size, however, resulted in nearly optimal predictions. We use underscores
        // in variable names to indicate when they are intentionally misleading.
        const size_t product_of_weights_dimensions = weights_shape[0] * weights_shape[1] * weights_shape[2];
        const size_t product_of_input_dimensions_  = inputs_shape[0] * inputs_shape[1] * inputs_shape[2];
        const float  mega_ops_                     = 1e-6 * product_of_weights_dimensions * product_of_input_dimensions_;

        switch(kernel_size)
        {
            case 1:
            {
                if(mega_ops_ < 1.f)
                {
                    lws_hint = cl::NDRange(1, 1, 8);
                }
                else if(mega_ops_ < 7.f)
                {
                    lws_hint = cl::NDRange(1, 1, 4);
                }
                else
                {
                    lws_hint = cl::NDRange(1, 1, 2);
                }
                break;
            }
            case 3:
            {
                if(mega_ops_ < 1.f)
                {
                    lws_hint = cl::NDRange(1, 1, 8);
                }
                else if(mega_ops_ < 13.f)
                {
                    lws_hint = cl::NDRange(2, 1, 4);
                }
                else if(mega_ops_ < 50.f)
                {
                    lws_hint = cl::NDRange(3, 1, 4);
                }
                else
                {
                    lws_hint = cl::NDRange(2, 1, 6);
                }
                break;
            }
            case 5:
            {
                if(mega_ops_ < 2.f || mega_ops_ > 80.f)
                {
                    lws_hint = cl::NDRange(2, 1, 4);
                }
                else
                {
                    lws_hint = cl::NDRange(2, 1, 8);
                }
                break;
            }
            default:
                break;
        }
        k.set_lws_hint(lws_hint);
    }
}

void tune_col2im_kernel(CLCol2ImKernel &k)
{
    cl::NDRange     lws_hint   = k.lws_hint();
    const GPUTarget gpu_target = k.get_target();

    // Configure the local work size for Bifrost with a value obtained
    // via exhaustive autotuning over 30 representative tensor shapes.
    if(gpu_target_is_in(gpu_target,
                        GPUTarget::G71, GPUTarget::G72, GPUTarget::G76,
                        GPUTarget::G51, GPUTarget::G51BIG, GPUTarget::G51LIT,
                        GPUTarget::G52, GPUTarget::G52LIT))
    {
        if((k._convolved_dims.width == 7) || (k._convolved_dims.width == 14))
        {
            lws_hint = cl::NDRange(1, 7, 1);
        }
        else
        {
            lws_hint = cl::NDRange(1, 8, 1);
        }
    }

    k.set_lws_hint(lws_hint);
}

void tune_im2col_kernel(CLIm2ColKernel &k)
{
    cl::NDRange     lws_hint   = k.lws_hint();
    const GPUTarget gpu_target = k.get_target();

    // Local work size optimized for the 11x11 AlexNet convolution on Bifrost.
    if(gpu_target_is_in(gpu_target,
                        GPUTarget::G71, GPUTarget::G72, GPUTarget::G76,
                        GPUTarget::G51, GPUTarget::G51BIG, GPUTarget::G51LIT,
                        GPUTarget::G52, GPUTarget::G52LIT)
       && k._kernel_dims.width == 11)
    {
        const bool is_square_kernel = (k._kernel_dims.width == k._kernel_dims.height);
        if(!is_square_kernel && k._kernel_dims.width > 1 && !k._conv_info.has_padding())
        {
            lws_hint = cl::NDRange(1, 1, 1);
        }
    }
    k.set_lws_hint(lws_hint);
}

void tune_gemv_kernel(CLGEMMMatrixVectorMultiplyKernel &k)
{
    cl::NDRange     lws_hint   = k.lws_hint();
    const GPUTarget gpu_target = k.get_target();

    // Configure the local work size for Bifrost with a value obtained
    // via exhaustive autotuning for the MobileNets tensor shapes.
    if(gpu_target_is_in(gpu_target,
                        GPUTarget::G71, GPUTarget::G72, GPUTarget::G76,
                        GPUTarget::G51, GPUTarget::G51BIG, GPUTarget::G51LIT,
                        GPUTarget::G52, GPUTarget::G52LIT))
    {
        lws_hint = cl::NDRange(1, 1, 1);
    }

    k.set_lws_hint(lws_hint);
}

void tune_gemm_kernel(CLGEMMMatrixMultiplyKernel &k)
{
    cl::NDRange     lws_hint   = k.lws_hint();
    const GPUTarget gpu_target = k.get_target();

    // Configure LWS hint
    switch(gpu_target)
    {
        case GPUTarget::G71:
        case GPUTarget::G72:
        case GPUTarget::G51:
        case GPUTarget::G51BIG:
        case GPUTarget::G51LIT:
        case GPUTarget::G52:
        case GPUTarget::G52LIT:
        case GPUTarget::G76:
            if(k._input1->info()->dimension(1) == 24)
            {
                // LWS optimized for the 11x11 AlexNet convolution on Bifrost.
                lws_hint = cl::NDRange(2, 2);
            }
            else if(k._output->info()->dimension(1) == 196)
            {
                lws_hint = cl::NDRange(1, 7);
            }
            else
            {
                lws_hint = cl::NDRange(8, 8);
            }
            break;
        default:
            lws_hint = cl::NullRange;
    }

    k.set_lws_hint(lws_hint);
}

void tune_pooling_kernel(CLPoolingLayerKernel &k)
{
    cl::NDRange     lws_hint   = k.lws_hint();
    const GPUTarget gpu_target = k.get_target();

    // Configure the local work size (hint) from the first two dimensions of the global work size.
    // On Bifrost, this works for up to 35x35xC filters, for which the pooling_layer_3_optimized
    // kernel is launched with gws=(9, 33, C). In any case, the hint will be ignored if it is
    // invalid (e.g. exceeds the maximum workgroup size that the kernel can be launched with).
    if(k._input->info()->data_layout() == DataLayout::NCHW)
    {
        if(gpu_target_is_in(gpu_target,
                            GPUTarget::G71, GPUTarget::G72, GPUTarget::G76,
                            GPUTarget::G51, GPUTarget::G51BIG, GPUTarget::G51LIT,
                            GPUTarget::G52, GPUTarget::G52LIT))
        {
            cl::NDRange gws = ICLKernel::gws_from_window(k.window());
            lws_hint        = cl::NDRange(gws[0], gws[1], 1);
        }
    }

    k.set_lws_hint(lws_hint);
}

void tune_scale_kernel(CLScaleKernel &k)
{
    cl::NDRange               lws_hint      = k.lws_hint();
    const GPUTarget           gpu_target    = k.get_target();
    const DataType            dt            = k.input()->info()->data_type();
    const InterpolationPolicy interpolation = k.get_interpolation_policy();

    // Configure the local work size for Bifrost, interpolation (bilinear) and datatype F32.
    // The value are obtained via exhaustive autotuning.
    if(gpu_target_is_in(gpu_target, GPUTarget::G71, GPUTarget::G72) && (dt == DataType::F32) && (interpolation == InterpolationPolicy::BILINEAR))
    {
        auto dim_0 = k.output()->info()->dimension(0);
        if(dim_0 == 480)
        {
            lws_hint = cl::NDRange(2, 1);
        }
        else if(dim_0 == 3120)
        {
            lws_hint = cl::NDRange(2, 8);
        }
        else if(dim_0 == 4160)
        {
            lws_hint = cl::NDRange(4, 8);
        }
        k.set_lws_hint(lws_hint);
    }
}
} // namespace

void BifrostTuner::tune_kernel_static(ICLKernel &kernel)
{
    if(dynamic_cast<CLDirectConvolutionLayerKernel *>(&kernel) != nullptr)
    {
        tune_direct_convolution_kernel(*utils::cast::polymorphic_downcast<CLDirectConvolutionLayerKernel *>(&kernel));
    }
    else if(dynamic_cast<CLCol2ImKernel *>(&kernel) != nullptr)
    {
        tune_col2im_kernel(*utils::cast::polymorphic_downcast<CLCol2ImKernel *>(&kernel));
    }
    else if(dynamic_cast<CLIm2ColKernel *>(&kernel) != nullptr)
    {
        tune_im2col_kernel(*utils::cast::polymorphic_downcast<CLIm2ColKernel *>(&kernel));
    }
    else if(dynamic_cast<CLGEMMMatrixVectorMultiplyKernel *>(&kernel) != nullptr)
    {
        tune_gemv_kernel(*utils::cast::polymorphic_downcast<CLGEMMMatrixVectorMultiplyKernel *>(&kernel));
    }
    else if(dynamic_cast<CLGEMMMatrixMultiplyKernel *>(&kernel) != nullptr)
    {
        tune_gemm_kernel(*utils::cast::polymorphic_downcast<CLGEMMMatrixMultiplyKernel *>(&kernel));
    }
    else if(dynamic_cast<CLPoolingLayerKernel *>(&kernel) != nullptr)
    {
        tune_pooling_kernel(*utils::cast::polymorphic_downcast<CLPoolingLayerKernel *>(&kernel));
    }
    else if(dynamic_cast<CLScaleKernel *>(&kernel) != nullptr)
    {
        tune_scale_kernel(*utils::cast::polymorphic_downcast<CLScaleKernel *>(&kernel));
    }
}

void BifrostTuner::tune_kernel_dynamic(ICLKernel &kernel)
{
    ARM_COMPUTE_UNUSED(kernel);
}

void BifrostTuner::tune_kernel_dynamic(ICLKernel &kernel, ITensorPack &tensors)
{
    ARM_COMPUTE_UNUSED(kernel, tensors);
}
} // namespace tuners
} // namespace arm_compute