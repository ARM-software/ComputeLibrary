/*
 * Copyright (c) 2018 ARM Limited.
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
#include "arm_compute/core/utils/misc/Cast.h"

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
} // namespace

void BifrostTuner::tune_kernel_static(ICLKernel &kernel)
{
    // Continue on tuning if dynamic tuning
    if(dynamic_cast<CLDirectConvolutionLayerKernel *>(&kernel) != nullptr)
    {
        tune_direct_convolution_kernel(*utils::cast::polymorphic_downcast<CLDirectConvolutionLayerKernel *>(&kernel));
    }
}

void BifrostTuner::tune_kernel_dynamic(ICLKernel &kernel)
{
    ARM_COMPUTE_UNUSED(kernel);
}
} // namespace tuners
} // namespace arm_compute