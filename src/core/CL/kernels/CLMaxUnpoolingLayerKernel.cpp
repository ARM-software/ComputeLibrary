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
#include "src/core/CL/kernels/CLMaxUnpoolingLayerKernel.h"

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
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/StringSupport.h"

namespace arm_compute
{
using namespace misc::shape_calculator;

namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const PoolingLayerInfo &pool_info, const ITensorInfo *indices)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output, indices);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(indices, 1, DataType::U32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, indices);

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

CLMaxUnpoolingLayerKernel::CLMaxUnpoolingLayerKernel()
    : _input(nullptr), _output(nullptr), _indices(nullptr)
{
}

void CLMaxUnpoolingLayerKernel::configure(const CLCompileContext &compile_context, const ICLTensor *input, const ICLTensor *indices, ICLTensor *output, const PoolingLayerInfo &pool_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), pool_info, indices->info()));
    auto padding_info = get_padding_info({ input, indices, output });

    _input   = input;
    _output  = output;
    _indices = indices;

    // Create build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_unsigned_type_from_element_size(input->info()->element_size()));
    build_opts.add_option("-DWIDTH_DST=" + support::cpp11::to_string(output->info()->dimension(0)));
    build_opts.add_option("-DHEIGHT_DST=" + support::cpp11::to_string(output->info()->dimension(1)));
    build_opts.add_option("-DDEPTH_DST=" + support::cpp11::to_string(output->info()->dimension(2)));

    const std::string kernel_name("max_unpooling_layer_2");

    // Create kernel
    _kernel = create_kernel(compile_context, kernel_name, build_opts.options());

    const TensorShape output_shape = compute_unpool_shape(*input->info(), pool_info);
    auto_init_if_empty(*output->info(), input->info()->clone()->set_tensor_shape(output_shape));

    auto window = calculate_max_window(*input->info(), Steps());
    ICLKernel::configure_internal(window);

    // Set config_id for enabling LWS tuning
    _config_id = kernel_name;
    _config_id += lower_string(string_from_data_type(input->info()->data_type()));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(1));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(2));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(3));
    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
}

Status CLMaxUnpoolingLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *indices, const ITensorInfo *output, const PoolingLayerInfo &pool_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, indices, output);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, pool_info, indices));
    return Status{};
}

void CLMaxUnpoolingLayerKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window slice = window.first_slice_window_3D();

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, slice);
        add_3D_tensor_argument(idx, _output, slice);
        add_3D_tensor_argument(idx, _indices, slice);
        enqueue(queue, *this, slice, lws_hint());
    }
    while(window.slide_window_slice_3D(slice));
}
} // namespace arm_compute
