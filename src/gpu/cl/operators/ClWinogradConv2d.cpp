/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#include "src/gpu/cl/operators/ClWinogradConv2d.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/experimental/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "src/core/CL/kernels/CLFillBorderKernel.h"
#include "src/core/CL/kernels/CLFillBorderKernel.h"
#include "src/core/helpers/MemoryHelpers.h"
#include "src/gpu/cl/kernels/ClWinogradFilterTransformKernel.h"
#include "src/gpu/cl/kernels/ClWinogradInputTransformKernel.h"
#include "src/gpu/cl/kernels/ClWinogradOutputTransformKernel.h"
#include "src/gpu/cl/utils/ClAuxTensorHandler.h"
#include "support/Cast.h"

using namespace arm_compute::experimental;

namespace arm_compute
{
namespace opencl
{
namespace
{
Size2D winograd_output_tile(const Size2D &input_dims, const Size2D &kernel_dims, DataLayout data_layout)
{
    Size2D output_tile = Size2D{};

    const unsigned int kernel_max_dim = std::max(kernel_dims.width, kernel_dims.height);

    // Check if the input spatial dimensions are smaller than 4
    const bool is_input_lt4_nchw = (input_dims.width <= 4 && input_dims.height <= 4) && (data_layout == DataLayout::NCHW);

    if(kernel_max_dim == 3U)
    {
        if(kernel_dims == Size2D(3U, 3U))
        {
            output_tile = is_input_lt4_nchw ? Size2D(2U, 2U) : Size2D(4U, 4U);
        }
        else if(kernel_dims == Size2D(3U, 1U))
        {
            output_tile = is_input_lt4_nchw ? Size2D(2U, 1U) : Size2D(4U, 1U);
        }
        else
        {
            output_tile = is_input_lt4_nchw ? Size2D(1U, 2U) : Size2D(1U, 4U);
        }
    }
    else if(kernel_max_dim == 5U)
    {
        output_tile = Size2D(kernel_dims.width == 1 ? 1U : 4U,
                             kernel_dims.height == 1 ? 1U : 4U);
    }
    else if(kernel_max_dim == 7U)
    {
        output_tile = Size2D(kernel_dims.width == 1 ? 1U : 2U,
                             kernel_dims.height == 1 ? 1U : 2U);
    }

    return output_tile;
}

bool check_support_fast_math(const Size2D &output_tile, const Size2D &kernel_size)
{
    // Check if we want to configure a Winograd configuration which requires fast math
    using WinogradConfiguration = std::pair<std::pair<int, int>, std::pair<int, int>>;

    std::vector<WinogradConfiguration> fast_math_winograd =
    {
        WinogradConfiguration(std::pair<int, int>(4, 4), std::pair<int, int>(5, 5)),
        WinogradConfiguration(std::pair<int, int>(2, 2), std::pair<int, int>(7, 7))
    };

    auto p = std::make_pair(std::pair<int, int>(output_tile.width, output_tile.height),
                            std::pair<int, int>(kernel_size.width, kernel_size.height));

    return std::find(fast_math_winograd.begin(), fast_math_winograd.end(), p) != fast_math_winograd.end();
}

Status validate_arguments(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *dst, const PadStrideInfo &conv_info,
                          const ActivationLayerInfo &act_info, bool enable_fast_math)
{
    // Get indeces for the width and height
    const size_t idx_width  = get_data_layout_dimension_index(src->data_layout(), DataLayoutDimension::WIDTH);
    const size_t idx_height = get_data_layout_dimension_index(src->data_layout(), DataLayoutDimension::HEIGHT);

    // Input shape, kernel size and output tile
    const Size2D input_dims  = Size2D(src->tensor_shape()[idx_width], src->tensor_shape()[idx_height]);
    const Size2D kernel_size = Size2D(weights->tensor_shape()[idx_width], weights->tensor_shape()[idx_height]);
    const Size2D output_tile = winograd_output_tile(input_dims, kernel_size, src->data_layout());

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(((conv_info.pad_left() > (kernel_size.x() / 2u)) || (conv_info.pad_right() > (kernel_size.x() / 2u))), "Winograd only supports padding up to half kernel size");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(((conv_info.pad_top() > (kernel_size.y() / 2u)) || (conv_info.pad_bottom() > (kernel_size.y() / 2u))), "Winograd only supports padding up to half kernel size");

    // Check if the Winograd configuration requires fast math
    if(!enable_fast_math)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::F32); //disable winograd for fp16 if fast math is false.
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(check_support_fast_math(output_tile, kernel_size), "This Winograd configuration requires enable_fast_math=true");
    }

    const WinogradInfo winograd_info = WinogradInfo(output_tile,
                                                    kernel_size,
                                                    input_dims,
                                                    conv_info,
                                                    src->data_layout());

    // Validate input transform
    const TensorShape input0_shape = misc::shape_calculator::compute_winograd_input_transform_shape(*src, winograd_info);
    const TensorInfo  input0       = src->clone()->set_tensor_shape(input0_shape);
    ARM_COMPUTE_RETURN_ON_ERROR(kernels::ClWinogradInputTransformKernel::validate(src, &input0, winograd_info));

    // Validate filter transform
    const TensorShape input1_shape = misc::shape_calculator::compute_winograd_filter_transform_shape(*weights, winograd_info);
    const TensorInfo  input1       = weights->clone()->set_tensor_shape(input1_shape);
    ARM_COMPUTE_RETURN_ON_ERROR(kernels::ClWinogradFilterTransformKernel::validate(weights, &input1, winograd_info));

    // Validate batched matrix multiply
    TensorShape batched_mm_output_shape = input0.tensor_shape();
    batched_mm_output_shape[0]          = input1.tensor_shape()[0];
    const TensorInfo batched_mm_output  = input0.clone()->set_tensor_shape(batched_mm_output_shape);
    ARM_COMPUTE_RETURN_ON_ERROR(ClGemm::validate(&input0, &input1, nullptr, &batched_mm_output, 1.0f, 0.0f, GEMMInfo(false, false, true /* Reshape weights only for the first run*/, 0, false, false,
                                                                                                                     GEMMLowpOutputStageInfo(), (src->data_type() == DataType::F16))));

    // Configure output transform
    ARM_COMPUTE_RETURN_ON_ERROR(kernels::ClWinogradOutputTransformKernel::validate(&batched_mm_output, biases, dst, winograd_info, act_info));
    return Status{};
}

} // namespace

ClWinogradConv2d::ClWinogradConv2d()
    : _batched_mm(),
      _input_transform(std::make_unique<kernels::ClWinogradInputTransformKernel>()),
      _filter_transform(std::make_unique<kernels::ClWinogradFilterTransformKernel>()),
      _output_transform(std::make_unique<kernels::ClWinogradOutputTransformKernel>()),
      _border_handler(),
      _input0(),
      _input1(),
      _batched_mm_output(),
      _is_prepared(false),
      _aux_mem()
{
}

ClWinogradConv2d::~ClWinogradConv2d() = default;

void ClWinogradConv2d::configure(const ClCompileContext &compile_context, ITensorInfo *src, ITensorInfo *weights, ITensorInfo *biases, ITensorInfo *dst,
                                 const PadStrideInfo &conv_info, const ActivationLayerInfo &act_info, bool enable_fast_math)
{
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, weights, biases, dst, conv_info, act_info, enable_fast_math));
    // Get indices for the width and height
    const size_t idx_width  = get_data_layout_dimension_index(src->data_layout(), DataLayoutDimension::WIDTH);
    const size_t idx_height = get_data_layout_dimension_index(src->data_layout(), DataLayoutDimension::HEIGHT);

    // Input shape, kernel size and output tile
    const Size2D input_dims  = Size2D(src->tensor_shape()[idx_width], src->tensor_shape()[idx_height]);
    const Size2D kernel_size = Size2D(weights->tensor_shape()[idx_width], weights->tensor_shape()[idx_height]);
    const Size2D output_tile = winograd_output_tile(input_dims, kernel_size, src->data_layout());

    // Check if the Winograd configuration requires fast math
    if(!enable_fast_math)
    {
        ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::F32); //disable winograd for fp16 if fast math is false.
        ARM_COMPUTE_ERROR_ON_MSG(check_support_fast_math(output_tile, kernel_size), "This Winograd configuration requires enable_fast_math=true");
    }
    const WinogradInfo winograd_info = WinogradInfo(output_tile,
                                                    kernel_size,
                                                    input_dims,
                                                    conv_info,
                                                    src->data_layout());

    _is_prepared = false;

    // Configure input transform
    _input_transform->configure(compile_context, src, &_input0, winograd_info);
    _border_handler.configure(compile_context, src, _input_transform->border_size(), BorderMode::CONSTANT, PixelValue());

    // Configure filter transform
    _filter_transform->configure(compile_context, weights, &_input1, winograd_info);

    // Configure batched matrix multiply
    _batched_mm.configure(compile_context, &_input0, &_input1, nullptr, &_batched_mm_output, 1.0f, 0.0f, GEMMInfo(false, false, true /* Reshape weights only for the first run*/, 0,
                                                                                                                  false, false,
                                                                                                                  GEMMLowpOutputStageInfo(),
                                                                                                                  (src->data_type() == DataType::F16)));

    // Configure output transform
    _output_transform->configure(compile_context, &_batched_mm_output, biases, dst, winograd_info, act_info);

    _aux_mem                             = _batched_mm.workspace();
    const MemoryLifetime wino_wei_lifetm = std::any_of(std::begin(_aux_mem), std::end(_aux_mem), [](const auto & r)
    {
        return (r.lifetime == MemoryLifetime::Persistent) && (r.size > 0);
    }) ?
    MemoryLifetime::Prepare :
    MemoryLifetime::Persistent;
    _aux_mem.push_back(MemoryInfo(offset_int_vec(2), MemoryLifetime::Temporary, _input0.total_size()));
    _aux_mem.push_back(MemoryInfo(offset_int_vec(3), wino_wei_lifetm, _input1.total_size()));
    _aux_mem.push_back(MemoryInfo(offset_int_vec(4), MemoryLifetime::Temporary, _batched_mm_output.total_size()));
}

Status ClWinogradConv2d::validate(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *dst, const PadStrideInfo &conv_info,
                                  const ActivationLayerInfo &act_info, bool enable_fast_math)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, weights, biases, dst, conv_info, act_info, enable_fast_math));
    return Status{};
}

void ClWinogradConv2d::run(ITensorPack &tensors)
{
    const bool is_gemm_reshaped = _aux_mem[3].lifetime == MemoryLifetime::Prepare;

    auto src    = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_0));
    auto biases = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_2));
    auto dst    = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));

    CLAuxTensorHandler input0(offset_int_vec(2), _input0, tensors, true);
    CLAuxTensorHandler input1(offset_int_vec(3), _input1, tensors, true, is_gemm_reshaped);
    CLAuxTensorHandler batched_mm_output(offset_int_vec(4), _batched_mm_output, tensors, true);

    prepare(tensors);

    // Run input transform
    ITensorPack pack_it
    {
        { TensorType::ACL_SRC, src },
        { TensorType::ACL_DST, input0.get() },
    };
    CLScheduler::get().enqueue_op(_border_handler, pack_it, false);
    CLScheduler::get().enqueue_op(*_input_transform, pack_it, false);

    // Run batched matrix multiplication
    ITensorPack pack_mm = tensors;
    pack_mm.add_const_tensor(TensorType::ACL_SRC_0, input0.get());
    pack_mm.add_tensor(TensorType::ACL_DST, batched_mm_output.get());
    is_gemm_reshaped ? pack_mm.remove_tensor(TensorType::ACL_SRC_1) : pack_mm.add_const_tensor(TensorType::ACL_SRC_1, input1.get());
    _batched_mm.run(pack_mm);

    // Run output transform
    ITensorPack pack_ot
    {
        { TensorType::ACL_SRC_0, batched_mm_output.get() },
        { TensorType::ACL_SRC_1, biases },
        { TensorType::ACL_DST, dst },
    };
    CLScheduler::get().enqueue_op(*_output_transform, pack_ot);
}

void ClWinogradConv2d::prepare(ITensorPack &tensors)
{
    if(!_is_prepared)
    {
        auto       weights = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_1));
        ICLTensor *in1_aux = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(offset_int_vec(3)));

        CLAuxTensorHandler input1(_input1, *in1_aux);
        ITensorPack        pack_ft
        {
            { TensorType::ACL_SRC, weights },
            { TensorType::ACL_DST, input1.get() },
        };
        // Run filter transform and mark original weights as unused
        CLScheduler::get().enqueue_op(*_filter_transform, pack_ft, false);
        weights->mark_as_unused();

        // Prepare GEMM and release reshaped weights if marked unused by ClGemm
        ITensorPack mm_prepare_pack = tensors;
        mm_prepare_pack.add_tensor(ACL_SRC_1, input1.get());
        _batched_mm.prepare(mm_prepare_pack);

        CLScheduler::get().queue().finish();
        _is_prepared = true;
    }
}

experimental::MemoryRequirements ClWinogradConv2d::workspace() const
{
    return _aux_mem;
}
} // namespace opencl
} // namespace arm_compute