/*
 * Copyright (c) 2019 ARM Limited.
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
#include "arm_compute/runtime/CL/functions/CLGEMMDeconvolutionLayer.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "utils/TypePrinter.h"

#include <memory>
#include <tuple>

namespace arm_compute
{
namespace
{
std::pair<Coordinates, Coordinates> compute_start_end_slice_coordinates(const ITensorInfo &output_info, const PadStrideInfo &deconv_info, bool is_nchw)
{
    Coordinates start;
    Coordinates end;

    if(is_nchw)
    {
        start.set(0, deconv_info.pad_left());
        start.set(1, deconv_info.pad_top());
        end.set(0, output_info.dimension(0) - deconv_info.pad_right());
        end.set(1, output_info.dimension(1) - deconv_info.pad_bottom());
    }
    else
    {
        start.set(0, 0);
        start.set(1, deconv_info.pad_left());
        start.set(2, deconv_info.pad_top());

        end.set(0, output_info.dimension(0));
        end.set(1, output_info.dimension(1) - deconv_info.pad_right());
        end.set(2, output_info.dimension(2) - deconv_info.pad_bottom());
    }

    return { start, end };
}
} // namespace

CLGEMMDeconvolutionLayer::CLGEMMDeconvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager) // NOLINT
    : _memory_group(std::move(memory_manager)),
      _mm_gemm(),
      _mm_gemmlowp(),
      _gemmlowp_output_stage(),
      _permute_input_to_nhwc(),
      _permute_weights_to_nhwc(),
      _reshape_weights(),
      _transpose_weights(),
      _deconv_reshape(),
      _slice_gemm(),
      _gemmlowp_final(),
      _reshaped_weights(),
      _reshaped_weights_t(),
      _permuted_input(),
      _permuted_weights(),
      _gemm_output(),
      _slice_gemm_input(),
      _original_weights(),
      _is_prepared(false),
      _padded_input(false),
      _is_nchw(false),
      _is_quantized(false)
{
}

Status CLGEMMDeconvolutionLayer::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *bias, const ITensorInfo *output, const PadStrideInfo &deconv_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32, DataType::F16, DataType::QASYMM8);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(input, weights);

    DataLayout data_layout  = input->data_layout();
    const bool padded_input = deconv_info.pad_bottom() > 0 || deconv_info.pad_left() > 0 || deconv_info.pad_right() > 0 || deconv_info.pad_top() > 0;
    const bool is_nchw      = input->data_layout() == DataLayout::NCHW;
    const bool is_quantized = is_data_type_quantized_asymmetric(input->data_type());

    const size_t idx_w = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const size_t idx_h = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const size_t idx_b = get_data_layout_dimension_index(data_layout, DataLayoutDimension::BATCHES);

    ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(idx_w) != deconv_info.stride().first);
    ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(idx_h) != deconv_info.stride().second);

    TensorShape nhwc_weights_shape = weights->tensor_shape();
    TensorShape nhwc_input_shape   = input->tensor_shape();

    if(is_nchw)
    {
        permute(nhwc_weights_shape, PermutationVector(2, 0, 1));
        permute(nhwc_input_shape, PermutationVector(2, 0, 1));

        TensorInfo nhwc_input_info = input->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(nhwc_input_shape).set_data_layout(DataLayout::NCHW);

        TensorInfo nhwc_weights_info = weights->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(nhwc_weights_shape).set_data_layout(DataLayout::NCHW);

        CLPermute::validate(weights, &nhwc_weights_info, PermutationVector(2, 0, 1));
        CLPermute::validate(input, &nhwc_input_info, PermutationVector(2, 0, 1));
    }

    const TensorShape reshaped_shape = TensorShape(nhwc_weights_shape[0], nhwc_weights_shape[1] * nhwc_weights_shape[2] * nhwc_weights_shape[3]);
    const TensorInfo  reshaped_info  = weights->clone()->set_tensor_shape(reshaped_shape).set_data_layout(DataLayout::NCHW).set_is_resizable(true);
    ARM_COMPUTE_RETURN_ON_ERROR(CLReshapeLayer::validate(weights, &reshaped_info));

    TensorShape      transposed_shape(reshaped_shape[1], reshaped_shape[0]);
    const TensorInfo reshaped_t_info = reshaped_info.clone()->set_is_resizable(true).set_tensor_shape(transposed_shape);
    ARM_COMPUTE_RETURN_ON_ERROR(CLTranspose::validate(&reshaped_info, &reshaped_t_info));

    TensorShape gemm_output_shape(weights->dimension(idx_w) * weights->dimension(idx_h) * weights->dimension(idx_b),
                                  input->dimension(idx_w),
                                  input->dimension(idx_h),
                                  input->dimension(idx_b));

    TensorInfo gemm_output_info = reshaped_t_info.clone()->set_tensor_shape(gemm_output_shape).set_is_resizable(true);
    GEMMInfo   gemm_info(false, false, true, input->dimension(idx_h), true);

    if(is_quantized)
    {
        ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMLowpMatrixMultiplyCore::validate(&input->clone()->set_tensor_shape(nhwc_input_shape), &reshaped_t_info, nullptr, &gemm_output_info.set_data_type(DataType::S32),
                                                                           gemm_info));
    }
    else
    {
        ARM_COMPUTE_RETURN_ON_ERROR(CLGEMM::validate(&input->clone()->set_tensor_shape(nhwc_input_shape).set_is_resizable(true), &reshaped_t_info, nullptr, &gemm_output_info, 1.0f, 0.0f, gemm_info));
    }

    auto out_dims = deconvolution_output_dimensions(input->dimension(idx_w), input->dimension(idx_h), weights->dimension(idx_w), weights->dimension(idx_h),
                                                    0, 0, deconv_info.stride().first, deconv_info.stride().second);
    const TensorShape deconv_shape       = misc::shape_calculator::compute_deconvolution_output_shape(out_dims, *input, *weights);
    TensorInfo        col2im_output_info = gemm_output_info.clone()->set_tensor_shape(deconv_shape).set_is_resizable(true);

    if(padded_input && is_quantized)
    {
        const auto start_end = compute_start_end_slice_coordinates(col2im_output_info, deconv_info, is_nchw);
        ARM_COMPUTE_RETURN_ON_ERROR(CLDeconvolutionReshapeOutputKernel::validate(&gemm_output_info, bias, &col2im_output_info, input, weights, deconv_info));
        ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint::validate(&col2im_output_info, nullptr,
                                                                                                  &col2im_output_info.clone()->set_is_resizable(true).set_data_type(DataType::QASYMM8)));
        ARM_COMPUTE_RETURN_ON_ERROR(CLSlice::validate(&col2im_output_info.clone()->set_is_resizable(true).set_data_type(DataType::QASYMM8), output, start_end.first, start_end.second));
    }
    else if(padded_input)
    {
        const auto start_end = compute_start_end_slice_coordinates(col2im_output_info, deconv_info, is_nchw);
        ARM_COMPUTE_RETURN_ON_ERROR(CLDeconvolutionReshapeOutputKernel::validate(&gemm_output_info, bias, &col2im_output_info, input, weights, deconv_info));
        ARM_COMPUTE_RETURN_ON_ERROR(CLSlice::validate(&col2im_output_info, output, start_end.first, start_end.second));
    }
    else if(is_quantized)
    {
        ARM_COMPUTE_RETURN_ON_ERROR(CLDeconvolutionReshapeOutputKernel::validate(&gemm_output_info, bias, &col2im_output_info, input, weights, deconv_info));
        ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint::validate(&col2im_output_info, nullptr, output));
    }
    else
    {
        ARM_COMPUTE_RETURN_ON_ERROR(CLDeconvolutionReshapeOutputKernel::validate(&gemm_output_info, bias, output, input, weights, deconv_info));
    }

    return Status{};
}

void CLGEMMDeconvolutionLayer::configure(const ICLTensor *input, const ICLTensor *weights, const ICLTensor *bias, ICLTensor *output, const PadStrideInfo &deconv_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_ERROR_THROW_ON(CLGEMMDeconvolutionLayer::validate(input->info(),
                                                                  weights->info(),
                                                                  bias != nullptr ? bias->info() : nullptr,
                                                                  output->info(),
                                                                  deconv_info));

    _original_weights = weights;
    _padded_input     = deconv_info.pad_bottom() > 0 || deconv_info.pad_left() > 0 || deconv_info.pad_right() > 0 || deconv_info.pad_top() > 0;
    _is_nchw          = input->info()->data_layout() == DataLayout::NCHW;
    _is_quantized     = is_data_type_quantized_asymmetric(input->info()->data_type());

    const ICLTensor *input_to_use   = input;
    const ICLTensor *weights_to_use = weights;

    // If the data layout is NCHW, transform everything in NHWC. Another alternative could be to
    // do an outer product in NCHW and then an accumulation through a reduction. This would have two
    // drawbacks: first, the outer product is less efficient than a full GEMM. Second, the reduction
    // might be slower than GEMM.
    if(_is_nchw)
    {
        _memory_group.manage(&_permuted_input);
        _permute_input_to_nhwc.configure(input, &_permuted_input, PermutationVector(2U, 0U, 1U));

        _permute_weights_to_nhwc.configure(weights, &_permuted_weights, PermutationVector(2U, 0U, 1U));

        input_to_use   = &_permuted_input;
        weights_to_use = &_permuted_weights;
    }

    // Reshape the input weights. The weights will be reshaped only once during the call to prepare()
    _reshaped_weights.allocator()->init(TensorInfo(TensorShape(weights_to_use->info()->dimension(0),
                                                               weights_to_use->info()->dimension(1) * weights_to_use->info()->dimension(2) * weights_to_use->info()->dimension(3)),
                                                   1,
                                                   input->info()->data_type(), weights->info()->quantization_info()));

    _reshape_weights.configure(weights_to_use, &_reshaped_weights);
    _transpose_weights.configure(&_reshaped_weights, &_reshaped_weights_t);

    const size_t idx_h = get_data_layout_dimension_index(input->info()->data_layout(), DataLayoutDimension::HEIGHT);
    GEMMInfo     gemm_info(false, false, true, input->info()->dimension(idx_h), true);

    // Configure output stage for asymmetric quantized types
    if(_is_quantized)
    {
        _mm_gemmlowp.configure(input_to_use, &_reshaped_weights_t, nullptr, &_gemm_output, gemm_info);
    }
    else
    {
        _mm_gemm.configure(input_to_use, &_reshaped_weights_t, nullptr, &_gemm_output, 1.f, 0.0f, gemm_info);
    }

    if(_is_nchw)
    {
        _permuted_input.allocator()->allocate();
    }

    ICLTensor *deconv_reshape_output = nullptr;
    ICLTensor *slice_output          = nullptr;
    ICLTensor *output_stage_output   = nullptr;

    if(_padded_input && _is_quantized)
    {
        _memory_group.manage(&_slice_gemm_input);
        _memory_group.manage(&_gemmlowp_final);
        deconv_reshape_output = &_gemmlowp_final;
        output_stage_output   = &_slice_gemm_input;
        slice_output          = output;
    }
    else if(_padded_input)
    {
        _memory_group.manage(&_slice_gemm_input);
        deconv_reshape_output = &_slice_gemm_input;
        slice_output          = output;
    }
    else if(_is_quantized)
    {
        _memory_group.manage(&_gemmlowp_final);
        deconv_reshape_output = &_gemmlowp_final;
        output_stage_output   = output;
    }
    else
    {
        deconv_reshape_output = output;
    }

    // Configure a Col2Im call to reshape the output of GEMM
    _deconv_reshape.configure(&_gemm_output, bias, deconv_reshape_output, input->info(), weights->info(), deconv_info);
    _gemm_output.allocator()->allocate();

    if(_is_quantized)
    {
        const UniformQuantizationInfo iq_info = input->info()->quantization_info().uniform();
        const UniformQuantizationInfo wq_info = weights->info()->quantization_info().uniform();
        const UniformQuantizationInfo oq_info = _gemmlowp_final.info()->quantization_info().uniform();

        float multiplier = iq_info.scale * wq_info.scale / oq_info.scale;
        int   output_multiplier(0);
        int   output_shift(0);
        quantization::calculate_quantized_multiplier_less_than_one(multiplier, &output_multiplier, &output_shift);
        _gemmlowp_output_stage.configure(&_gemmlowp_final, nullptr, output_stage_output, output_multiplier, output_shift, oq_info.offset);
        _gemmlowp_final.allocator()->allocate();
    }

    // If the input was padded, the output needs to be sliced.
    if(_padded_input)
    {
        const auto start_end = compute_start_end_slice_coordinates(*deconv_reshape_output->info(), deconv_info, _is_nchw);
        _slice_gemm.configure(&_slice_gemm_input, slice_output, start_end.first, start_end.second);
        _slice_gemm_input.allocator()->allocate();
    }
}

void CLGEMMDeconvolutionLayer::run()
{
    prepare();

    MemoryGroupResourceScope scope_mg(_memory_group);

    if(_is_nchw)
    {
        _permute_input_to_nhwc.run();
    }

    if(_is_quantized)
    {
        _mm_gemmlowp.run();
    }
    else
    {
        _mm_gemm.run();
    }

    CLScheduler::get().enqueue(_deconv_reshape, false);

    if(_is_quantized)
    {
        _gemmlowp_output_stage.run();
    }

    if(_padded_input)
    {
        _slice_gemm.run();
    }
}

void CLGEMMDeconvolutionLayer::prepare()
{
    if(!_is_prepared)
    {
        ARM_COMPUTE_ERROR_ON(!_original_weights->is_used());

        if(_is_nchw)
        {
            _permuted_weights.allocator()->allocate();
            _permute_weights_to_nhwc.run();
        }

        _reshaped_weights.allocator()->allocate();
        _reshape_weights.run();

        if(_is_nchw)
        {
            _permuted_weights.allocator()->free();
        }

        _reshaped_weights_t.allocator()->allocate();
        _transpose_weights.run();

        // Prepare gemm
        if(!_is_quantized)
        {
            _mm_gemm.prepare();
        }
        else
        {
            _mm_gemmlowp.prepare();
        }

        // Free resources
        if(!_reshaped_weights_t.is_used())
        {
            _reshaped_weights_t.allocator()->free();
        }

        _original_weights->mark_as_unused();
        _is_prepared = true;
    }
}
} // namespace arm_compute
