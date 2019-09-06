/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#include "arm_compute/runtime/CPP/functions/CPPBoxWithNonMaximaSuppressionLimit.h"

#include "arm_compute/core/CPP/kernels/CPPBoxWithNonMaximaSuppressionLimitKernel.h"
#include "arm_compute/runtime/Scheduler.h"

namespace arm_compute
{
namespace
{
void dequantize_tensor(const ITensor *input, ITensor *output, DataType data_type)
{
    const UniformQuantizationInfo qinfo = input->info()->quantization_info().uniform();

    Window window;
    window.use_tensor_dimensions(input->info()->tensor_shape());
    Iterator input_it(input, window);
    Iterator output_it(output, window);

    switch(data_type)
    {
        case DataType::QASYMM8:
            execute_window_loop(window, [&](const Coordinates &)
            {
                *reinterpret_cast<float *>(output_it.ptr()) = dequantize(*reinterpret_cast<const uint8_t *>(input_it.ptr()), qinfo.scale, qinfo.offset);
            },
            input_it, output_it);
            break;
        case DataType::QASYMM16:
            execute_window_loop(window, [&](const Coordinates &)
            {
                *reinterpret_cast<float *>(output_it.ptr()) = dequantize(*reinterpret_cast<const uint16_t *>(input_it.ptr()), qinfo.scale, qinfo.offset);
            },
            input_it, output_it);
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported data type");
    }
}

void quantize_tensor(const ITensor *input, ITensor *output, DataType data_type)
{
    const UniformQuantizationInfo qinfo = input->info()->quantization_info().uniform();

    Window window;
    window.use_tensor_dimensions(input->info()->tensor_shape());
    Iterator input_it(input, window);
    Iterator output_it(output, window);

    switch(data_type)
    {
        case DataType::QASYMM8:
            execute_window_loop(window, [&](const Coordinates &)
            {
                *reinterpret_cast<uint8_t *>(output_it.ptr()) = quantize_qasymm8(*reinterpret_cast<const float *>(input_it.ptr()), qinfo);
            },
            input_it, output_it);
            break;
        case DataType::QASYMM16:
            execute_window_loop(window, [&](const Coordinates &)
            {
                *reinterpret_cast<uint16_t *>(output_it.ptr()) = quantize_qasymm16(*reinterpret_cast<const float *>(input_it.ptr()), qinfo);
            },
            input_it, output_it);
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported data type");
    }
}
} // namespace

CPPBoxWithNonMaximaSuppressionLimit::CPPBoxWithNonMaximaSuppressionLimit(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)),
      _box_with_nms_limit_kernel(),
      _scores_in(),
      _boxes_in(),
      _batch_splits_in(),
      _scores_out(),
      _boxes_out(),
      _classes(),
      _batch_splits_out(),
      _keeps(),
      _keeps_size(),
      _scores_in_f32(),
      _boxes_in_f32(),
      _batch_splits_in_f32(),
      _scores_out_f32(),
      _boxes_out_f32(),
      _classes_f32(),
      _batch_splits_out_f32(),
      _keeps_f32(),
      _keeps_size_f32(),
      _is_qasymm8(false)
{
}

void CPPBoxWithNonMaximaSuppressionLimit::configure(const ITensor *scores_in, const ITensor *boxes_in, const ITensor *batch_splits_in, ITensor *scores_out, ITensor *boxes_out, ITensor *classes,
                                                    ITensor *batch_splits_out, ITensor *keeps, ITensor *keeps_size, const BoxNMSLimitInfo info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(scores_in, boxes_in, batch_splits_in, scores_out, boxes_out, classes);

    _is_qasymm8 = scores_in->info()->data_type() == DataType::QASYMM8;

    _scores_in        = scores_in;
    _boxes_in         = boxes_in;
    _batch_splits_in  = batch_splits_in;
    _scores_out       = scores_out;
    _boxes_out        = boxes_out;
    _classes          = classes;
    _batch_splits_out = batch_splits_out;
    _keeps            = keeps;
    _keeps_size       = keeps_size;

    if(_is_qasymm8)
    {
        // Manage intermediate buffers
        _memory_group.manage(&_scores_in_f32);
        _memory_group.manage(&_boxes_in_f32);
        _memory_group.manage(&_batch_splits_in_f32);
        _memory_group.manage(&_scores_out_f32);
        _memory_group.manage(&_boxes_out_f32);
        _memory_group.manage(&_classes_f32);
        _scores_in_f32.allocator()->init(scores_in->info()->clone()->set_data_type(DataType::F32));
        _boxes_in_f32.allocator()->init(boxes_in->info()->clone()->set_data_type(DataType::F32));
        _batch_splits_in_f32.allocator()->init(batch_splits_in->info()->clone()->set_data_type(DataType::F32));
        _scores_out_f32.allocator()->init(scores_out->info()->clone()->set_data_type(DataType::F32));
        _boxes_out_f32.allocator()->init(boxes_out->info()->clone()->set_data_type(DataType::F32));
        _classes_f32.allocator()->init(classes->info()->clone()->set_data_type(DataType::F32));
        if(batch_splits_out != nullptr)
        {
            _memory_group.manage(&_batch_splits_out_f32);
            _batch_splits_out_f32.allocator()->init(batch_splits_out->info()->clone()->set_data_type(DataType::F32));
        }
        if(keeps != nullptr)
        {
            _memory_group.manage(&_keeps_f32);
            _keeps_f32.allocator()->init(keeps->info()->clone()->set_data_type(DataType::F32));
        }
        if(keeps_size != nullptr)
        {
            _memory_group.manage(&_keeps_size_f32);
            _keeps_size_f32.allocator()->init(keeps_size->info()->clone()->set_data_type(DataType::F32));
        }

        _box_with_nms_limit_kernel.configure(&_scores_in_f32, &_boxes_in_f32, &_batch_splits_in_f32, &_scores_out_f32, &_boxes_out_f32, &_classes_f32,
                                             (batch_splits_out != nullptr) ? &_batch_splits_out_f32 : nullptr, (keeps != nullptr) ? &_keeps_f32 : nullptr,
                                             (keeps_size != nullptr) ? &_keeps_size_f32 : nullptr, info);
    }
    else
    {
        _box_with_nms_limit_kernel.configure(scores_in, boxes_in, batch_splits_in, scores_out, boxes_out, classes, batch_splits_out, keeps, keeps_size, info);
    }

    if(_is_qasymm8)
    {
        _scores_in_f32.allocator()->allocate();
        _boxes_in_f32.allocator()->allocate();
        _batch_splits_in_f32.allocator()->allocate();
        _scores_out_f32.allocator()->allocate();
        _boxes_out_f32.allocator()->allocate();
        _classes_f32.allocator()->allocate();
        if(batch_splits_out != nullptr)
        {
            _batch_splits_out_f32.allocator()->allocate();
        }
        if(keeps != nullptr)
        {
            _keeps_f32.allocator()->allocate();
        }
        if(keeps_size != nullptr)
        {
            _keeps_size_f32.allocator()->allocate();
        }
    }
}

Status validate(const ITensorInfo *scores_in, const ITensorInfo *boxes_in, const ITensorInfo *batch_splits_in, const ITensorInfo *scores_out, const ITensorInfo *boxes_out, const ITensorInfo *classes,
                const ITensorInfo *batch_splits_out, const ITensorInfo *keeps, const ITensorInfo *keeps_size, const BoxNMSLimitInfo info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(scores_in, boxes_in, batch_splits_in, scores_out, boxes_out, classes);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(scores_in, 1, DataType::QASYMM8, DataType::F16, DataType::F32);

    const bool is_qasymm8 = scores_in->data_type() == DataType::QASYMM8;
    if(is_qasymm8)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(boxes_in, 1, DataType::QASYMM16);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(boxes_in, boxes_out);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(boxes_in, boxes_out);
        const UniformQuantizationInfo boxes_qinfo = boxes_in->quantization_info().uniform();
        ARM_COMPUTE_RETURN_ERROR_ON(boxes_qinfo.scale != 0.125f);
        ARM_COMPUTE_RETURN_ERROR_ON(boxes_qinfo.offset != 0);
    }

    return Status{};
}

void CPPBoxWithNonMaximaSuppressionLimit::run()
{
    if(_is_qasymm8)
    {
        dequantize_tensor(_scores_in, &_scores_in_f32, _scores_in->info()->data_type());
        dequantize_tensor(_boxes_in, &_boxes_in_f32, _boxes_in->info()->data_type());
        dequantize_tensor(_batch_splits_in, &_batch_splits_in_f32, _batch_splits_in->info()->data_type());
    }

    Scheduler::get().schedule(&_box_with_nms_limit_kernel, Window::DimY);

    if(_is_qasymm8)
    {
        quantize_tensor(&_scores_out_f32, _scores_out, _scores_out->info()->data_type());
        quantize_tensor(&_boxes_out_f32, _boxes_out, _boxes_out->info()->data_type());
        quantize_tensor(&_classes_f32, _classes, _classes->info()->data_type());
        if(_batch_splits_out != nullptr)
        {
            quantize_tensor(&_batch_splits_out_f32, _batch_splits_out, _batch_splits_out->info()->data_type());
        }
        if(_keeps != nullptr)
        {
            quantize_tensor(&_keeps_f32, _keeps, _keeps->info()->data_type());
        }
        if(_keeps_size != nullptr)
        {
            quantize_tensor(&_keeps_size_f32, _keeps_size, _keeps_size->info()->data_type());
        }
    }
}
} // namespace arm_compute
