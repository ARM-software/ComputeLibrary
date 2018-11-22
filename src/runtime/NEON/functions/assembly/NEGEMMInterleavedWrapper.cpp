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

#include "arm_compute/runtime/NEON/functions/assembly/NEGEMMInterleavedWrapper.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/kernels/assembly/Helpers.h"
#include "arm_compute/core/NEON/kernels/assembly/NEGEMMInterleavedMatrixMultiplyWrapper.h"
#include "arm_compute/core/NEON/kernels/assembly/NEGEMMInterleavedPrepareBWrapperKernel.h"
#include "arm_compute/core/NEON/kernels/assembly/NEGEMMInterleavedTransformAWrapper.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

namespace arm_compute
{
NEGEMMInterleavedWrapper::NEGEMMInterleavedWrapper(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager))
{
}
void NEGEMMInterleavedWrapper::run()
{
    prepare();

    _memory_group.acquire();
    NEScheduler::get().run_tagged_workloads(_workloads, _tag.c_str());
    _memory_group.release();
}

void NEGEMMInterleavedWrapper::prepare()
{
    if(!_is_prepared)
    {
        if(_pretranspose_b)
        {
            _transformed_b.allocator()->allocate();
            NEScheduler::get().schedule(_prepare_b.get(), Window::DimX);
            _b->mark_as_unused();
        }
        else
        {
            _prepare_b->create_workloads(_b_workloads);
        }
        _transform_a->create_workloads(_a_workloads);
        _matrix_multiply->create_workloads(_mm_workloads);

        //Maximum number of workloads to create:
        const unsigned int num_threads    = NEScheduler::get().num_threads();
        const unsigned int max_iterations = num_threads == 1 ? 1 : num_threads;
        //Maximum number of iterations the parameters allow:
        const unsigned int num_iterations = _batch_window.num_iterations_total();
        // Keep the smallest of the two:
        const unsigned int num_windows  = std::min(num_iterations, max_iterations);
        const TensorShape  window_shape = _batch_window.shape();

        // Create a 1D window to dynamically split the batch window:
        Window win_1D;
        win_1D.set(0, Window::Dimension(0, num_iterations));

        // Create one workload for each sub-window:
        for(unsigned int w = 0; w < num_windows; w++)
        {
            Window             win          = win_1D.split_window(0, w, num_windows);
            const Coordinates  start_offset = index2coords(window_shape, win.x().start());
            const Coordinates  end_offset   = index2coords(window_shape, win.x().end() - 1);
            const unsigned int num_x_blocks = _block_walker.num_iterations(Window::DimX);

            auto workload = [start_offset, end_offset, num_x_blocks, this](const ThreadInfo & info)
            {
                //For each block of rows in "M"
                auto workload_mm = this->_mm_workloads.begin();
                for(auto workload_a = this->_a_workloads.begin(); workload_a != this->_a_workloads.end(); workload_a++)
                {
                    // Transform one k_block from A:
                    this->_transform_a->transform(*workload_a, info, this->_batch_window, start_offset, end_offset);
                    // Then perform the matrix multiplication for each x block along N:
                    for(unsigned int i = 0; i < num_x_blocks; i++)
                    {
                        ARM_COMPUTE_ERROR_ON(workload_mm == this->_mm_workloads.end());
                        this->_matrix_multiply->transform(*workload_mm++, info, this->_batch_window, start_offset, end_offset);
                    }
                }
            };
            _workloads.push_back(workload);
        }

        _is_prepared = true;
    }
}

namespace
{
// Factory to instantiate NEGEMMInterleavedPrepareBWrapperKernel:
template <typename InputType, bool use_dot = false>
std::unique_ptr<NEGEMMInterleavedPrepareBWrapperKernel> instantiate_prepareB(const ITensor *b, ITensor *transformed_b, const INEGEMMWrapperKernel::Params &params)
{
    auto prepare_b = support::cpp14::make_unique<NEGEMMInterleavedPrepareBWrapperKernelTemplate<InputType, use_dot>>();
    prepare_b->configure(b, transformed_b, false, NEScheduler::get().cpu_info(), params);
    return std::move(prepare_b);
}

// Factory to instantiate NEGEMMInterleavedTransformAWrapperTemplate:
template <typename InputType, bool use_dot = false>
std::unique_ptr<NEGEMMInterleavedTransformAWrapper> instantiate_transformA(const ITensor *a, ITensor *transformed_a, const Window &block_walker, const INEGEMMWrapperKernel::Params &params)
{
    auto transform_a = support::cpp14::make_unique<NEGEMMInterleavedTransformAWrapperTemplate<InputType, use_dot>>();
    transform_a->configure(a, transformed_a, false, block_walker, params);
    return std::move(transform_a);
}

// Factory to instantiate NEGEMMInterleavedTransformAWrapperTemplate:
template <typename InputType, typename OutputType, bool use_dot = false>
std::unique_ptr<NEGEMMInterleavedMatrixMultiplyWrapper> instantiate_matrix_multiply(const ITensor *transformed_a, const ITensor *transformed_b, ITensor *tmp_c, ITensor *c, const Window &block_walker,
                                                                                    const BlockSizes &block_sizes, const INEGEMMWrapperKernel::Params &params, bool pretranspose_b, float alpha, float beta)
{
    auto matrix_multiply = support::cpp14::make_unique<NEGEMMInterleavedMatrixMultiplyWrapperTemplate<InputType, OutputType, use_dot>>();
    matrix_multiply->configure(transformed_a, transformed_b, tmp_c, c, block_walker, block_sizes, params, pretranspose_b, alpha, beta, NEScheduler::get().num_threads());
    return std::move(matrix_multiply);
}
} // namespace

void NEGEMMInterleavedWrapper::configure(const ITensor *a, const ITensor *b, ITensor *c, float alpha, float beta, bool pretranspose_b, bool use_dot)
{
    _params         = INEGEMMWrapperKernel::extract_parameters(a, b, c);
    _a              = a;
    _b              = b;
    _c              = c;
    _pretranspose_b = pretranspose_b;

    DataType input_type = a->info()->data_type();

    // Forcing 128-byte alignment (required by 32-bit kernels)
    const unsigned int alignment = 128;
    _transformed_b.allocator()->init(TensorInfo{}, alignment);
    _tmp_c.allocator()->init(TensorInfo{}, alignment);
    _tag = "NEGEMMInterleaved_";
    _tag += get_strategy_name(input_type, use_dot);

    if(!_pretranspose_b)
    {
        // If B is transposed at every iteration then transformed_B can be managed:
        _memory_group.manage(&_transformed_b);
        _block_sizes = calculate_block_sizes_from_data_type(NEScheduler::get().cpu_info(), _params.M, _params.N, _params.K, input_type, use_dot);
    }
    else
    {
        _tag += "_preB";
        switch(input_type)
        {
            case DataType::F32:
                _prepare_b = instantiate_prepareB<float>(_b, &_transformed_b, _params);
                break;
#ifdef __aarch64__
            case DataType::U8:
            case DataType::QASYMM8:
                if(use_dot)
                {
                    _prepare_b = instantiate_prepareB<uint8_t, true>(_b, &_transformed_b, _params);
                }
                else
                {
                    _prepare_b = instantiate_prepareB<uint8_t, false>(_b, &_transformed_b, _params);
                }
                break;
            case DataType::S8:
                if(use_dot)
                {
                    _prepare_b = instantiate_prepareB<int8_t, true>(_b, &_transformed_b, _params);
                }
                else
                {
                    _prepare_b = instantiate_prepareB<int8_t, false>(_b, &_transformed_b, _params);
                }
                break;
#endif /* __aarch64__ */
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            case DataType::F16:
                _prepare_b = instantiate_prepareB<__fp16>(_b, &_transformed_b, _params);
                break;
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
            default:
                ARM_COMPUTE_ERROR("DataType not supported");
                break;
        }
        ARM_COMPUTE_ERROR_ON(_prepare_b == nullptr);

        _block_sizes = _prepare_b->block_sizes();
    }

    _block_walker.set(Window::DimX, Window::Dimension(0, ceil_to_multiple(_params.N, _block_sizes.x_block), _block_sizes.x_block));
    _block_walker.set(Window::DimY, Window::Dimension(0, ceil_to_multiple(_params.K, _block_sizes.k_block), _block_sizes.k_block));
    _block_walker.set(Window::DimZ, Window::Dimension(0, _params.multis));

    _batch_window.set(Window::DimX, Window::Dimension(0, ceil_to_multiple(_block_sizes.m_round, _block_sizes.strategy_out_height), _block_sizes.strategy_out_height));
    _batch_window.set(Window::DimY, Window::Dimension(0, _params.batches));

    _transformed_a.allocator()->init(TensorInfo(TensorShape{ _block_sizes.k_block, _block_sizes.m_round, _params.batches }, 1, input_type), alignment);
    _memory_group.manage(&_transformed_a);
    _memory_group.manage(&_tmp_c);

    switch(input_type)
    {
        case DataType::F32:
            _transform_a     = instantiate_transformA<float>(_a, &_transformed_a, _block_walker, _params);
            _matrix_multiply = instantiate_matrix_multiply<float, float>(&_transformed_a, &_transformed_b, &_tmp_c, c, _block_walker, _block_sizes, _params, pretranspose_b, alpha, beta);
            break;
#ifdef __aarch64__
        case DataType::U8:
        case DataType::QASYMM8:
            if(use_dot)
            {
                _transform_a     = instantiate_transformA<uint8_t, true>(_a, &_transformed_a, _block_walker, _params);
                _matrix_multiply = instantiate_matrix_multiply<uint8_t, uint32_t, true>(&_transformed_a, &_transformed_b, &_tmp_c, c, _block_walker, _block_sizes, _params, pretranspose_b, alpha, beta);
            }
            else
            {
                _transform_a     = instantiate_transformA<uint8_t, false>(_a, &_transformed_a, _block_walker, _params);
                _matrix_multiply = instantiate_matrix_multiply<uint8_t, uint32_t, false>(&_transformed_a, &_transformed_b, &_tmp_c, c, _block_walker, _block_sizes, _params, pretranspose_b, alpha, beta);
            }
            break;
        case DataType::S8:
            if(use_dot)
            {
                _transform_a     = instantiate_transformA<int8_t, true>(_a, &_transformed_a, _block_walker, _params);
                _matrix_multiply = instantiate_matrix_multiply<int8_t, int32_t, true>(&_transformed_a, &_transformed_b, &_tmp_c, c, _block_walker, _block_sizes, _params, pretranspose_b, alpha, beta);
            }
            else
            {
                _transform_a     = instantiate_transformA<int8_t, false>(_a, &_transformed_a, _block_walker, _params);
                _matrix_multiply = instantiate_matrix_multiply<int8_t, int32_t, false>(&_transformed_a, &_transformed_b, &_tmp_c, c, _block_walker, _block_sizes, _params, pretranspose_b, alpha, beta);
            }
            break;
#endif /* __aarch64__ */
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F16:
            _transform_a     = instantiate_transformA<__fp16>(_a, &_transformed_a, _block_walker, _params);
            _matrix_multiply = instantiate_matrix_multiply<__fp16, __fp16>(&_transformed_a, &_transformed_b, &_tmp_c, c, _block_walker, _block_sizes, _params, pretranspose_b, alpha, beta);
            break;
            break;
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
        default:
            break;
    }
    ARM_COMPUTE_ERROR_ON(_transform_a == nullptr);
    ARM_COMPUTE_ERROR_ON(_matrix_multiply == nullptr);
    _transformed_a.allocator()->allocate();
    _tmp_c.allocator()->allocate();
    if(!_pretranspose_b)
    {
        _transformed_b.allocator()->allocate();
    }
}
} // namespace arm_compute
