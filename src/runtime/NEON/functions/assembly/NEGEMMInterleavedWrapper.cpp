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

#include <atomic>
#include <condition_variable>
#include <mutex>

namespace arm_compute
{
#ifndef NO_MULTI_THREADING
class BufferManagerMultipleThreads final : public IBufferManager
{
public:
    /** Number of buffers to ping pong between */
    static constexpr unsigned int NUM_BUFFERS = 3;

    explicit BufferManagerMultipleThreads(unsigned int max_num_users)
        : _max_num_users(max_num_users)
    {
    }
    unsigned int num_buffers() const override
    {
        return NUM_BUFFERS;
    }
    /* - Lock the requested index if it's free and return true if it needs reshaping.
     * - Return false without acquiring the lock if the buffer at the index is already reshaped / being reshaped.
     * - Block if the corresponding buffer for the given index is still being used by a different index.
     */
    bool lock_to_reshape_if_needed(unsigned int index) override
    {
        Buffer &buf = get_buffer_from_index(index);
        while(true)
        {
            if(buf.index == index && buf.state != State::FREE)
            {
                //Another thread already is reshaping / has reshaped this block: nothing to do
                return false;
            }
            else
            {
                std::unique_lock<std::mutex> lock(buf.mutex);
                //If the buffer is free then lock it for reshaping:
                if(buf.state == State::FREE)
                {
                    buf.index = index;
                    buf.state = State::BEING_RESHAPED;
                    return true;
                }
                // Check again just in case it changed while we were acquiring the lock:
                if(buf.index == index)
                {
                    //Another thread is reshaping this block already, nothing to do
                    return false;
                }
                // buf.index != index: Buffer still being used by another block, need to wait
                buf.sem.wait(lock);
            }
        }
    }
    /* Mark the buffer at the given index as reshaped and release the lock acquired via lock_to_reshape_if_needed() */
    void mark_as_reshaped(unsigned int index) override
    {
        Buffer &buf = get_buffer_from_index(index);
        {
            std::lock_guard<std::mutex> lock(buf.mutex);
            buf.users = _max_num_users;
            buf.state = State::IN_USE;
        }
        buf.sem.notify_all();
    }

    /* Block until the buffer at the given index is reshaped */
    void wait_for_reshaping(unsigned int index) override
    {
        Buffer &buf = get_buffer_from_index(index);
        ARM_COMPUTE_ERROR_ON(buf.index != index); // Should have blocked in lock_to_reshape_if_needed()
        // Check if it's already ready to use:
        if(buf.state == State::IN_USE)
            return;
        std::unique_lock<std::mutex> lock(buf.mutex);
        //Double check it didn't change while we were acquiring the lock:
        if(buf.state == State::IN_USE)
            return;
        buf.sem.wait(lock);
    }
    /* Mark the buffer at the given index as not used by this thread anymore.
     * Once all the threads have called this method then the buffer is marked as free again.
     */
    void mark_as_unused(unsigned int index) override
    {
        Buffer &buf = get_buffer_from_index(index);
        ARM_COMPUTE_ERROR_ON(buf.index != index); // Should have blocked in lock_to_reshape_if_needed()
        if(--buf.users == 0)
        {
            std::unique_lock<std::mutex> lock(buf.mutex);
            buf.state = State::FREE;
            lock.unlock();
            buf.sem.notify_all();
        }
    }

private:
    enum class State
    {
        FREE,
        BEING_RESHAPED,
        IN_USE
    };
    struct Buffer
    {
        unsigned int            index{};
        std::atomic_uint        users{};
        State                   state{ State::FREE };
        std::mutex              mutex{};
        std::condition_variable sem{};
    } _buffers[NUM_BUFFERS];
    Buffer &get_buffer_from_index(unsigned int index)
    {
        return _buffers[index % NUM_BUFFERS];
    }
    unsigned int _max_num_users;
};
#endif /* NO_MULTI_THREADING */

class BufferManagerSingleThread : public IBufferManager
{
public:
    unsigned int num_buffers() const override
    {
        return 1;
    }
    bool lock_to_reshape_if_needed(unsigned int index) override
    {
        return true;
    }
    void mark_as_reshaped(unsigned int index) override
    {
    }
    void wait_for_reshaping(unsigned int index) override
    {
    }
    void mark_as_unused(unsigned int index) override
    {
    }
};

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
        const unsigned int num_x_blocks = _block_walker.num_iterations(Window::DimX);

        // Create a 1D window to dynamically split the batch window:
        Window win_1D;
        win_1D.set(0, Window::Dimension(0, num_iterations));

        // Create one workload for each sub-window:
        for(unsigned int w = 0; w < num_windows; w++)
        {
            Window            win          = win_1D.split_window(0, w, num_windows);
            const Coordinates start_offset = index2coords(window_shape, win.x().start());
            const Coordinates end_offset   = index2coords(window_shape, win.x().end() - 1);

            if(_pretranspose_b)
            {
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
            else
            {
                auto workload = [num_threads, start_offset, end_offset, num_x_blocks, this](const ThreadInfo & info)
                {
                    //For each block of rows in "M"
                    auto         workload_mm = this->_mm_workloads.begin();
                    unsigned int workload_b  = 0;
                    //If there is only one thread then only reshape the B blocks as you need them:
                    unsigned int workload_b_next = num_threads == 1 ? this->_b_workloads.size() : 1;

                    for(auto workload_a = this->_a_workloads.begin(); workload_a != this->_a_workloads.end(); workload_a++)
                    {
                        // Transform one k_block from A:
                        this->_transform_a->transform(*workload_a, info, this->_batch_window, start_offset, end_offset);
                        // Then perform the matrix multiplication for each x block along N:
                        for(unsigned int i = 0; i < num_x_blocks; i++)
                        {
                            ARM_COMPUTE_ERROR_ON(workload_mm == this->_mm_workloads.end());
                            if(workload_b_next < this->_b_workloads.size())
                            {
                                //Lock on BufferManager: need to run it ?
                                if(this->_buffer_manager->lock_to_reshape_if_needed(workload_b_next))
                                {
                                    this->_prepare_b->transform(this->_b_workloads[workload_b_next], info);
                                    this->_buffer_manager->mark_as_reshaped(workload_b_next);
                                }
                                workload_b_next++;
                            }
                            ARM_COMPUTE_ERROR_ON(workload_b >= this->_b_workloads.size());
                            // Run if needed or wait
                            if(this->_buffer_manager->lock_to_reshape_if_needed(workload_b))
                            {
                                this->_prepare_b->transform(this->_b_workloads[workload_b], info);
                                this->_buffer_manager->mark_as_reshaped(workload_b);
                            }
                            this->_buffer_manager->wait_for_reshaping(workload_b);
                            this->_matrix_multiply->transform(*workload_mm++, info, this->_batch_window, start_offset, end_offset);
                            this->_buffer_manager->mark_as_unused(workload_b);
                            workload_b++;
                        }
                    }
                };
                _workloads.push_back(workload);
            }
        }
        if(!_pretranspose_b && num_windows > 1 && num_windows % num_threads != 0)
        {
            //Make sure the number of workloads is a multiple of the number of threads to avoid dead locks:
            for(unsigned int leftover = num_windows % num_threads; leftover != num_threads; leftover++)
            {
                auto workload = [this](const ThreadInfo & info)
                {
                    unsigned int workload_b = 0;
                    //If there is only one thread then only reshape the B blocks as you need them:
                    unsigned int workload_b_next = 1;

                    for(unsigned int iteration = 0; iteration < this->_mm_workloads.size(); iteration++)
                    {
                        if(workload_b_next < this->_b_workloads.size())
                        {
                            //Lock on BufferManager: need to run it ?
                            if(this->_buffer_manager->lock_to_reshape_if_needed(workload_b_next))
                            {
                                this->_prepare_b->transform(this->_b_workloads[workload_b_next], info);
                                this->_buffer_manager->mark_as_reshaped(workload_b_next);
                            }
                            workload_b_next++;
                        }
                        ARM_COMPUTE_ERROR_ON(workload_b >= this->_b_workloads.size());
                        // Run if needed or wait
                        if(this->_buffer_manager->lock_to_reshape_if_needed(workload_b))
                        {
                            this->_prepare_b->transform(this->_b_workloads[workload_b], info);
                            this->_buffer_manager->mark_as_reshaped(workload_b);
                        }
                        this->_buffer_manager->wait_for_reshaping(workload_b);
                        this->_buffer_manager->mark_as_unused(workload_b);
                        workload_b++;
                    }
                };
                _workloads.push_back(workload);
            }
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
        _block_sizes = calculate_block_sizes_from_data_type(NEScheduler::get().cpu_info(), _params.M, _params.N, _params.K, input_type, use_dot);
        _batch_window.set(Window::DimX, Window::Dimension(0, ceil_to_multiple(_block_sizes.m_round, _block_sizes.strategy_out_height), _block_sizes.strategy_out_height));
        _batch_window.set(Window::DimY, Window::Dimension(0, _params.batches));
        // If the execution is single threaded or has only one window then the buffer manager only needs 1 buffer else we will use NUM_BUFFERS buffers and ping pong between them:
        const unsigned int num_iterations = _batch_window.num_iterations_total();
        if(NEScheduler::get().num_threads() == 1 || num_iterations == 1)
        {
            _buffer_manager = support::cpp14::make_unique<BufferManagerSingleThread>();
        }
        else
        {
#ifdef NO_MULTI_THREADING
            ARM_COMPUTE_ERROR("Can't have more than 1 buffer without multiple threads");
#else  /* NO_MULTI_THREADING */
            _buffer_manager = support::cpp14::make_unique<BufferManagerMultipleThreads>(NEScheduler::get().num_threads());
#endif /* NO_MULTI_THREADING */
        }
        // If B is transposed at every iteration then transformed_B can be managed:
        _memory_group.manage(&_transformed_b);
        auto_init_if_empty(*_transformed_b.info(), _b->info()->clone()->set_tensor_shape(TensorShape(_block_sizes.x_block * _block_sizes.k_block, _buffer_manager->num_buffers())));
    }
    else
    {
        _tag += "_preB";
    }
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

    if(_pretranspose_b)
    {
        _block_sizes = _prepare_b->block_sizes();
        _batch_window.set(Window::DimX, Window::Dimension(0, ceil_to_multiple(_block_sizes.m_round, _block_sizes.strategy_out_height), _block_sizes.strategy_out_height));
        _batch_window.set(Window::DimY, Window::Dimension(0, _params.batches));
    }

    _block_walker.set(Window::DimX, Window::Dimension(0, ceil_to_multiple(_params.N, _block_sizes.x_block), _block_sizes.x_block));
    _block_walker.set(Window::DimY, Window::Dimension(0, ceil_to_multiple(_params.K, _block_sizes.k_block), _block_sizes.k_block));
    _block_walker.set(Window::DimZ, Window::Dimension(0, _params.multis));

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
