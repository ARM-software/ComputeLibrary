/*
 * Copyright (c) 2019-2021 Arm Limited.
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

#pragma once

#include "arm_gemm.hpp"

#include "barrier.hpp"
#include "gemm_implementation.hpp"
#include "quantized.hpp"

namespace arm_gemm {

/* Quantized wrapper - do an integer GEMM and wrap around the quantization. */

template<typename To, typename Tr, typename Tgemm>
class QuantizeWrapper : public GemmCommon<To, Tr> {
private:
    UniqueGemmCommon<To, Tgemm>  _subgemm = nullptr;
    int32_t                     *_row_sums = nullptr;
    int32_t                     *_col_sums = nullptr;
    Requantize32                 _params;
    GemmArgs                     _args;
    barrier                      _barrier;

    void *working_space = nullptr;
    bool  arrays_set = false;

    /* We need a subgemm which outputs the 32-bit intermediates - how much space is needed for that? */
    size_t subgemm_output_size() const {
        return (_args._Msize * _args._Nsize * _args._nbatches * _args._nmulti * sizeof(int32_t));
    }

    size_t col_sum_size() const {
        return (_args._Nsize * _args._nmulti * sizeof(int32_t));
    }

    size_t row_sum_size() const {
        return (_args._Msize * _args._nbatches * _args._nmulti * sizeof(int32_t));
    }

    /* Local working space: We need space for the subgemm output (above) and
     * the row sums.  */
    size_t local_working_size() const {
        return subgemm_output_size() + row_sum_size();
    }

    void set_child_arrays() {
        if (working_space == nullptr || arrays_set == false)
            return;

        /* Use the first part of our working space for the subgemm result, pass the operand details straight through. */
        _subgemm->set_arrays(this->_Aptr, this->_lda, this->_A_batch_stride, this->_A_multi_stride,
                             this->_Bptr, this->_ldb,                        this->_B_multi_stride,
                             reinterpret_cast<Tgemm *>(working_space), _args._Nsize, (_args._Nsize * _args._Msize), (_args._Nsize * _args._Msize * _args._nbatches),
                             nullptr, 0);
    }

    void col_sums_pretransposed(const To *B, const int ldb, const int B_multi_stride) {
        for (unsigned int multi=0; multi<_args._nmulti; multi++) {
            compute_col_sums(_params, _args._Nsize, _args._Ksize, B + (multi * B_multi_stride), ldb, _col_sums + (multi * _args._Nsize), _args._Ksize, multi, 0);
        }
    }

    void requantize_runtime(unsigned int threadid) {
        unsigned int first_row = (threadid * _args._Msize) / _args._maxthreads;
        unsigned int last_row = ((threadid+1) * _args._Msize) / _args._maxthreads;

        for (unsigned int multi=0; multi<_args._nmulti; multi++) {
            for (unsigned int batch=0; batch<_args._nbatches; batch++) {
                /* Compute row sums now */
                compute_row_sums(_params, _args._Ksize, (last_row - first_row), this->_Aptr + (multi * this->_A_multi_stride) + (batch * this->_A_batch_stride) + (first_row * this->_lda),
                                           this->_lda, _row_sums + (multi * _args._nbatches * _args._Msize) + (batch * _args._Msize) + first_row);
                // If we don't care about negative values, call the version of this function that doesn't correct before shifting.
                // 'c_offset' represents zero, so if the lowest possible quantized output value is the same or more than that we will not output negative numbers.
                requantize_block_32(_params, _args._Nsize, (last_row - first_row),
                                    reinterpret_cast<Tgemm *>(working_space) + (multi * (_args._Msize * _args._Nsize * _args._nbatches)) + (batch * (_args._Msize * _args._Nsize)) + (first_row * _args._Nsize),
                                    _args._Nsize,
                                    this->_Cptr + (multi * this->_C_multi_stride) + (batch * this->_C_batch_stride) + (first_row * this->_ldc), this->_ldc,
                                    _row_sums + (multi * _args._nbatches * _args._Msize) + (batch * _args._Msize) + first_row,
                                    _col_sums + (multi * _args._Nsize), 0);
            }
        }
    }


public:
    QuantizeWrapper(const QuantizeWrapper &) = delete;
    QuantizeWrapper operator=(const QuantizeWrapper &) = delete;

    QuantizeWrapper(const GemmArgs &args, const Requantize32 &qp) : _params(qp), _args(args), _barrier(args._maxthreads) {
        GemmArgs newargs = GemmArgs(args._ci, args._Msize, args._Nsize, args._Ksize, args._Ksections, args._nbatches, args._nmulti, args._indirect_input, Activation(), args._maxthreads);
        _subgemm = gemm<To, Tgemm>(newargs);

        if (_subgemm == nullptr) {
            return;
        }
    }

    void set_arrays(const To *A, const int lda, const int A_batch_stride, const int A_multi_stride,
                    const To *B, const int ldb, const int B_multi_stride,
                          Tr *C, const int ldc, const int C_batch_stride, const int C_multi_stride,
                    const Tr *bias, const int bias_multi_stride) override {
        GemmCommon<To, Tr>::set_arrays(A, lda, A_batch_stride, A_multi_stride, B, ldb, B_multi_stride, C, ldc, C_batch_stride, C_multi_stride, bias, bias_multi_stride);

        arrays_set = true;
        set_child_arrays();
    }

    ndrange_t get_window_size() const override {
        return { _subgemm->get_window_size() };
    }

    void set_nthreads(int nthreads) override {
        _subgemm->set_nthreads(nthreads);
        _barrier.set_nthreads(nthreads);
        _args._maxthreads = nthreads;
    }

    void execute(const ndcoord_t &work_range, const ndcoord_t &thread_locator, int threadid) override {
        _subgemm->execute(work_range, thread_locator, threadid);

        _barrier.arrive_and_wait();

        requantize_runtime(threadid);
    }

    size_t get_working_size() const override {
        return _subgemm->get_working_size() + local_working_size();
    }

    // Space arrangement:

    // ptr
    // V
    // | subgemm output | row_sums | subgemm working space |
    void set_working_space(void *space) override {
        uintptr_t space_int = reinterpret_cast<uintptr_t>(space);

        working_space = space;
        _subgemm->set_working_space(reinterpret_cast<void *>(space_int + local_working_size()));

        _row_sums = reinterpret_cast<int32_t *>(space_int + subgemm_output_size());

        set_child_arrays();
    }

    bool B_is_pretransposed() const override {
        /* We clear this flag if the subgemm isn't pretransposed, so just return its value */
        return _subgemm->B_is_pretransposed();
    }

    bool B_pretranspose_required() const override {
        return _subgemm->B_pretranspose_required();
    }

    size_t get_B_pretransposed_array_size() const override {
        return _subgemm->get_B_pretransposed_array_size() + col_sum_size();
    }

    void requantize_bias(void *in_buffer, const To *B, const int ldb, const int B_multi_stride) override {
        _col_sums = reinterpret_cast<int32_t *>(in_buffer);
        col_sums_pretransposed(B, ldb, B_multi_stride);
    }

    void pretranspose_B_array(void *buffer, const To *B, const int ldb, const int B_multi_stride) override {
        uintptr_t buffer_int = reinterpret_cast<uintptr_t>(buffer);
        _subgemm->pretranspose_B_array(reinterpret_cast<void *>(buffer_int + col_sum_size()), B, ldb, B_multi_stride);

        requantize_bias(buffer, B, ldb, B_multi_stride);
    }

    void set_pretransposed_B_data(void *buffer) override {
        uintptr_t buffer_int = reinterpret_cast<uintptr_t>(buffer);
        _subgemm->set_pretransposed_B_data(reinterpret_cast<void *>(buffer_int + col_sum_size()));
        _col_sums = reinterpret_cast<int32_t *>(buffer);
    }

    void set_quantized_bias(const int32_t *bias, size_t bias_multi_stride) override {
        _params.bias = bias;
        _params.bias_multi_stride = bias_multi_stride;
    }

    GemmConfig get_config() override {
        GemmConfig c = _subgemm->get_config();

        std::string n = "quantize_wrapper[";
        n.append(c.filter);
        n.append("]");

        c.method = GemmMethod::QUANTIZE_WRAPPER;
        c.filter = n;

        return c;
    }
};

} // namespace arm_gemm
