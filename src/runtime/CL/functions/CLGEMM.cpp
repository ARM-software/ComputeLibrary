/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#include "arm_compute/runtime/CL/functions/CLGEMM.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/GPUTarget.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/gemm_reshaped/CLGEMMReshapedConfiguration.h"
#include "arm_compute/runtime/ITensorAllocator.h"

namespace arm_compute
{
using namespace arm_compute::misc::shape_calculator;
using namespace arm_compute::cl_gemm;

namespace
{
inline bool is_interleaved_transposed(unsigned int m, unsigned int n, unsigned int k, DataType data_type, bool reshape_b_only_on_first_run, GPUTarget gpu_target)
{
    bool flag = true;

    if(gpu_target_is_in(gpu_target, GPUTarget::G52, GPUTarget::G52LIT, GPUTarget::G71, GPUTarget::G72, GPUTarget::G76))
    {
        if((m > 1) && n < 16)
        {
            flag = true;
        }
        else
        {
            // COMPMID-852
            if(k > 256 && m > 4 && is_data_type_float(data_type) && reshape_b_only_on_first_run)
            {
                constexpr float alpha = 3.2f;
                constexpr float fact0 = 1.51f;
                constexpr float fact1 = 1.66f;
                constexpr float ops   = 12.0f;
                const float     scale = k > 1024 ? 1.07f : 1.0f;
                flag                  = alpha + ((n * fact0) / ops) < ((fact1 * n * scale) / ops);
            }
            else
            {
                flag = false;
            }
        }
    }
    else
    {
        // We reshape the matrices only if we do not have the vector-by-matrix case and we reshape the matrix B only once
        flag = m != 1 && reshape_b_only_on_first_run;
    }

    return flag;
}
} // namespace

CLGEMM::CLGEMM(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)),
      _mm_kernel(),
      _ma_kernel(),
      _reshape_lhs_kernel(),
      _reshape_rhs_kernel(),
      _mm_reshaped_kernel(),
      _tmp_a(),
      _tmp_b(),
      _original_b(nullptr),
      _is_interleaved_transposed(false),
      _run_addition(false),
      _reshape_b_only_on_first_run(false),
      _is_prepared(false),
      _is_new_gemm_reshaped(false)
{
}

void CLGEMM::configure(const ICLTensor *a, const ICLTensor *b, const ICLTensor *c, ICLTensor *output, float alpha, float beta, const GEMMInfo &gemm_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(a, b, output);

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate(a->info(), b->info(), c != nullptr ? c->info() : nullptr, output->info(), alpha, beta, gemm_info));

    // Check if we need to reshape the matrix B only on the first run
    _reshape_b_only_on_first_run = gemm_info.reshape_b_only_on_first_run();
    _is_prepared                 = gemm_info.retain_internal_weights();
    _original_b                  = b;

    const ICLTensor *matrix_a = a;
    const ICLTensor *matrix_b = b;

    // Get the GPU target
    const GPUTarget gpu_target = CLScheduler::get().target();

    // Set the target for the kernels
    _reshape_lhs_kernel.set_target(gpu_target);
    _mm_kernel.set_target(gpu_target);

    // Arguments used by GEMMReshapeInfo
    // If we pass the matrix A and matrix B reshaped to CLGEMMMatrixMultiplyKernel, we need to pass m, n, k, mult_transpose1xW_width and mult_interleave4x4_height to CLGEMMReshapeInfo
    // in order to know how the matrices have been reshaped
    DataType           data_type                 = a->info()->data_type();
    bool               reinterpret_input_as_3d   = gemm_info.reinterpret_input_as_3d();
    const unsigned int m                         = reinterpret_input_as_3d ? (a->info()->dimension(1) * a->info()->dimension(2)) : a->info()->dimension(1);
    const unsigned int n                         = b->info()->dimension(0);
    const unsigned int k                         = a->info()->dimension(0);
    const unsigned int batch_size                = reinterpret_input_as_3d ? a->info()->dimension(3) : a->info()->dimension(2);
    const int          depth_output_gemm3d       = gemm_info.depth_output_gemm3d();
    int                mult_transpose1xW_width   = 1;
    int                mult_interleave4x4_height = 1;

    if(get_arch_from_target(gpu_target) == GPUTarget::BIFROST)
    {
        mult_transpose1xW_width   = 4;
        mult_interleave4x4_height = 2;
    }
    GEMMRHSMatrixInfo rhs_info;
    rhs_info.n0         = 16 / b->info()->element_size();
    rhs_info.k0         = 1;
    rhs_info.h0         = mult_transpose1xW_width;
    rhs_info.interleave = false;
    rhs_info.transpose  = false;

    GEMMLHSMatrixInfo lhs_info;
    lhs_info.m0         = 4;
    lhs_info.k0         = 4;
    lhs_info.v0         = mult_interleave4x4_height;
    lhs_info.interleave = true;
    lhs_info.transpose  = true;

    // Check if we need to reshape the matrix A and matrix B
    _is_interleaved_transposed = is_interleaved_transposed(m, n, k, a->info()->data_type(), _reshape_b_only_on_first_run, gpu_target);

    // Check if we can run the new reshaped GEMM
    const auto workload   = static_cast<float>((m * n) / 20.0f);
    _is_new_gemm_reshaped = (workload > 1600.0f) && (get_arch_from_target(gpu_target) == GPUTarget::BIFROST) && _is_interleaved_transposed && (data_type == DataType::F32);

    const bool add_matrix_c  = (beta != 0.f && c != nullptr);
    const bool is_beta_one   = std::abs(1.0f - beta) < 0.00001f;
    const bool use_fused_add = is_beta_one && (c != nullptr && c->info()->num_dimensions() == 1) && !_is_new_gemm_reshaped;

    // if _is_interleaved_transposed is set, force reinterpret_input_as_3d to be false as the output of CLGEMMInterleaveKernel will be 2D
    if(_is_interleaved_transposed)
    {
        reinterpret_input_as_3d = false;

        matrix_a = &_tmp_a;
        matrix_b = &_tmp_b;

        // Manage intermediate buffers
        _memory_group.manage(&_tmp_a);
        if(!_reshape_b_only_on_first_run)
        {
            _memory_group.manage(&_tmp_b);
        }
        // _tmp_a and _tmp_b will be auto configured in _interleave_kernel and in _transpose_kernel

        if(_is_new_gemm_reshaped)
        {
            GEMMLHSMatrixInfo lhs_info;

            // Pick up the GEMM configuration
            std::tie(lhs_info, rhs_info) = CLGEMMReshapedConfigurationFactory::create()->configure(m, n, k, batch_size, data_type);

            _reshape_lhs_kernel.configure(a, &_tmp_a, lhs_info, gemm_info.reinterpret_input_as_3d());
            _reshape_rhs_kernel.configure(b, &_tmp_b, rhs_info);

            // Configure and tune matrix multiply kernel
            _mm_reshaped_kernel.configure(matrix_a, matrix_b, output, alpha, lhs_info, rhs_info, GEMMReshapeInfo(m, n, k, 1, 1,
                                                                                                                 depth_output_gemm3d, reinterpret_input_as_3d));
        }
        else
        {
            // Configure interleave kernel
            _reshape_lhs_kernel.configure(a, &_tmp_a, lhs_info, gemm_info.reinterpret_input_as_3d());
            // Configure transpose kernel
            _reshape_rhs_kernel.configure(b, &_tmp_b, rhs_info);
        }
    }

    if(!_is_new_gemm_reshaped)
    {
        // Configure and tune matrix multiply kernel
        _mm_kernel.configure(matrix_a, matrix_b, (add_matrix_c && !use_fused_add) ? nullptr : c, output, alpha, beta, _is_interleaved_transposed,
                             GEMMReshapeInfo(m, n, k, mult_transpose1xW_width, mult_interleave4x4_height, depth_output_gemm3d, reinterpret_input_as_3d),
                             gemm_info.fp_mixed_precision());
        CLScheduler::get().tune_kernel_static(_mm_kernel);
    }

    if(_is_interleaved_transposed)
    {
        // Allocate intermediate tensors
        _tmp_a.allocator()->allocate();
        if(!_reshape_b_only_on_first_run)
        {
            _tmp_b.allocator()->allocate();
        }
    }

    // Configure matrix addition kernel
    if(add_matrix_c && !use_fused_add)
    {
        _ma_kernel.configure(c, output, beta);
        _run_addition = true;
    }
}

Status CLGEMM::validate(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, const ITensorInfo *output, float alpha, float beta, const GEMMInfo &gemm_info)
{
    ARM_COMPUTE_UNUSED(alpha);
    ARM_COMPUTE_UNUSED(output);

    // Check if we need to reshape the matrix B only on the first run
    const bool reshape_b_only_on_first_run = gemm_info.reshape_b_only_on_first_run();

    const ITensorInfo *matrix_a_info = a;
    const ITensorInfo *matrix_b_info = b;

    TensorInfo tmp_a_info{};
    TensorInfo tmp_b_info{};

    // Get the GPU target
    const GPUTarget gpu_target = CLScheduler::get().target();

    // Arguments used by GEMMReshapeInfo
    // If we pass the matrix A and matrix B reshaped to CLGEMMMatrixMultiplyKernel, we need to pass m, n, k, mult_transpose1xW_width and mult_interleave4x4_height to CLGEMMReshapeInfo
    // in order to know how the matrices have been reshaped
    DataType           data_type                 = a->data_type();
    bool               reinterpret_input_as_3d   = gemm_info.reinterpret_input_as_3d();
    const unsigned int m                         = reinterpret_input_as_3d ? (a->dimension(1) * a->dimension(2)) : a->dimension(1);
    const unsigned int n                         = b->dimension(0);
    const unsigned int k                         = a->dimension(0);
    const unsigned int batch_size                = reinterpret_input_as_3d ? a->dimension(3) : a->dimension(2);
    int                mult_transpose1xW_width   = 1;
    int                mult_interleave4x4_height = 1;
    const int          depth_output_gemm3d       = gemm_info.depth_output_gemm3d();

    if(get_arch_from_target(gpu_target) == GPUTarget::BIFROST)
    {
        mult_transpose1xW_width   = 4;
        mult_interleave4x4_height = 2;
    }

    GEMMRHSMatrixInfo rhs_info;
    rhs_info.n0         = 16 / b->element_size();
    rhs_info.k0         = 1;
    rhs_info.h0         = mult_transpose1xW_width;
    rhs_info.interleave = false;
    rhs_info.transpose  = false;

    GEMMLHSMatrixInfo lhs_info;
    lhs_info.m0         = 4;
    lhs_info.k0         = 4;
    lhs_info.v0         = mult_interleave4x4_height;
    lhs_info.interleave = true;
    lhs_info.transpose  = true;

    // Check if we need to reshape the matrix A and matrix B
    const bool run_interleave_transpose = is_interleaved_transposed(m, n, k, a->data_type(), reshape_b_only_on_first_run, gpu_target);

    // Check if we can run the new reshaped GEMM
    const auto workload             = static_cast<float>((m * n) / 20.0f);
    const bool is_new_gemm_reshaped = (workload > 1600.f) && (get_arch_from_target(gpu_target) == GPUTarget::BIFROST) && run_interleave_transpose && (data_type == DataType::F32);

    const bool add_matrix_c  = (beta != 0.f && c != nullptr);
    const bool is_beta_one   = std::abs(1.0f - beta) < 0.00001f;
    const bool use_fused_add = is_beta_one && (c != nullptr && c->num_dimensions() == 1) && !is_new_gemm_reshaped;

    // if _is_interleaved_transposed is set, force reinterpret_input_as_3d to be false as the output of CLGEMMInterleaveKernel will be 2D
    if(run_interleave_transpose)
    {
        reinterpret_input_as_3d = false;
    }

    const GEMMReshapeInfo reshape_info = GEMMReshapeInfo(m, n, k, mult_transpose1xW_width, mult_interleave4x4_height, depth_output_gemm3d, reinterpret_input_as_3d);

    if(run_interleave_transpose)
    {
        matrix_a_info = &tmp_a_info;
        matrix_b_info = &tmp_b_info;

        if(is_new_gemm_reshaped)
        {
            GEMMLHSMatrixInfo lhs_info;

            // Pick up the GEMM configuration
            std::tie(lhs_info, rhs_info) = CLGEMMReshapedConfigurationFactory::create()->configure(m, n, k, batch_size, data_type);

            auto_init_if_empty(tmp_a_info, a->clone()->set_tensor_shape(compute_lhs_reshaped_shape(*a, lhs_info, gemm_info.reinterpret_input_as_3d())));
            ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMReshapeLHSMatrixKernel::validate(a, &tmp_a_info, lhs_info, gemm_info.reinterpret_input_as_3d()));

            auto_init_if_empty(tmp_b_info, b->clone()->set_tensor_shape(compute_rhs_reshaped_shape(*b, rhs_info)));
            ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMReshapeRHSMatrixKernel::validate(b, &tmp_b_info, rhs_info));

            // Validate matrix multiply
            ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMMatrixMultiplyReshapedKernel::validate(matrix_a_info, matrix_b_info, output, alpha, lhs_info, rhs_info, GEMMReshapeInfo(m, n, k, 1, 1,
                                                                                     depth_output_gemm3d, reinterpret_input_as_3d)));
        }
        else
        {
            // Validate interleave kernel
            auto_init_if_empty(tmp_a_info, a->clone()->set_tensor_shape(compute_lhs_reshaped_shape(*a, lhs_info, gemm_info.reinterpret_input_as_3d())));
            ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMReshapeLHSMatrixKernel::validate(a, &tmp_a_info, lhs_info, gemm_info.reinterpret_input_as_3d()));
            // Validate transpose kernel
            auto_init_if_empty(tmp_b_info, b->clone()->set_tensor_shape(compute_rhs_reshaped_shape(*b, rhs_info)));
            ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMReshapeRHSMatrixKernel::validate(b, &tmp_b_info, rhs_info));
        }
    }

    if(!is_new_gemm_reshaped)
    {
        // Validate matrix multiply
        ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMMatrixMultiplyKernel::validate(matrix_a_info, matrix_b_info, (add_matrix_c && !use_fused_add) ? nullptr : c, output, alpha, beta,
                                                                         run_interleave_transpose, reshape_info, gpu_target, gemm_info.fp_mixed_precision()));
    }

    if(add_matrix_c && !use_fused_add)
    {
        // Validate matrix addition kernel
        ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMMatrixAdditionKernel::validate(c, output, beta));
    }

    return Status{};
}

void CLGEMM::run()
{
    prepare();

    _memory_group.acquire();

    if(_is_interleaved_transposed)
    {
        // Run interleave kernel
        CLScheduler::get().enqueue(_reshape_lhs_kernel, false);

        if(!_reshape_b_only_on_first_run)
        {
            // Run transpose kernel
            CLScheduler::get().enqueue(_reshape_rhs_kernel, false);
        }
    }

    // Run matrix multiply kernel
    if(_is_new_gemm_reshaped)
    {
        CLScheduler::get().enqueue(_mm_reshaped_kernel, !_run_addition);
    }
    else
    {
        CLScheduler::get().enqueue(_mm_kernel, !_run_addition);
    }

    // Run matrix addition kernel
    if(_run_addition)
    {
        CLScheduler::get().enqueue(_ma_kernel);
    }

    _memory_group.release();
}

void CLGEMM::prepare()
{
    if(!_is_prepared)
    {
        if(_is_interleaved_transposed && _reshape_b_only_on_first_run)
        {
            // Run transpose kernel and mark original weights tensor as unused
            _tmp_b.allocator()->allocate();
            CLScheduler::get().enqueue(_reshape_rhs_kernel, false);
            _original_b->mark_as_unused();
        }
        CLScheduler::get().queue().finish();
        _is_prepared = true;
    }
}
} // namespace arm_compute
