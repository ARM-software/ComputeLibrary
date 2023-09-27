/*
 * Copyright (c) 2021-2023 Arm Limited.
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
#include "src/cpu/operators/CpuGemm.h"

#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

#include "src/common/utils/Log.h"
#include "src/core/CPP/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/MemoryHelpers.h"
#include "src/cpu/utils/CpuAuxTensorHandler.h"

using namespace arm_compute::experimental;
using namespace arm_compute::misc::shape_calculator;

namespace arm_compute
{
namespace cpu
{
namespace
{
cpu::AsmGemmInfo init_assembly_metadata(const GEMMInfo &info)
{
    cpu::AsmGemmInfo asm_info;
    asm_info.method                  = cpu::AsmConvMethod::Im2Col;
    asm_info.reinterpret_input_as_3d = info.reinterpret_input_as_3d();
    asm_info.depth_output_gemm3d     = info.depth_output_gemm3d();
    asm_info.activation_info         = info.activation_info();
    asm_info.fast_mode               = info.fast_math();
    asm_info.fixed_format            = info.fixed_format();
    asm_info.weight_format           = info.weight_format();

    return asm_info;
}
} // namespace

void CpuGemm::configure(const ITensorInfo *a,
                        const ITensorInfo *b,
                        const ITensorInfo *c,
                        ITensorInfo       *d,
                        float              alpha,
                        float              beta,
                        const GEMMInfo    &gemm_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(a, b, d);
    ARM_COMPUTE_ERROR_THROW_ON(CpuGemm::validate(a, b, c, d, alpha, beta, gemm_info));
    ARM_COMPUTE_LOG_PARAMS(a, b, c, d, alpha, beta, gemm_info);

    const cpu::AsmGemmInfo asm_info  = init_assembly_metadata(gemm_info);
    const bool             is_c_bias = beta == 1 && c != nullptr;
    bool                   run_optimised =
        bool(cpu::CpuGemmAssemblyDispatch::validate(a, b, (is_c_bias) ? c : nullptr, d, asm_info)) &&
        (c == nullptr || beta == 0.f || beta == 1.f) && // Optimized GeMM doesn't support beta coefficient.
        !(!b->are_values_constant() &&
          b->tensor_shape().z() > 1); // Disable batch matmul as optimized GeMM handles batching differently.

    // Check if we need to reshape the matrix B only on the first run
    _is_prepared                      = false;
    _reshape_b_only_on_first_run      = b->are_values_constant();
    _run_vector_matrix_multiplication = a->dimension(1) < 2;
    _run_alpha_scale                  = alpha != 1.f;
    _run_bias_addition                = is_c_bias;
    _run_addition                     = beta != 0 && beta != 1 && c != nullptr;
    _run_activation =
        gemm_info.activation_info().enabled() &&
        (!run_optimised ||
         (run_optimised && !cpu::CpuGemmAssemblyDispatch::is_activation_supported(gemm_info.activation_info())));

    if (run_optimised)
    {
        const ITensorInfo *c_to_use = is_c_bias ? c : nullptr;
        _asm_glue                   = std::make_unique<cpu::CpuGemmAssemblyDispatch>();
        _asm_glue->configure(a, b, c_to_use, d, asm_info);
        ARM_COMPUTE_ERROR_ON(!_asm_glue->is_configured());

        auto asm_mem_req           = _asm_glue->workspace();
        _aux_mem[AsmGemmWorkspace] = asm_mem_req[AsmGemmWorkspace];
        _aux_mem[Pretraspose]      = asm_mem_req[Pretraspose];

        // Scale product by alpha
        if (_run_alpha_scale)
        {
            _alpha_scale_func = std::make_unique<cpu::CpuActivation>();
            _alpha_scale_func->configure(
                d, nullptr, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LINEAR, alpha, 0.f));
        }
    }
    else
    {
        // Pick output tensor in case bias addition should be performed
        ITensorInfo *gemm_output_to_use = (_run_bias_addition) ? &_tmp_d : d;

        _mm_kernel = std::make_unique<cpu::kernels::CpuGemmMatrixMultiplyKernel>();

        // Select between GEMV and GEMM
        if (_run_vector_matrix_multiplication)
        {
            // Configure the matrix multiply kernel
            _mm_kernel->configure(a, b, gemm_output_to_use, alpha, false);
        }
        else
        {
            const int m = a->dimension(1);
            const int n = b->dimension(0);
            const int k = a->dimension(0);

            // Configure interleave kernel
            _interleave_kernel = std::make_unique<cpu::kernels::CpuGemmInterleave4x4Kernel>();
            _interleave_kernel->configure(a, &_tmp_a);
            _aux_mem[InterleavedLHS] =
                MemoryInfo(offset_int_vec(InterleavedLHS), MemoryLifetime::Temporary, _tmp_a.total_size());

            // Configure transpose kernel
            _transpose_kernel = std::make_unique<cpu::kernels::CpuGemmTranspose1xWKernel>();
            _transpose_kernel->configure(b, &_tmp_b);
            _aux_mem[TransposedRHS] =
                MemoryInfo(offset_int_vec(TransposedRHS), MemoryLifetime::Persistent, _tmp_b.total_size());

            // Configure matrix multiplication kernel
            _mm_kernel->configure(&_tmp_a, &_tmp_b, gemm_output_to_use, alpha, true, GEMMReshapeInfo(m, n, k));
        }

        if (_run_bias_addition)
        {
            _add_bias = std::make_unique<cpu::CpuAdd>();
            _add_bias->configure(gemm_output_to_use, c, d, ConvertPolicy::SATURATE);
            _aux_mem[TempResult] =
                MemoryInfo(offset_int_vec(TempResult), MemoryLifetime::Temporary, _tmp_d.total_size());
        }
    }

    // Configure matrix addition kernel
    if (_run_addition)
    {
        _ma_kernel = std::make_unique<cpu::kernels::CpuGemmMatrixAdditionKernel>();
        _ma_kernel->configure(c, d, beta);
    }

    // Configure activation
    if (_run_activation)
    {
        _activation_func = std::make_unique<cpu::CpuActivation>();
        _activation_func->configure(d, nullptr, gemm_info.activation_info());
    }
}

Status CpuGemm::validate(const ITensorInfo *a,
                         const ITensorInfo *b,
                         const ITensorInfo *c,
                         const ITensorInfo *d,
                         float              alpha,
                         float              beta,
                         const GEMMInfo    &gemm_info)
{
    ARM_COMPUTE_UNUSED(alpha);
    const bool is_c_bias    = beta == 1 && c != nullptr;
    const bool run_addition = c != nullptr && beta != 0 && beta != 1;

    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(a);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_BF16_UNSUPPORTED(a);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(a, 1, DataType::BFLOAT16, DataType::F16, DataType::F32);

    if (is_fixed_format_fast_math(gemm_info.weight_format()))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(a, DataType::F32);
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(b, DataType::BFLOAT16);
    }
    else
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(a, b);
    }

    const int block_by = arm_compute::block_by(gemm_info.weight_format());
    // test if im2col has changed the dimensions that are needed for padding
    if (a->dimension(0) != b->dimension(1) && block_by > 1)
    {
        // have to verify bias
        const size_t dim0_sz = a->dimension(0);
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(
            (dim0_sz % block_by) != 0,
            ("The matrix A number of columns must be a multiple of block_by=" + std::to_string(block_by)).c_str());
        // a->dimension(0) = kernel_area * input_channel + kernel_area * input_pad_right
        // b->dimension(1) = kernel_area * input_channel
        // a->dimension(0) = b->dimension(1) + kernel_area * input_pad_right
        const size_t input_pad_right = (dim0_sz - b->dimension(1)) % block_by;
        const size_t kernel_area     = (dim0_sz - b->dimension(1)) / input_pad_right;
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(
            (dim0_sz - kernel_area * input_pad_right) != b->dimension(1),
            "The product AB is defined only if A number of columns and B number of rows are related");
    }
    else
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(
            a->dimension(0) != b->dimension(1),
            "The product AB is defined only if the number of columns in A is equal to the number of rows in B");
    }

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(gemm_info.is_a_reshaped(), "Matrix A already reshaped is not supported");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(gemm_info.is_b_reshaped(), "Matrix B already reshaped is not supported");
    if (a->data_type() != DataType::BFLOAT16)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(a, d);
    }

    if (run_addition)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(gemm_info.depth_output_gemm3d() != 0);
        ARM_COMPUTE_RETURN_ERROR_ON(gemm_info.reinterpret_input_as_3d());
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(c, d);
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(a->dimension(1) != c->dimension(1),
                                        "The C matrix must have the same number of rows as the matrix A");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(b->dimension(0) != c->dimension(0),
                                        "The C matrix must have the same number of columns as the matrix B");
    }

    if (d->total_size() != 0)
    {
        // For fixed format we are expecting some kind of blocked format for B/RHS so the dimension won't necessarily match the result matrix any more.
        ARM_COMPUTE_RETURN_ERROR_ON(!gemm_info.fixed_format() && b->dimension(0) != d->dimension(0));
        if (gemm_info.depth_output_gemm3d() != 0)
        {
            if (gemm_info.reinterpret_input_as_3d())
            {
                ARM_COMPUTE_RETURN_ERROR_ON(a->dimension(1) != d->dimension(1));
                ARM_COMPUTE_RETURN_ERROR_ON(a->dimension(2) != d->dimension(2));
            }
            else
            {
                ARM_COMPUTE_RETURN_ERROR_ON(a->dimension(1) != d->dimension(1) * d->dimension(2));
            }
        }
        else
        {
            ARM_COMPUTE_RETURN_ERROR_ON(a->dimension(1) != d->dimension(1));
        }
    }

    // Check if we need to run the optimized assembly kernel
    cpu::AsmGemmInfo asm_info = init_assembly_metadata(gemm_info);
    const bool       run_optimised =
        bool(cpu::CpuGemmAssemblyDispatch::validate(a, b, is_c_bias ? c : nullptr, d, asm_info)) &&
        (c == nullptr || beta == 0.f || beta == 1.f) && // Optimized GeMM doesn't support beta coefficient.
        !(!b->are_values_constant() &&
          b->tensor_shape().z() > 1); // Disable batch matmul as optimized GeMM handles batching differently.

    if (!run_optimised)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(gemm_info.reinterpret_input_as_3d(),
                                        "CpuGemm cannot reinterpret the input tensor as 3D");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(gemm_info.depth_output_gemm3d() != 0,
                                        "CpuGemm cannot reinterpret the output tensor as 3D");

        // Check if the first input tensor is a vector.
        const bool run_vector_matrix_multiplication = a->dimension(1) < 2;
        // Check if we need to reshape the matrix A and matrix B
        const bool run_interleave_transpose = !run_vector_matrix_multiplication && !b->are_values_constant();

        // Arguments used by GEMMReshapeInfo
        // If we pass the matrix A and matrix B reshaped to CpuGemmMatrixMultiplyKernel, we need to pass m, n, k, mult_transpose1xW_width and mult_interleave4x4_height to GEMMReshapeInfo
        // in order to know how the matrices have been reshaped
        const int m                         = a->dimension(1);
        const int n                         = b->dimension(0);
        const int k                         = a->dimension(0);
        int       mult_transpose1xW_width   = 1;
        int       mult_interleave4x4_height = 1;

        const GEMMReshapeInfo reshape_info = GEMMReshapeInfo(
            m, n, k, mult_transpose1xW_width, mult_interleave4x4_height, gemm_info.depth_output_gemm3d());

        const ITensorInfo *matrix_a_info = a;
        const ITensorInfo *matrix_b_info = b;

        TensorInfo tmp_a_info{};
        TensorInfo tmp_b_info{};
        TensorInfo tmp_output_info = *d->clone();

        if (run_interleave_transpose)
        {
            matrix_a_info = &tmp_a_info;
            matrix_b_info = &tmp_b_info;

            // Validate interleave kernel
            auto_init_if_empty(tmp_a_info, a->clone()->set_tensor_shape(compute_interleaved_shape(
                                               *a, mult_interleave4x4_height, gemm_info.reinterpret_input_as_3d())));
            ARM_COMPUTE_RETURN_ON_ERROR(cpu::kernels::CpuGemmInterleave4x4Kernel::validate(a, &tmp_a_info));

            // Validate transpose kernel
            auto_init_if_empty(tmp_b_info, b->clone()->set_tensor_shape(compute_transpose1xW_with_element_size_shape(
                                               *b, mult_transpose1xW_width)));
            ARM_COMPUTE_RETURN_ON_ERROR(cpu::kernels::CpuGemmTranspose1xWKernel::validate(b, &tmp_b_info));
        }

        // Validate matrix multiply
        auto_init_if_empty(tmp_output_info,
                           matrix_a_info->clone()->set_tensor_shape(compute_mm_shape(
                               *matrix_a_info, *matrix_b_info, run_interleave_transpose, reshape_info)));
        ARM_COMPUTE_RETURN_ON_ERROR(cpu::kernels::CpuGemmMatrixMultiplyKernel::validate(
            matrix_a_info, matrix_b_info, &tmp_output_info, alpha, run_interleave_transpose, reshape_info));

        if (is_c_bias)
        {
            ARM_COMPUTE_RETURN_ON_ERROR(cpu::CpuAdd::validate(&tmp_output_info, c, d, ConvertPolicy::SATURATE));
        }
    }

    // Validate matrix addition kernel
    if (run_addition)
    {
        ARM_COMPUTE_RETURN_ON_ERROR(cpu::kernels::CpuGemmMatrixAdditionKernel::validate(c, d, beta));
    }

    // Validate activation
    const ActivationLayerInfo &activation = gemm_info.activation_info();
    if (activation.enabled())
    {
        ARM_COMPUTE_RETURN_ON_ERROR(cpu::CpuActivation::validate(d, nullptr, activation));
    }

    return Status{};
}

void CpuGemm::run(ITensorPack &tensors)
{
    prepare(tensors);

    auto a = tensors.get_const_tensor(ACL_SRC_0);
    auto b = tensors.get_const_tensor(ACL_SRC_1);
    auto c = tensors.get_const_tensor(ACL_SRC_2);
    auto d = tensors.get_tensor(ACL_DST);

    if (_asm_glue && _asm_glue->is_configured())
    {
        // Pass c to asm dispatch only if it's the bias tensor
        ITensorPack asm_pack = tensors;
        asm_pack.add_const_tensor(ACL_SRC_2, _run_bias_addition ? c : nullptr);
        _asm_glue->run(asm_pack);
        if (_run_alpha_scale)
        {
            ITensorPack pack{{ACL_SRC, d}, {ACL_DST, d}};
            _alpha_scale_func->run(pack);
        }
    }
    else
    {
        CpuAuxTensorHandler interleaved_a(offset_int_vec(InterleavedLHS), _tmp_a, tensors, true);
        CpuAuxTensorHandler transposed_b(offset_int_vec(TransposedRHS), _tmp_b, tensors, true);
        CpuAuxTensorHandler temp_d(offset_int_vec(TempResult), _tmp_d, tensors, true);

        ITensorPack mm_pack{{ACL_SRC_0, a}, {ACL_SRC_1, b}, {ACL_DST, (_run_bias_addition) ? temp_d.get() : d}};
        if (!_run_vector_matrix_multiplication)
        {
            // Run interleave kernel
            ITensorPack interleave_pack{{ACL_SRC, a}, {ACL_DST, interleaved_a.get()}};
            NEScheduler::get().schedule_op(_interleave_kernel.get(), Window::DimY, _interleave_kernel->window(),
                                           interleave_pack);

            if (!_reshape_b_only_on_first_run)
            {
                // Run transpose kernel
                ITensorPack transpose_pack{{ACL_SRC, b}, {ACL_DST, transposed_b.get()}};
                NEScheduler::get().schedule_op(_transpose_kernel.get(), Window::DimY, _transpose_kernel->window(),
                                               transpose_pack);
            }

            // Use reshaped matrices
            mm_pack.add_const_tensor(ACL_SRC_0, interleaved_a.get());
            mm_pack.add_const_tensor(ACL_SRC_1, transposed_b.get());
        }

        NEScheduler::get().schedule_op(_mm_kernel.get(),
                                       _run_vector_matrix_multiplication ? Window::DimX : Window::DimY,
                                       _mm_kernel->window(), mm_pack);

        // Run bias addition kernel
        if (_run_bias_addition)
        {
            ITensorPack pack{{ACL_SRC_0, temp_d.get()}, {ACL_SRC_1, c}, {ACL_DST, d}};
            _add_bias->run(pack);
        }
    }

    // Run matrix addition kernel
    if (_run_addition)
    {
        ITensorPack c_add_pack{{ACL_SRC, c}, {ACL_DST, d}};
        NEScheduler::get().schedule_op(_ma_kernel.get(), Window::DimY, _ma_kernel->window(), c_add_pack);
    }

    // Run activation function
    if (_run_activation)
    {
        ITensorPack pack{{ACL_SRC, d}, {ACL_DST, d}};
        _activation_func->run(pack);
    }
}

void CpuGemm::prepare(ITensorPack &tensors)
{
    if (!_is_prepared)
    {
        if (_asm_glue && _asm_glue->is_configured())
        {
            _asm_glue->prepare(tensors);
        }
        else if (_reshape_b_only_on_first_run && !_run_vector_matrix_multiplication)
        {
            const ITensor *b = tensors.get_const_tensor(ACL_SRC_1);
            ITensor       *b_aux =
                utils::cast::polymorphic_cast<ITensor *>(tensors.get_tensor(offset_int_vec(TransposedRHS)));
            ARM_COMPUTE_ERROR_ON_NULLPTR(b, b_aux);

            CpuAuxTensorHandler transposed_b(_tmp_b, *b_aux);
            ITensorPack         transpose_pack{{ACL_SRC, b}, {ACL_DST, transposed_b.get()}};
            NEScheduler::get().schedule_op(_transpose_kernel.get(), Window::DimY, _transpose_kernel->window(),
                                           transpose_pack);
        }
        _is_prepared = true;
    }
}

experimental::MemoryRequirements CpuGemm::workspace() const
{
    return _aux_mem;
}

Status CpuGemm::has_opt_impl(arm_compute::WeightFormat &expected_weight_format,
                             const ITensorInfo         *a,
                             const ITensorInfo         *b,
                             const ITensorInfo         *c,
                             const ITensorInfo         *d,
                             const GEMMInfo            &gemm_info)
{
    const cpu::AsmGemmInfo asm_info = init_assembly_metadata(gemm_info);

    return CpuGemmAssemblyDispatch::has_opt_impl(expected_weight_format, a, b, c, d, asm_info);
}

bool CpuGemm::isVarWeightsKernel() const
{
    return _asm_glue && _asm_glue->isVarWeightsKernel();
}
} // namespace cpu
} // namespace arm_compute
