/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEGEMM.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "src/core/CPP/Validate.h"
#include "src/core/NEON/kernels/NEGEMMMatrixMultiplyKernel.h"
#include "src/core/cpu/kernels/CpuGemmInterleave4x4Kernel.h"
#include "src/core/cpu/kernels/CpuGemmMatrixAdditionKernel.h"
#include "src/core/cpu/kernels/CpuGemmTranspose1xWKernel.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/MemoryHelpers.h"
#include "src/runtime/cpu/operators/CpuActivation.h"
#include "src/runtime/cpu/operators/CpuAdd.h"
#include "src/runtime/cpu/operators/internal/CpuGemmAssemblyDispatch.h"

using namespace arm_compute::experimental;
using namespace arm_compute::misc::shape_calculator;

namespace arm_compute
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

    return asm_info;
}
} // namespace

struct NEGEMM::Impl
{
    MemoryGroup      memory_group{};
    IWeightsManager *weights_manager{ nullptr };

    std::unique_ptr<cpu::kernels::CpuGemmInterleave4x4Kernel>  interleave_kernel{ nullptr };
    std::unique_ptr<cpu::kernels::CpuGemmTranspose1xWKernel>   transpose_kernel{ nullptr };
    std::unique_ptr<NEGEMMMatrixMultiplyKernel>                mm_kernel{ nullptr };
    std::unique_ptr<cpu::CpuGemmAssemblyDispatch>              asm_glue{ nullptr };
    std::unique_ptr<cpu::kernels::CpuGemmMatrixAdditionKernel> ma_kernel{ nullptr };
    std::unique_ptr<cpu::CpuActivation>                        alpha_scale_func{ nullptr };
    std::unique_ptr<cpu::CpuAdd>                               add_bias{ nullptr };
    std::unique_ptr<cpu::CpuActivation>                        activation_func{ nullptr };

    const ITensor *a{ nullptr };
    const ITensor *c{ nullptr };
    ITensor       *d{ nullptr };
    ITensor       *gemm_output_to_use{ nullptr };
    Tensor         tmp_a{};
    Tensor         tmp_b{};
    Tensor         tmp_d{};
    const ITensor *original_b{ nullptr };
    bool           run_vector_matrix_multiplication{ false };
    bool           run_alpha_scale{ false };
    bool           run_addition{ false };
    bool           run_bias_addition{ false };
    bool           run_activation{ false };
    bool           reshape_b_only_on_first_run{ false };
    bool           is_prepared{ false };

    ITensorPack                      asm_glue_run_pack{};
    ITensorPack                      asm_glue_prep_pack{};
    WorkspaceData<Tensor>            asm_glue_workspace{};
    experimental::MemoryRequirements aux_mem_req{};
};

NEGEMM::NEGEMM(std::shared_ptr<IMemoryManager> memory_manager, IWeightsManager *weights_manager)
    : _impl(std::make_unique<Impl>())
{
    _impl->memory_group    = MemoryGroup(std::move(memory_manager));
    _impl->weights_manager = weights_manager;
}

NEGEMM::~NEGEMM() = default;

void NEGEMM::configure(const ITensor *a, const ITensor *b, const ITensor *c, ITensor *d, float alpha, float beta, const GEMMInfo &gemm_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(a, b, d);
    ARM_COMPUTE_ERROR_THROW_ON(NEGEMM::validate(a->info(), b->info(), (c != nullptr) ? c->info() : nullptr, d->info(), alpha, beta, gemm_info));

    const cpu::AsmGemmInfo asm_info      = init_assembly_metadata(gemm_info);
    const bool             is_c_bias     = gemm_info.reshape_b_only_on_first_run();
    bool                   run_optimised = bool(cpu::CpuGemmAssemblyDispatch::validate(a->info(), b->info(), (is_c_bias && c != nullptr) ? c->info() : nullptr, d->info(), asm_info));

    _impl->a                  = a;
    _impl->c                  = c;
    _impl->d                  = d;
    _impl->gemm_output_to_use = d;
    // Check if we need to reshape the matrix B only on the first run
    _impl->is_prepared                      = false;
    _impl->reshape_b_only_on_first_run      = gemm_info.reshape_b_only_on_first_run();
    _impl->run_vector_matrix_multiplication = a->info()->dimension(1) < 2;
    _impl->original_b                       = b;
    _impl->run_alpha_scale                  = alpha != 1.f;
    _impl->run_bias_addition                = c != nullptr && gemm_info.reshape_b_only_on_first_run();
    _impl->run_addition                     = beta != 0 && c != nullptr && !gemm_info.reshape_b_only_on_first_run();
    _impl->run_activation                   = gemm_info.activation_info().enabled() && (!run_optimised || (run_optimised
                                                                                                           && !cpu::CpuGemmAssemblyDispatch::is_activation_supported(gemm_info.activation_info())));

    if(run_optimised)
    {
        const ITensor     *c_to_use      = is_c_bias ? c : nullptr;
        const ITensorInfo *c_info_to_use = c_to_use != nullptr ? c_to_use->info() : nullptr;
        _impl->asm_glue                  = std::make_unique<cpu::CpuGemmAssemblyDispatch>();
        _impl->asm_glue->configure(a->info(), b->info(), c_info_to_use, d->info(), asm_info);
        ARM_COMPUTE_ERROR_ON(!_impl->asm_glue->is_configured());

        _impl->aux_mem_req = _impl->asm_glue->workspace();
        _impl->asm_glue_run_pack =
        {
            { ACL_SRC_0, a },
            { ACL_SRC_1, b },
            { ACL_SRC_2, c_to_use },
            { ACL_DST, d },
        };
        _impl->asm_glue_prep_pack = { { ACL_SRC_1, b }, { ACL_SRC_2, c_to_use } };
        _impl->asm_glue_workspace = manage_workspace<Tensor>(_impl->aux_mem_req, _impl->memory_group, _impl->asm_glue_run_pack, _impl->asm_glue_prep_pack);

        // Scale product by alpha
        if(_impl->run_alpha_scale)
        {
            _impl->alpha_scale_func = std::make_unique<cpu::CpuActivation>();
            _impl->alpha_scale_func->configure(d->info(), nullptr, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LINEAR, alpha, 0.f));
        }
    }
    else
    {
        // Pick output tensor in case bias addition should be performed
        if(_impl->run_bias_addition)
        {
            _impl->gemm_output_to_use = &_impl->tmp_d;
            _impl->memory_group.manage(&_impl->tmp_d);
        }

        _impl->mm_kernel = std::make_unique<NEGEMMMatrixMultiplyKernel>();

        // Select between GEMV and GEMM
        if(_impl->run_vector_matrix_multiplication)
        {
            // Configure the matrix multiply kernel
            _impl->mm_kernel->configure(a, b, _impl->gemm_output_to_use, alpha, false);
        }
        else
        {
            TensorShape shape_tmp_a = a->info()->tensor_shape();
            TensorShape shape_tmp_b = b->info()->tensor_shape();

            shape_tmp_a.set(0, a->info()->dimension(0) * 4);
            shape_tmp_a.set(1, std::ceil(a->info()->dimension(1) / 4.0f));

            const unsigned int transpose_w = 16 / data_size_from_type(b->info()->data_type());
            shape_tmp_b.set(0, b->info()->dimension(1) * transpose_w);
            shape_tmp_b.set(1, std::ceil(b->info()->dimension(0) / static_cast<float>(transpose_w)));

            TensorInfo info_a = a->info()->clone()->set_tensor_shape(shape_tmp_a).set_is_resizable(true);
            TensorInfo info_b = b->info()->clone()->set_tensor_shape(shape_tmp_b).set_is_resizable(true);

            _impl->tmp_a.allocator()->init(info_a);
            _impl->tmp_b.allocator()->init(info_b);

            // Manage intermediate buffers
            _impl->memory_group.manage(&_impl->tmp_a);
            if(!_impl->reshape_b_only_on_first_run)
            {
                _impl->memory_group.manage(&_impl->tmp_b);
            }

            int m = a->info()->dimension(1);
            int n = b->info()->dimension(0);
            int k = a->info()->dimension(0);

            // Configure interleave kernel
            _impl->interleave_kernel = std::make_unique<cpu::kernels::CpuGemmInterleave4x4Kernel>();
            _impl->interleave_kernel->configure(a->info(), &info_a);

            // Configure transpose kernel
            _impl->transpose_kernel = std::make_unique<cpu::kernels::CpuGemmTranspose1xWKernel>();
            _impl->transpose_kernel->configure(b->info(), _impl->tmp_b.info());

            // Configure matrix multiplication kernel
            _impl->mm_kernel->configure(&_impl->tmp_a, &_impl->tmp_b, _impl->gemm_output_to_use, alpha, true, GEMMReshapeInfo(m, n, k));

            // Allocate once the all configure methods have been called
            _impl->tmp_a.allocator()->allocate();
            if(!_impl->reshape_b_only_on_first_run)
            {
                _impl->tmp_b.allocator()->allocate();
            }
        }

        if(_impl->run_bias_addition)
        {
            _impl->add_bias = std::make_unique<cpu::CpuAdd>();
            _impl->add_bias->configure(_impl->gemm_output_to_use->info(), c->info(), d->info(), ConvertPolicy::SATURATE);
            _impl->tmp_d.allocator()->allocate();
        }
    }

    // Configure matrix addition kernel
    if(_impl->run_addition)
    {
        _impl->ma_kernel = std::make_unique<cpu::kernels::CpuGemmMatrixAdditionKernel>();
        _impl->ma_kernel->configure(c->info(), d->info(), beta);
    }

    // Configure activation
    const ActivationLayerInfo &activation = gemm_info.activation_info();
    if(_impl->run_activation)
    {
        _impl->activation_func = std::make_unique<cpu::CpuActivation>();
        _impl->activation_func->configure(d->info(), nullptr, activation);
    }
}

Status NEGEMM::validate(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, const ITensorInfo *output, float alpha, float beta, const GEMMInfo &gemm_info)
{
    ARM_COMPUTE_UNUSED(alpha);
    const bool is_c_bias = gemm_info.reshape_b_only_on_first_run();

    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(a);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_BF16_UNSUPPORTED(a);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(a, 1, DataType::BFLOAT16, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(a, b);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(a->dimension(0) != b->dimension(1), "The product AB is defined only if the number of columns in A is equal to the number of rows in B");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(gemm_info.is_a_reshaped(), "Matrix A already reshaped is not supported");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(gemm_info.is_b_reshaped(), "Matrix B already reshaped is not supported");
    if(a->data_type() != DataType::BFLOAT16)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(a, output);
    }

    if(c != nullptr && !is_c_bias)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(gemm_info.depth_output_gemm3d() != 0);
        ARM_COMPUTE_RETURN_ERROR_ON(gemm_info.reinterpret_input_as_3d());
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(c, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(a->dimension(1) != c->dimension(1), "The C matrix must have the same number of rows as the matrix A");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(b->dimension(0) != c->dimension(0), "The C matrix must have the same number of columns as the matrix B");
    }

    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(b->dimension(0) != output->dimension(0));
        if(gemm_info.depth_output_gemm3d() != 0)
        {
            if(gemm_info.reinterpret_input_as_3d())
            {
                ARM_COMPUTE_RETURN_ERROR_ON(a->dimension(1) != output->dimension(1));
                ARM_COMPUTE_RETURN_ERROR_ON(a->dimension(2) != output->dimension(2));
            }
            else
            {
                ARM_COMPUTE_RETURN_ERROR_ON(a->dimension(1) != output->dimension(1) * output->dimension(2));
            }
        }
        else
        {
            ARM_COMPUTE_RETURN_ERROR_ON(a->dimension(1) != output->dimension(1));
        }
    }

    // Check if we need to run the optimized assembly kernel
    cpu::AsmGemmInfo asm_info      = init_assembly_metadata(gemm_info);
    const bool       run_optimised = bool(cpu::CpuGemmAssemblyDispatch::validate(a, b, is_c_bias ? c : nullptr, output, asm_info));

    if(!run_optimised)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(gemm_info.reinterpret_input_as_3d(), "NEGEMM cannot reinterpret the input tensor as 3D");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(gemm_info.depth_output_gemm3d() != 0, "NEGEMM cannot reinterpret the output tensor as 3D");

        // Check if the first input tensor is a vector.
        const bool run_vector_matrix_multiplication = a->dimension(1) < 2;
        // Check if we need to reshape the matrix A and matrix B
        const bool run_interleave_transpose = !run_vector_matrix_multiplication && !(gemm_info.reshape_b_only_on_first_run());

        // Arguments used by GEMMReshapeInfo
        // If we pass the matrix A and matrix B reshaped to NEGEMMMatrixMultiplyKernel, we need to pass m, n, k, mult_transpose1xW_width and mult_interleave4x4_height to GEMMReshapeInfo
        // in order to know how the matrices have been reshaped
        const int m                         = a->dimension(1);
        const int n                         = b->dimension(0);
        const int k                         = a->dimension(0);
        int       mult_transpose1xW_width   = 1;
        int       mult_interleave4x4_height = 1;

        const GEMMReshapeInfo reshape_info = GEMMReshapeInfo(m, n, k, mult_transpose1xW_width, mult_interleave4x4_height, gemm_info.depth_output_gemm3d());

        const ITensorInfo *matrix_a_info = a;
        const ITensorInfo *matrix_b_info = b;

        TensorInfo tmp_a_info{};
        TensorInfo tmp_b_info{};
        TensorInfo tmp_output_info = *output->clone();

        if(run_interleave_transpose)
        {
            matrix_a_info = &tmp_a_info;
            matrix_b_info = &tmp_b_info;

            // Validate interleave kernel
            auto_init_if_empty(tmp_a_info, a->clone()->set_tensor_shape(compute_interleaved_shape(*a, mult_interleave4x4_height, gemm_info.reinterpret_input_as_3d())));
            ARM_COMPUTE_RETURN_ON_ERROR(cpu::kernels::CpuGemmInterleave4x4Kernel::validate(a, &tmp_a_info));

            // Validate transpose kernel
            auto_init_if_empty(tmp_b_info, b->clone()->set_tensor_shape(compute_transpose1xW_with_element_size_shape(*b, mult_transpose1xW_width)));
            ARM_COMPUTE_RETURN_ON_ERROR(cpu::kernels::CpuGemmTranspose1xWKernel::validate(b, &tmp_b_info));
        }

        // Validate matrix multiply
        auto_init_if_empty(tmp_output_info, matrix_a_info->clone()->set_tensor_shape(compute_mm_shape(*matrix_a_info, *matrix_b_info, run_interleave_transpose, reshape_info)));
        ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMMatrixMultiplyKernel::validate(matrix_a_info, matrix_b_info, &tmp_output_info, alpha, run_interleave_transpose, reshape_info));

        if(c != nullptr && gemm_info.reshape_b_only_on_first_run())
        {
            ARM_COMPUTE_RETURN_ON_ERROR(cpu::CpuAdd::validate(&tmp_output_info, c, output, ConvertPolicy::SATURATE));
        }
    }

    // Validate matrix addition kernel
    if(beta != 0 && c != nullptr && !is_c_bias)
    {
        ARM_COMPUTE_RETURN_ON_ERROR(cpu::kernels::CpuGemmMatrixAdditionKernel::validate(c, output, beta));
    }

    // Validate activation
    const ActivationLayerInfo &activation = gemm_info.activation_info();
    if(activation.enabled())
    {
        ARM_COMPUTE_RETURN_ON_ERROR(cpu::CpuActivation::validate(output, nullptr, activation));
    }

    return Status{};
}

void NEGEMM::run()
{
    prepare();

    MemoryGroupResourceScope scope_mg(_impl->memory_group);

    if(_impl->asm_glue->is_configured())
    {
        _impl->asm_glue->run(_impl->asm_glue_run_pack);
        if(_impl->run_alpha_scale)
        {
            ITensorPack pack{ { ACL_SRC, _impl->d }, { ACL_DST, _impl->d } };
            _impl->alpha_scale_func->run(pack);
        }
    }
    else
    {
        if(!_impl->run_vector_matrix_multiplication)
        {
            // Run interleave kernel
            ITensorPack interleave_pack{ { ACL_SRC, _impl->a }, { ACL_DST, &_impl->tmp_a } };
            NEScheduler::get().schedule_op(_impl->interleave_kernel.get(), Window::DimY, _impl->interleave_kernel->window(), interleave_pack);

            if(!_impl->reshape_b_only_on_first_run)
            {
                // Run transpose kernel
                ITensorPack transpose_pack{ { ACL_SRC, _impl->original_b }, { ACL_DST, &_impl->tmp_b } };
                NEScheduler::get().schedule_op(_impl->transpose_kernel.get(), Window::DimY, _impl->transpose_kernel->window(), transpose_pack);
            }
        }

        NEScheduler::get().schedule(_impl->mm_kernel.get(), _impl->run_vector_matrix_multiplication ? Window::DimX : Window::DimY);

        // Run bias addition kernel
        if(_impl->run_bias_addition)
        {
            ITensorPack pack{ { ACL_SRC_0, _impl->gemm_output_to_use }, { ACL_SRC_1, _impl->c }, { ACL_DST, _impl->d } };
            _impl->add_bias->run(pack);
        }
    }

    // Run matrix addition kernel
    if(_impl->run_addition)
    {
        ITensorPack c_add_pack{ { ACL_SRC, _impl->c }, { ACL_DST, _impl->d } };
        NEScheduler::get().schedule_op(_impl->ma_kernel.get(), Window::DimY, _impl->ma_kernel->window(), c_add_pack);
    }

    // Run activation function
    if(_impl->run_activation)
    {
        ITensorPack pack{ { ACL_SRC, _impl->d }, { ACL_DST, _impl->d } };
        _impl->activation_func->run(pack);
    }
}

void NEGEMM::prepare()
{
    if(!_impl->is_prepared)
    {
        const bool original_b_managed_by_weights_manager = _impl->weights_manager && _impl->weights_manager->are_weights_managed(_impl->original_b);
        if(_impl->asm_glue->is_configured())
        {
            _impl->asm_glue->prepare(_impl->asm_glue_prep_pack);

            auto has_reshape = std::find_if(_impl->aux_mem_req.begin(),
                                            _impl->aux_mem_req.end(),
                                            [](const MemoryInfo & m) -> bool { return m.lifetime == MemoryLifetime::Persistent; });

            if(has_reshape != std::end(_impl->aux_mem_req))
            {
                _impl->original_b->mark_as_unused();
            }
            else
            {
                _impl->asm_glue_run_pack.add_const_tensor(ACL_SRC_1, _impl->original_b);
            }
        }
        else if(_impl->reshape_b_only_on_first_run && !_impl->run_vector_matrix_multiplication && !_impl->asm_glue->is_configured())
        {
            if(!original_b_managed_by_weights_manager)
            {
                ARM_COMPUTE_ERROR_ON(!_impl->original_b->is_used());
            }

            _impl->tmp_b.allocator()->allocate();
            ITensorPack transpose_pack{ { ACL_SRC, _impl->original_b }, { ACL_DST, &_impl->tmp_b } };
            NEScheduler::get().schedule_op(_impl->transpose_kernel.get(), Window::DimY, _impl->transpose_kernel->window(), transpose_pack);
            if(!original_b_managed_by_weights_manager)
            {
                _impl->original_b->mark_as_unused();
            }
        }

        _impl->is_prepared = true;
    }
}
} // namespace arm_compute
