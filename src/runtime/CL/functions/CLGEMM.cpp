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

#include "arm_compute/core/CL/ICLGEMMKernelConfiguration.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/gemm/reshaped/CLGEMMReshapedKernelConfiguration.h"
#include "arm_compute/core/CL/gemm/reshaped_only_rhs/CLGEMMReshapedOnlyRHSKernelConfiguration.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/GPUTarget.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/helpers/float_ops.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/ITensorAllocator.h"

namespace arm_compute
{
using namespace arm_compute::misc::shape_calculator;
using namespace arm_compute::cl_gemm;

CLGEMM::CLGEMM(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)),
      _mm_kernel(),
      _reshape_lhs_kernel(),
      _reshape_rhs_kernel(),
      _mm_reshaped_kernel(),
      _mm_reshaped_only_rhs_kernel(),
      _tmp_a(),
      _tmp_b(),
      _original_b(nullptr),
      _reshape_b_only_on_first_run(false),
      _is_prepared(false),
      _gemm_type(GEMMType::NATIVE)
{
}

CLGEMM::GEMMType CLGEMM::select_gemm_type(unsigned int m, unsigned int n, unsigned int k, DataType data_type, bool reshape_b_only_on_first_run, GPUTarget gpu_target)
{
    GEMMType gemm_type = GEMMType::RESHAPED_V1;

    if(gpu_target_is_in(gpu_target, GPUTarget::G52, GPUTarget::G52LIT, GPUTarget::G71, GPUTarget::G72, GPUTarget::G76))
    {
        if((m > 1) && (n < 16))
        {
            gemm_type = GEMMType::RESHAPED_V1;
        }
        else if((m == 1) && (data_type == DataType::F32))
        {
            gemm_type = GEMMType::RESHAPED_ONLY_RHS;
        }
        else
        {
            // COMPMID-852
            if((k > 256) && (m > 4) && is_data_type_float(data_type) && reshape_b_only_on_first_run)
            {
                constexpr float alpha = 3.2f;
                constexpr float fact0 = 1.51f;
                constexpr float fact1 = 1.66f;
                constexpr float ops   = 12.0f;
                const float     scale = k > 1024 ? 1.07f : 1.0f;
                gemm_type             = (alpha + ((n * fact0) / ops) < ((fact1 * n * scale) / ops)) ? GEMMType::RESHAPED_V1 : GEMMType::NATIVE;
            }
            else
            {
                gemm_type = GEMMType::NATIVE;
            }
        }

        const auto workload = static_cast<float>((m * n) / 20.0f);

        gemm_type = ((workload > 1600.0f) && (gemm_type == GEMMType::RESHAPED_V1) && (data_type == DataType::F32)) ? GEMMType::RESHAPED_V2 : gemm_type;
    }
    else
    {
        // We reshape the matrices only if we do not have the vector-by-matrix case and we reshape the matrix B only once
        gemm_type = ((m != 1) && reshape_b_only_on_first_run) ? GEMMType::RESHAPED_V1 : GEMMType::NATIVE;
    }

    return gemm_type;
}

void CLGEMM::configure_native(const ICLTensor *a, const ICLTensor *b, const ICLTensor *c, ICLTensor *output, float alpha, float beta, const GEMMInfo &gemm_info)
{
    const unsigned int m          = gemm_info.reinterpret_input_as_3d() ? (a->info()->dimension(1) * a->info()->dimension(2)) : a->info()->dimension(1);
    const unsigned int n          = b->info()->dimension(0);
    const unsigned int k          = a->info()->dimension(0);
    const GPUTarget    gpu_target = CLScheduler::get().target();

    // Set the target for the kernels
    _mm_kernel.set_target(gpu_target);

    GEMMReshapeInfo reshape_info(m, n, k, 1, 1, gemm_info.depth_output_gemm3d(), gemm_info.reinterpret_input_as_3d(), gemm_info.broadcast_bias());

    // Configure and tune matrix multiply kernel
    _mm_kernel.configure(a, b, c, output, alpha, beta, false, reshape_info, gemm_info.fp_mixed_precision(), gemm_info.activation_info());

    // Tune kernel statically
    CLScheduler::get().tune_kernel_static(_mm_kernel);
}

void CLGEMM::configure_reshaped_v1(const ICLTensor *a, const ICLTensor *b, const ICLTensor *c, ICLTensor *output, float alpha, float beta, const GEMMInfo &gemm_info)
{
    bool               reinterpret_input_as_3d   = gemm_info.reinterpret_input_as_3d();
    const unsigned int m                         = reinterpret_input_as_3d ? (a->info()->dimension(1) * a->info()->dimension(2)) : a->info()->dimension(1);
    const unsigned int n                         = b->info()->dimension(0);
    const unsigned int k                         = a->info()->dimension(0);
    const int          depth_output_gemm3d       = gemm_info.depth_output_gemm3d();
    const GPUTarget    gpu_target                = CLScheduler::get().target();
    int                mult_transpose1xW_width   = 1;
    int                mult_interleave4x4_height = 1;

    // Set the target for the kernels
    _reshape_lhs_kernel.set_target(gpu_target);
    _mm_kernel.set_target(gpu_target);

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

    GEMMReshapeInfo reshape_info(m, n, k, mult_transpose1xW_width, mult_interleave4x4_height, depth_output_gemm3d, false, gemm_info.broadcast_bias());

    _memory_group.manage(&_tmp_a);
    if(!_reshape_b_only_on_first_run)
    {
        _memory_group.manage(&_tmp_b);
    }

    // Configure interleave kernel
    _reshape_lhs_kernel.configure(a, &_tmp_a, lhs_info, reinterpret_input_as_3d);

    // Configure transpose kernel
    _reshape_rhs_kernel.configure(b, &_tmp_b, rhs_info);

    // Configure and tune matrix multiply kernel
    _mm_kernel.configure(&_tmp_a, &_tmp_b, c, output, alpha, beta, true, reshape_info, gemm_info.fp_mixed_precision(), gemm_info.activation_info());

    CLScheduler::get().tune_kernel_static(_mm_kernel);

    // Allocate intermediate tensors
    _tmp_a.allocator()->allocate();
    if(!_reshape_b_only_on_first_run)
    {
        _tmp_b.allocator()->allocate();
    }
}

void CLGEMM::configure_reshaped_v2(const ICLTensor *a, const ICLTensor *b, const ICLTensor *c, ICLTensor *output, float alpha, float beta, const GEMMInfo &gemm_info)
{
    DataType           data_type               = a->info()->data_type();
    bool               reinterpret_input_as_3d = gemm_info.reinterpret_input_as_3d();
    const unsigned int m                       = reinterpret_input_as_3d ? (a->info()->dimension(1) * a->info()->dimension(2)) : a->info()->dimension(1);
    const unsigned int n                       = b->info()->dimension(0);
    const unsigned int k                       = a->info()->dimension(0);
    const unsigned int batch_size              = reinterpret_input_as_3d ? a->info()->dimension(3) : a->info()->dimension(2);
    const int          depth_output_gemm3d     = gemm_info.depth_output_gemm3d();
    const GPUTarget    gpu_target              = CLScheduler::get().target();
    bool               broadcast_bias          = gemm_info.broadcast_bias();

    GEMMKernelInfo kernel_info;
    kernel_info.m                       = m;
    kernel_info.n                       = n;
    kernel_info.k                       = k;
    kernel_info.depth_output_gemm3d     = depth_output_gemm3d;
    kernel_info.reinterpret_input_as_3d = false;
    kernel_info.broadcast_bias          = broadcast_bias;
    kernel_info.activation_info         = gemm_info.activation_info();

    // Set the target for the kernels
    _reshape_lhs_kernel.set_target(gpu_target);
    _mm_kernel.set_target(gpu_target);

    // Manage intermediate buffers
    _memory_group.manage(&_tmp_a);
    if(!_reshape_b_only_on_first_run)
    {
        _memory_group.manage(&_tmp_b);
    }
    // _tmp_a and _tmp_b will be auto configured in _interleave_kernel and in _transpose_kernel

    GEMMLHSMatrixInfo lhs_info{};
    GEMMRHSMatrixInfo rhs_info{};

    // Pick up the GEMM configuration
    std::unique_ptr<ICLGEMMKernelConfiguration> gemm_config = CLGEMMReshapedKernelConfigurationFactory::create(gpu_target);
    ARM_COMPUTE_ERROR_ON_NULLPTR(gemm_config.get());

    // Configure lhs_info and rhs_info
    std::tie(lhs_info, rhs_info) = gemm_config->configure(m, n, k, batch_size, data_type);

    _reshape_lhs_kernel.configure(a, &_tmp_a, lhs_info, gemm_info.reinterpret_input_as_3d());
    _reshape_rhs_kernel.configure(b, &_tmp_b, rhs_info);

    // Configure and tune matrix multiply kernel
    _mm_reshaped_kernel.configure(&_tmp_a, &_tmp_b, c, output, alpha, beta, lhs_info, rhs_info, kernel_info);

    // Allocate intermediate tensors
    _tmp_a.allocator()->allocate();
    if(!_reshape_b_only_on_first_run)
    {
        _tmp_b.allocator()->allocate();
    }
}

void CLGEMM::configure_reshaped_only_rhs(const ICLTensor *a, const ICLTensor *b, const ICLTensor *c, ICLTensor *output, float alpha, float beta, const GEMMInfo &gemm_info)
{
    DataType           data_type               = a->info()->data_type();
    bool               reinterpret_input_as_3d = gemm_info.reinterpret_input_as_3d();
    const unsigned int m                       = reinterpret_input_as_3d ? (a->info()->dimension(1) * a->info()->dimension(2)) : a->info()->dimension(1);
    const unsigned int n                       = b->info()->dimension(0);
    const unsigned int k                       = a->info()->dimension(0);
    const unsigned int batch_size              = reinterpret_input_as_3d ? a->info()->dimension(3) : a->info()->dimension(2);
    const int          depth_output_gemm3d     = gemm_info.depth_output_gemm3d();
    const GPUTarget    gpu_target              = CLScheduler::get().target();
    bool               broadcast_bias          = gemm_info.broadcast_bias();

    GEMMKernelInfo kernel_info;
    kernel_info.m                       = m;
    kernel_info.n                       = n;
    kernel_info.k                       = k;
    kernel_info.depth_output_gemm3d     = depth_output_gemm3d;
    kernel_info.reinterpret_input_as_3d = reinterpret_input_as_3d;
    kernel_info.broadcast_bias          = broadcast_bias;
    kernel_info.activation_info         = gemm_info.activation_info();

    // Set the target for the kernels
    _mm_kernel.set_target(gpu_target);

    // Manage intermediate buffers
    if(!_reshape_b_only_on_first_run)
    {
        _memory_group.manage(&_tmp_b);
    }

    GEMMLHSMatrixInfo lhs_info{};
    GEMMRHSMatrixInfo rhs_info{};

    // Pick up the GEMM configuration
    std::unique_ptr<ICLGEMMKernelConfiguration> gemm_config = CLGEMMReshapedOnlyRHSKernelConfigurationFactory::create(gpu_target);
    ARM_COMPUTE_ERROR_ON_NULLPTR(gemm_config.get());

    // Configure lhs_info and rhs_info
    std::tie(lhs_info, rhs_info) = gemm_config->configure(m, n, k, batch_size, data_type);

    _reshape_rhs_kernel.configure(b, &_tmp_b, rhs_info);

    // Configure and tune matrix multiply kernel
    _mm_reshaped_only_rhs_kernel.configure(a, &_tmp_b, c, output, alpha, beta, lhs_info, rhs_info, kernel_info);

    if(!_reshape_b_only_on_first_run)
    {
        _tmp_b.allocator()->allocate();
    }
}

Status CLGEMM::validate_native(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, const ITensorInfo *output, float alpha, float beta, const GEMMInfo &gemm_info)
{
    ARM_COMPUTE_UNUSED(alpha);
    ARM_COMPUTE_UNUSED(output);

    // Get the GPU target
    const GPUTarget    gpu_target              = CLScheduler::get().target();
    bool               reinterpret_input_as_3d = gemm_info.reinterpret_input_as_3d();
    const unsigned int m                       = reinterpret_input_as_3d ? (a->dimension(1) * a->dimension(2)) : a->dimension(1);
    const unsigned int n                       = b->dimension(0);
    const unsigned int k                       = a->dimension(0);
    const int          depth_output_gemm3d     = gemm_info.depth_output_gemm3d();

    const GEMMReshapeInfo reshape_info = GEMMReshapeInfo(m, n, k, 1, 1, depth_output_gemm3d, reinterpret_input_as_3d, gemm_info.broadcast_bias());

    // Validate matrix multiply
    ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMMatrixMultiplyKernel::validate(a, b, c, output, alpha, beta,
                                                                     false, reshape_info, gpu_target, gemm_info.fp_mixed_precision(), gemm_info.activation_info()));

    return Status{};
}

Status CLGEMM::validate_reshaped_v1(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, const ITensorInfo *output, float alpha, float beta, const GEMMInfo &gemm_info)
{
    ARM_COMPUTE_UNUSED(alpha);
    ARM_COMPUTE_UNUSED(output);

    TensorInfo tmp_a_info{};
    TensorInfo tmp_b_info{};

    // Get the GPU target
    const GPUTarget    gpu_target                = CLScheduler::get().target();
    const unsigned int m                         = gemm_info.reinterpret_input_as_3d() ? (a->dimension(1) * a->dimension(2)) : a->dimension(1);
    const unsigned int n                         = b->dimension(0);
    const unsigned int k                         = a->dimension(0);
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

    const GEMMReshapeInfo reshape_info = GEMMReshapeInfo(m, n, k, mult_transpose1xW_width, mult_interleave4x4_height, depth_output_gemm3d, false, gemm_info.broadcast_bias());

    // Validate interleave kernel
    auto_init_if_empty(tmp_a_info, a->clone()->set_tensor_shape(compute_lhs_reshaped_shape(*a, lhs_info, gemm_info.reinterpret_input_as_3d())));
    ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMReshapeLHSMatrixKernel::validate(a, &tmp_a_info, lhs_info, gemm_info.reinterpret_input_as_3d()));

    // Validate transpose kernel
    auto_init_if_empty(tmp_b_info, b->clone()->set_tensor_shape(compute_rhs_reshaped_shape(*b, rhs_info)));
    ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMReshapeRHSMatrixKernel::validate(b, &tmp_b_info, rhs_info));

    // Validate matrix multiply
    ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMMatrixMultiplyKernel::validate(&tmp_a_info, &tmp_b_info, c, output, alpha, beta,
                                                                     true, reshape_info, gpu_target, gemm_info.fp_mixed_precision(), gemm_info.activation_info()));

    return Status{};
}

Status CLGEMM::validate_reshaped_v2(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, const ITensorInfo *output, float alpha, float beta, const GEMMInfo &gemm_info)
{
    ARM_COMPUTE_UNUSED(alpha);
    ARM_COMPUTE_UNUSED(output);

    TensorInfo tmp_a_info{};
    TensorInfo tmp_b_info{};

    // Get the GPU target
    const GPUTarget    gpu_target              = CLScheduler::get().target();
    DataType           data_type               = a->data_type();
    bool               reinterpret_input_as_3d = gemm_info.reinterpret_input_as_3d();
    const unsigned int m                       = reinterpret_input_as_3d ? (a->dimension(1) * a->dimension(2)) : a->dimension(1);
    const unsigned int n                       = b->dimension(0);
    const unsigned int k                       = a->dimension(0);
    const unsigned int batch_size              = reinterpret_input_as_3d ? a->dimension(3) : a->dimension(2);
    const int          depth_output_gemm3d     = gemm_info.depth_output_gemm3d();
    const bool         broadcast_bias          = gemm_info.broadcast_bias();

    GEMMKernelInfo kernel_info;
    kernel_info.m                       = m;
    kernel_info.n                       = n;
    kernel_info.k                       = k;
    kernel_info.depth_output_gemm3d     = depth_output_gemm3d;
    kernel_info.reinterpret_input_as_3d = false;
    kernel_info.broadcast_bias          = broadcast_bias;
    kernel_info.activation_info         = gemm_info.activation_info();

    GEMMLHSMatrixInfo lhs_info;
    GEMMRHSMatrixInfo rhs_info;

    // Pick up the GEMM configuration
    std::unique_ptr<ICLGEMMKernelConfiguration> gemm_config = CLGEMMReshapedKernelConfigurationFactory::create(gpu_target);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(gemm_config.get());

    // Configure lhs_info and rhs_info
    std::tie(lhs_info, rhs_info) = gemm_config->configure(m, n, k, batch_size, data_type);

    auto_init_if_empty(tmp_a_info, a->clone()->set_tensor_shape(compute_lhs_reshaped_shape(*a, lhs_info, gemm_info.reinterpret_input_as_3d())));
    ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMReshapeLHSMatrixKernel::validate(a, &tmp_a_info, lhs_info, gemm_info.reinterpret_input_as_3d()));

    auto_init_if_empty(tmp_b_info, b->clone()->set_tensor_shape(compute_rhs_reshaped_shape(*b, rhs_info)));
    ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMReshapeRHSMatrixKernel::validate(b, &tmp_b_info, rhs_info));

    // Validate matrix multiply
    ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMMatrixMultiplyReshapedKernel::validate(&tmp_a_info, &tmp_b_info, c, output, alpha, beta, lhs_info, rhs_info, kernel_info));

    return Status{};
}

Status CLGEMM::validate_reshaped_only_rhs(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, const ITensorInfo *output, float alpha, float beta, const GEMMInfo &gemm_info)
{
    ARM_COMPUTE_UNUSED(alpha);
    ARM_COMPUTE_UNUSED(output);

    TensorInfo tmp_b_info{};

    // Get the GPU target
    const GPUTarget    gpu_target              = CLScheduler::get().target();
    const DataType     data_type               = a->data_type();
    bool               reinterpret_input_as_3d = gemm_info.reinterpret_input_as_3d();
    const unsigned int m                       = reinterpret_input_as_3d ? (a->dimension(1) * a->dimension(2)) : a->dimension(1);
    const unsigned int n                       = b->dimension(0);
    const unsigned int k                       = a->dimension(0);
    const unsigned int batch_size              = reinterpret_input_as_3d ? a->dimension(3) : a->dimension(2);
    const int          depth_output_gemm3d     = gemm_info.depth_output_gemm3d();
    const bool         broadcast_bias          = gemm_info.broadcast_bias();

    GEMMKernelInfo kernel_info;
    kernel_info.m                       = m;
    kernel_info.n                       = n;
    kernel_info.k                       = k;
    kernel_info.depth_output_gemm3d     = depth_output_gemm3d;
    kernel_info.reinterpret_input_as_3d = reinterpret_input_as_3d;
    kernel_info.broadcast_bias          = broadcast_bias;
    kernel_info.activation_info         = gemm_info.activation_info();

    GEMMLHSMatrixInfo lhs_info;
    GEMMRHSMatrixInfo rhs_info;

    // Pick up the GEMM configuration
    std::unique_ptr<ICLGEMMKernelConfiguration> gemm_config = CLGEMMReshapedOnlyRHSKernelConfigurationFactory::create(gpu_target);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(gemm_config.get());

    // Configure lhs_info and rhs_info
    std::tie(lhs_info, rhs_info) = gemm_config->configure(m, n, k, batch_size, data_type);

    auto_init_if_empty(tmp_b_info, b->clone()->set_tensor_shape(compute_rhs_reshaped_shape(*b, rhs_info)));
    ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMReshapeRHSMatrixKernel::validate(b, &tmp_b_info, rhs_info));

    // Validate matrix multiply
    ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMMatrixMultiplyReshapedOnlyRHSKernel::validate(a, &tmp_b_info, c, output, alpha, beta, lhs_info, rhs_info, kernel_info));

    return Status{};
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

    // Get the GPU target
    const GPUTarget    gpu_target              = CLScheduler::get().target();
    bool               reinterpret_input_as_3d = gemm_info.reinterpret_input_as_3d();
    const unsigned int m                       = reinterpret_input_as_3d ? (a->info()->dimension(1) * a->info()->dimension(2)) : a->info()->dimension(1);
    const unsigned int n                       = b->info()->dimension(0);
    const unsigned int k                       = a->info()->dimension(0);

    // Select GEMMType
    _gemm_type = select_gemm_type(m, n, k, a->info()->data_type(), _reshape_b_only_on_first_run, gpu_target);

    const bool fuse_add_c = (!(helpers::float_ops::is_zero(beta)) && c != nullptr);

    const ICLTensor *c_to_use = fuse_add_c ? c : nullptr;

    switch(_gemm_type)
    {
        case GEMMType::NATIVE:
        {
            configure_native(a, b, c_to_use, output, alpha, beta, gemm_info);
            break;
        }
        case GEMMType::RESHAPED_V1:
        {
            configure_reshaped_v1(a, b, c_to_use, output, alpha, beta, gemm_info);
            break;
        }
        case GEMMType::RESHAPED_V2:
        {
            configure_reshaped_v2(a, b, c_to_use, output, alpha, beta, gemm_info);
            break;
        }
        case GEMMType::RESHAPED_ONLY_RHS:
        {
            configure_reshaped_only_rhs(a, b, c_to_use, output, alpha, beta, gemm_info);
            break;
        }
        default:
        {
            ARM_COMPUTE_ERROR("GEMMType not supported");
        }
    }
}

Status CLGEMM::validate(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, const ITensorInfo *output, float alpha, float beta, const GEMMInfo &gemm_info)
{
    // Get the GPU target
    const GPUTarget    gpu_target              = CLScheduler::get().target();
    bool               reinterpret_input_as_3d = gemm_info.reinterpret_input_as_3d();
    const unsigned int m                       = reinterpret_input_as_3d ? (a->dimension(1) * a->dimension(2)) : a->dimension(1);
    const unsigned int n                       = b->dimension(0);
    const unsigned int k                       = a->dimension(0);

    // Select GEMMType
    GEMMType gemm_type = select_gemm_type(m, n, k, a->data_type(), gemm_info.reshape_b_only_on_first_run(), gpu_target);

    const bool fuse_add_c = (!(helpers::float_ops::is_zero(beta)) && c != nullptr);

    const ITensorInfo *c_to_use = fuse_add_c ? c : nullptr;

    switch(gemm_type)
    {
        case GEMMType::NATIVE:
        {
            ARM_COMPUTE_RETURN_ON_ERROR(validate_native(a, b, c_to_use, output, alpha, beta, gemm_info));
            break;
        }
        case GEMMType::RESHAPED_V1:
        {
            ARM_COMPUTE_RETURN_ON_ERROR(validate_reshaped_v1(a, b, c_to_use, output, alpha, beta, gemm_info));
            break;
        }
        case GEMMType::RESHAPED_V2:
        {
            ARM_COMPUTE_RETURN_ON_ERROR(validate_reshaped_v2(a, b, c_to_use, output, alpha, beta, gemm_info));
            break;
        }
        case GEMMType::RESHAPED_ONLY_RHS:
        {
            ARM_COMPUTE_RETURN_ON_ERROR(validate_reshaped_only_rhs(a, b, c_to_use, output, alpha, beta, gemm_info));
            break;
        }
        default:
        {
            ARM_COMPUTE_RETURN_ERROR_MSG("GEMMType not supported");
        }
    }

    return Status{};
}

void CLGEMM::run()
{
    prepare();

    MemoryGroupResourceScope scope_mg(_memory_group);

    // Run matrix multiply kernel
    switch(_gemm_type)
    {
        case GEMMType::NATIVE:
        {
            CLScheduler::get().enqueue(_mm_kernel, true);
            break;
        }
        case GEMMType::RESHAPED_V1:
        {
            // Run interleave kernel
            CLScheduler::get().enqueue(_reshape_lhs_kernel, false);

            if(!_reshape_b_only_on_first_run)
            {
                // Run transpose kernel
                CLScheduler::get().enqueue(_reshape_rhs_kernel, false);
            }

            CLScheduler::get().enqueue(_mm_kernel, true);
            break;
        }
        case GEMMType::RESHAPED_V2:
        {
            // Run interleave kernel
            CLScheduler::get().enqueue(_reshape_lhs_kernel, false);

            if(!_reshape_b_only_on_first_run)
            {
                // Run transpose kernel
                CLScheduler::get().enqueue(_reshape_rhs_kernel, false);
            }

            CLScheduler::get().enqueue(_mm_reshaped_kernel, true);
            break;
        }
        case GEMMType::RESHAPED_ONLY_RHS:
        {
            if(!_reshape_b_only_on_first_run)
            {
                // Run transpose kernel
                CLScheduler::get().enqueue(_reshape_rhs_kernel, false);
            }

            CLScheduler::get().enqueue(_mm_reshaped_only_rhs_kernel, true);
            break;
        }
        default:
        {
            ARM_COMPUTE_ERROR("GEMMType not supported");
        }
    }
}

void CLGEMM::prepare()
{
    if(!_is_prepared)
    {
        if(_gemm_type != GEMMType::NATIVE && _reshape_b_only_on_first_run)
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
