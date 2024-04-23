#include "src/cpu/operators/CpuLinear.h"

#include "arm_compute/runtime/NEON/NEScheduler.h"

#include "src/common/IOperator.h"
#include "src/common/utils/LegacySupport.h"
#include "src/common/utils/Log.h"
#include "src/cpu/CpuContext.h"
#include "src/cpu/kernels/CpuLinearKernel.h"

#include "src/core/helpers/MemoryHelpers.h"
#include "src/cpu/utils/CpuAuxTensorHandler.h"

namespace arm_compute
{
namespace cpu
{
void CpuLinear::configure(const ITensorInfo *a,
                          const ITensorInfo *b,
                          const ITensorInfo *c,
                          ITensorInfo       *d,
                          float              alpha,
                          float              beta, const LinearLayerInfo &linear_info)
{
    ARM_COMPUTE_LOG_PARAMS(a, b, c, d, alpha, beta, linear_info);
    ARM_COMPUTE_UNUSED(linear_info);

    const bool is_c_bias = beta == 1 && c != nullptr;
    const bool run_optimised = false;

    _run_vector_matrix_multiplication = a->dimension(1) < 2;
    _run_bias_addition                = is_c_bias;
    _reshape_b_only_on_first_run      = b->are_values_constant();
    
    if (run_optimised)
    {
        _run_interleave_transpose   = false;

    }else /* Normal matrix multiplication*/
    {
        _run_interleave_transpose = !_run_vector_matrix_multiplication;

        // Pick output tensor in case bias addition should be performed
        ITensorInfo *gemm_output_to_use = (_run_bias_addition) ? &_tmp_d : d;
        // Pick b tensor in case pretranspose should be performed
        const ITensorInfo *b_to_use = b;

        _mm_kernel = std::make_unique<cpu::kernels::CpuGemmMatrixMultiplyKernel>();
        
        if (_run_vector_matrix_multiplication)
        {
            // Configure the matrix multiply kernel
            _mm_kernel->configure(a, b_to_use, gemm_output_to_use, alpha, false);
        }
        else
        {
            // Configure interleave kernel
            _interleave_kernel = std::make_unique<cpu::kernels::CpuGemmInterleave4x4Kernel>();
            _interleave_kernel->configure(a, &_tmp_a);
            _aux_mem[InterleavedLHS] =
                experimental::MemoryInfo(offset_int_vec(InterleavedLHS), experimental::MemoryLifetime::Temporary, _tmp_a.total_size());
            
            // Configure rhs transpose1xw kernel
            _transpose1xW_b_kernel = std::make_unique<cpu::kernels::CpuGemmTranspose1xWKernel>();
            _transpose1xW_b_kernel->configure(b_to_use, &_tmp_b);
            _aux_mem[Transposed1xWRHS] =
                experimental::MemoryInfo(offset_int_vec(Transposed1xWRHS), experimental::MemoryLifetime::Persistent, _tmp_b.total_size());
            
            // Use a and b here instead of _tmp_a and _tmp_b because CpuGemmMatrixMultiplyKernel requires the original m,n,k in case of interleaved a and transposed1xw b
            const int m = a->dimension(1);
            const int n = b_to_use->dimension(0);
            const int k = a->dimension(0);

            // Configure matrix multiplication kernel
            _mm_kernel->configure(&_tmp_a, &_tmp_b, gemm_output_to_use, alpha, _run_interleave_transpose,
                                  GEMMReshapeInfo(m, n, k));
        }
    }

    
}

Status
CpuLinear::validate(const ITensorInfo *a,
                    const ITensorInfo *b,
                    const ITensorInfo *c,
                    ITensorInfo       *d,
                    float              alpha,
                    float              beta, const LinearLayerInfo &linear_info)
{
    ARM_COMPUTE_UNUSED(a);
    ARM_COMPUTE_UNUSED(b);
    ARM_COMPUTE_UNUSED(c);
    ARM_COMPUTE_UNUSED(d);
    ARM_COMPUTE_UNUSED(alpha);
    ARM_COMPUTE_UNUSED(beta);
    ARM_COMPUTE_UNUSED(linear_info);
    return Status{};
}

void CpuLinear::run(ITensorPack &tensors)
{

    ARM_COMPUTE_ERROR_ON_MSG(tensors.empty(), "No inputs provided");
    auto a = tensors.get_const_tensor(ACL_SRC_0);
    auto b = tensors.get_const_tensor(ACL_SRC_1);
    //auto c = tensors.get_const_tensor(ACL_SRC_2);
    auto d = tensors.get_tensor(ACL_DST);

    std::cout << "src/cpu/operators/CpuLinear.cpp " << std::endl;
    std::cout << "a->info()->valid_region().shape.x()" << a->info()->valid_region().shape.x() << std::endl;
    std::cout << "a->info()->valid_region().shape.y()" << a->info()->valid_region().shape.y() << std::endl;
    std::cout << "a->info()->valid_region().shape.z()" << a->info()->valid_region().shape.z() << std::endl;

    std::cout << "b->info()->valid_region().shape.x()" << b->info()->valid_region().shape.x() << std::endl;
    std::cout << "b->info()->valid_region().shape.y()" << b->info()->valid_region().shape.y() << std::endl;
    std::cout << "b->info()->valid_region().shape.z()" << b->info()->valid_region().shape.z() << std::endl;

    for(auto i : a->info()->valid_strides_in_bytes())std::cout << i << std::endl;

    std::cout << *reinterpret_cast<float *>(a->ptr_to_element(Coordinates(0,0)))  
              << " " 
              << *reinterpret_cast<float *>(a->ptr_to_element(Coordinates(0,7)))
              << std::endl;

    CpuAuxTensorHandler interleaved_a(offset_int_vec(InterleavedLHS), _tmp_a, tensors, true);
    CpuAuxTensorHandler transposed1xw_b(offset_int_vec(Transposed1xWRHS), _tmp_b, tensors, true);
    CpuAuxTensorHandler temp_d(offset_int_vec(TempResult), _tmp_d, tensors, true);

    ITensorPack mm_pack{{ACL_SRC_0, a}, {ACL_SRC_1, b}, {ACL_DST, (_run_bias_addition) ? temp_d.get() : d}};

    if (_run_interleave_transpose)
    {
        // Run interleave kernel
        ITensorPack interleave_pack{{ACL_SRC, a}, {ACL_DST, interleaved_a.get()}};
        NEScheduler::get().schedule_op(_interleave_kernel.get(), Window::DimY, _interleave_kernel->window(),
                                        interleave_pack);
        // Use reshaped matrices
        mm_pack.add_const_tensor(ACL_SRC_0, interleaved_a.get());
    }

    const ITensor *b_to_use = b;
    if (_run_interleave_transpose)
    {
        if (!_reshape_b_only_on_first_run)
        {
            // Run transpose1xw kernel
            ITensorPack transpose_pack{{ACL_SRC, b_to_use}, {ACL_DST, transposed1xw_b.get()}};
            NEScheduler::get().schedule_op(_transpose1xW_b_kernel.get(), Window::DimY,
                                            _transpose1xW_b_kernel->window(), transpose_pack);
        }
        b_to_use = transposed1xw_b.get();
    }
    // Use reshaped matrices
    mm_pack.add_const_tensor(ACL_SRC_1, b_to_use);
    
    NEScheduler::get().schedule_op(_mm_kernel.get(),
                                _run_vector_matrix_multiplication ? Window::DimX : Window::DimY,
                                _mm_kernel->window(), mm_pack);

}


} // namespace cpu
} // namespace arm_compute
