#include "src/cpu/operators/CpuScaleDotProduction.h"

#include "arm_compute/runtime/NEON/NEScheduler.h"

#include "src/common/IOperator.h"
#include "src/common/utils/LegacySupport.h"
#include "src/common/utils/Log.h"
#include "src/cpu/CpuContext.h"

#include "src/core/helpers/MemoryHelpers.h"
#include "src/cpu/utils/CpuAuxTensorHandler.h"


namespace arm_compute
{
namespace cpu
{

void CpuScaleDotProduction::configure(const ITensorInfo *key,
                                      const ITensorInfo *value,
                                      const ITensorInfo *query,
                                      ITensorInfo *output,
                                      const ScaleDotProductionAttentionLayerInfo& info)
{
    ARM_COMPUTE_LOG_PARAMS(key, value, query, output);

    _run_vector_matrix_multiplication   = key->dimension(1) < 2;
    _run_pretranspose                   = false;
    
    float   scale = sqrt(info.d_model());
    //float   v_alpha = 1.f;
    _run_scale = scale != 1.f;

    // Pick key tensor in case pretranspose should be performed
    const ITensorInfo *key_to_use = key;
    ITensorInfo *output_to_use = output;

    /* Pretranspose Key, K=K^T */
    _pretranspose_key_func = std::make_unique<CpuTranspose>();
    _pretranspose_key_func->configure(key_to_use, &_pretransposed_key);

    _aux_mem[PreTransposedRHS] =
                experimental::MemoryInfo(offset_int_vec(PreTransposedRHS), experimental::MemoryLifetime::Persistent, _pretransposed_key.total_size());
    key_to_use = &_pretransposed_key;
    

    /* Matrix multiply query and key */
    _mm_kernel1 = std::make_unique<cpu::kernels::CpuGemmMatrixMultiplyKernel>();
    /* Matrix multiply of production and value */
   // _mm_kernel2 = std::make_unique<cpu::kernels::CpuGemmMatrixMultiplyKernel>();

    // Select between GEMV and GEMM
    if (_run_vector_matrix_multiplication)
    {
        // Configure the matrix multiply kernel
        _mm_kernel1->configure(query, key_to_use, output_to_use, scale, false);
    }
    else
    {
        _run_interleave_transpose = !_run_vector_matrix_multiplication;
        // Configure interleave kernel
        _interleave_kernel = std::make_unique<cpu::kernels::CpuGemmInterleave4x4Kernel>();
        _interleave_kernel->configure(query, &_tmp_query);
        _aux_mem[InterleavedLHS] =
            experimental::MemoryInfo(offset_int_vec(InterleavedLHS), experimental::MemoryLifetime::Persistent, _tmp_query.total_size());

        // Configure rhs transpose1xw kernel
        _transpose1xW_key_kernel = std::make_unique<cpu::kernels::CpuGemmTranspose1xWKernel>();
        _transpose1xW_key_kernel->configure(key_to_use, &_tmp_key);
        _aux_mem[Transposed1xWRHS] =
            experimental::MemoryInfo(offset_int_vec(Transposed1xWRHS),experimental::MemoryLifetime::Persistent, _tmp_key.total_size());

        // Use a and b here instead of _tmp_a and _tmp_b because CpuGemmMatrixMultiplyKernel requires the original m,n,k in case of interleaved a and transposed1xw b
        const int m = query->dimension(1);
        const int n = key_to_use->dimension(0);
        const int k = query->dimension(0);

        // Configure matrix multiplication kernel
        _mm_kernel1->configure(&_tmp_query, &_tmp_key, &_tmp_scaled, scale, _run_interleave_transpose,
                                GEMMReshapeInfo(m, n, k));
        _aux_mem[ScaledOutput] =
            experimental::MemoryInfo(offset_int_vec(ScaledOutput),experimental::MemoryLifetime::Persistent, _tmp_scaled.total_size());

    }

    /* Softmax function apply to scaled product */
    _softmax_func = std::make_unique<CpuSoftmaxGeneric>();
    _softmax_func->configure(&_tmp_scaled,output_to_use);
    
    ARM_COMPUTE_UNUSED(value);
    ARM_COMPUTE_UNUSED(query);
    ARM_COMPUTE_UNUSED(output);


}

Status
CpuScaleDotProduction::validate(const ITensorInfo *key, const ITensorInfo *value, const ITensorInfo *query, ITensorInfo *output)
{
    ARM_COMPUTE_UNUSED(key);
    ARM_COMPUTE_UNUSED(value);
    ARM_COMPUTE_UNUSED(query);
    ARM_COMPUTE_UNUSED(output);
    return Status{};
}

void CpuScaleDotProduction::run(ITensorPack &tensors)
{
    ARM_COMPUTE_UNUSED(tensors);
    auto key    = tensors.get_const_tensor(ACL_SRC_0);
    auto value  = tensors.get_const_tensor(ACL_SRC_1);
    auto query  = tensors.get_const_tensor(ACL_SRC_2);
    auto output = tensors.get_tensor(ACL_DST);

    const ITensor *key_to_use = key;

    CpuAuxTensorHandler pretransposed_key(offset_int_vec(PreTransposedRHS), _pretransposed_key, tensors);
    CpuAuxTensorHandler interleaved_query(offset_int_vec(InterleavedLHS), _tmp_query, tensors, true);
    CpuAuxTensorHandler transposed1xw_key(offset_int_vec(Transposed1xWRHS), _tmp_key, tensors, true);
    CpuAuxTensorHandler scaled_output(offset_int_vec(ScaledOutput), _tmp_scaled, tensors, true);

    ITensorPack mm_pack{{ACL_SRC_0, query}, {ACL_SRC_1, key}, {ACL_DST, output}};

    if (_run_interleave_transpose)
    {
        // Run interleave kernel
        ITensorPack interleave_pack{{ACL_SRC, query}, {ACL_DST, interleaved_query.get()}};
        NEScheduler::get().schedule_op(_interleave_kernel.get(), Window::DimY, _interleave_kernel->window(),
                                        interleave_pack);
        // Use reshaped matrices
        mm_pack.add_const_tensor(ACL_SRC_0, interleaved_query.get());
    }

    if (_pretranspose_key_func && _run_pretranspose)
    {
        // Run pretranspose kernel
        ITensorPack pretranspose_pack{{ACL_SRC, key_to_use}, {ACL_DST, pretransposed_key.get()}};
        _pretranspose_key_func->run(pretranspose_pack);
        key_to_use = pretransposed_key.get();
    }

    if (_run_interleave_transpose)
    {
        // Run transpose1xw kernel
        ITensorPack transpose_pack{{ACL_SRC, key_to_use}, {ACL_DST, transposed1xw_key.get()}};
        NEScheduler::get().schedule_op(_transpose1xW_key_kernel.get(), Window::DimY,
                                        _transpose1xW_key_kernel->window(), transpose_pack);
        key_to_use = transposed1xw_key.get();
    }

    // Use reshaped matrices
    mm_pack.add_const_tensor(ACL_SRC_1, key_to_use);
    mm_pack.add_tensor(ACL_DST, scaled_output.get());

    NEScheduler::get().schedule_op(_mm_kernel1.get(),
                                    _run_vector_matrix_multiplication ? Window::DimX : Window::DimY,
                                    _mm_kernel1->window(), mm_pack);

    ITensorPack softmax_pack{{ACL_SRC, const_cast<const ITensor*>(scaled_output.get())}, {ACL_DST, output}};
    _softmax_func->run(softmax_pack);

    std::cout << "src/cpu/operators/CpuScaleDotProduction.cpp 6" << std::endl;

    ARM_COMPUTE_UNUSED(value);
    ARM_COMPUTE_UNUSED(query);
    ARM_COMPUTE_UNUSED(output);

}

experimental::MemoryRequirements CpuScaleDotProduction::workspace() const
{
    return _aux_mem;
}

} // namespace cpu
} // namespace arm_compute
