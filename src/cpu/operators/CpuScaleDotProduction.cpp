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

void CpuScaleDotProduction::configure(const ITensorInfo *query,
                                      const ITensorInfo *key,
                                      const ITensorInfo *value,
                                      ITensorInfo *output,
                                      const ScaleDotProductionAttentionLayerInfo& info)
{
    ARM_COMPUTE_LOG_PARAMS(key, value, query, output);
    ARM_COMPUTE_UNUSED(output);
    
    // Query multi-Head reshape 
    TensorShape query_reshape = TensorShape(query->tensor_shape().x()/info.h(),
                                            info.h(),
                                            query->tensor_shape().y(),
                                            1);
    _reshaped_query = query->clone()->set_tensor_shape(query_reshape);

    TensorShape query_permute = TensorShape(query->tensor_shape().x()/info.h(),
                                            query->tensor_shape().y(),
                                            info.h(),
                                            1);
    _permuted_query = query->clone()->set_tensor_shape(query_permute);

    _query_reshape_kernel = std::make_unique<kernels::CpuReshapeKernel>();
    _query_reshape_kernel->configure(query, &_reshaped_query);

    _query_permute_func = std::make_unique<CpuPermute>();
    _query_permute_func->configure(&_reshaped_query, &_permuted_query, PermutationVector(0U, 2U, 1U));


    // Key multi-Head reshape 
    TensorShape key_reshape = TensorShape(key->tensor_shape().x()/info.h(),
                                          info.h(),
                                          key->tensor_shape().y(),
                                          1);
    _reshaped_key = key->clone()->set_tensor_shape(key_reshape);

    TensorShape key_permute = TensorShape(key->tensor_shape().x()/info.h(),
                                          key->tensor_shape().y(),
                                          info.h(),
                                          1);
    _permuted_key = key->clone()->set_tensor_shape(key_permute);

    _key_reshape_kernel = std::make_unique<kernels::CpuReshapeKernel>();
    _key_reshape_kernel->configure(key, &_reshaped_key);

    _key_permute_func = std::make_unique<CpuPermute>();
    _key_permute_func->configure(&_reshaped_key, &_permuted_key, PermutationVector(0U, 2U, 1U));

    // Pretranspose Key, K=K^T 
    _key_transpose_func = std::make_unique<CpuTranspose>();
    _key_transpose_func->configure(&_permuted_key, &_transposed_key);


    // Configure interleave kernel
    _interleave_kernel = std::make_unique<cpu::kernels::CpuGemmInterleave4x4Kernel>();
    _interleave_kernel->configure(&_permuted_query, &_tmp_query);
    _aux_mem[InterleavedLHS] =
        experimental::MemoryInfo(offset_int_vec(InterleavedLHS), experimental::MemoryLifetime::Persistent, _tmp_query.total_size());
    
    // Configure rhs transpose1xw kernel
    _transpose1xW_key_kernel = std::make_unique<cpu::kernels::CpuGemmTranspose1xWKernel>();
    _transpose1xW_key_kernel->configure(&_transposed_key, &_tmp_key);
    _aux_mem[Transposed1xWRHS] =
        experimental::MemoryInfo(offset_int_vec(Transposed1xWRHS),experimental::MemoryLifetime::Persistent, _tmp_key.total_size());

    // Matrix multiply compute multi-head attention between Query and Key
    _mm_kernel = std::make_unique<cpu::kernels::CpuGemmMatrixMultiplyKernel>();
    const int m = _permuted_query.dimension(1);
    const int n = _transposed_key.dimension(0);
    const int k = _permuted_query.dimension(0);
    const float scale = 1.0f/sqrt(info.h());
    _mm_kernel->configure(&_tmp_query,&_tmp_key,&_scaled_query_key,scale,true,GEMMReshapeInfo(m, n, k));

    
    ARM_COMPUTE_UNUSED(value);
    ARM_COMPUTE_UNUSED(query);
    ARM_COMPUTE_UNUSED(output);


}

Status
CpuScaleDotProduction::validate(const ITensorInfo *query, const ITensorInfo *key, const ITensorInfo *value, ITensorInfo *output)
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

    auto query    = tensors.get_const_tensor(ACL_SRC_0);
    auto key  = tensors.get_const_tensor(ACL_SRC_1);
    auto value  = tensors.get_const_tensor(ACL_SRC_2);
    auto output = tensors.get_tensor(ACL_DST);

    std::cout << "src/cpu/operators/CpuScaleDotProduction.cpp " << std::endl;

    std::cout <<"query x: " << query->info()->tensor_shape().x() << std::endl;
    std::cout <<"query y: " << query->info()->tensor_shape().y() << std::endl;
    std::cout <<"query z: " << query->info()->tensor_shape().z() << std::endl;
    std::cout << *reinterpret_cast<float *>(query->ptr_to_element(Coordinates(0,0)))  << std::endl;
    std::cout << *reinterpret_cast<float *>(query->ptr_to_element(Coordinates(767,6)))  << std::endl;

    std::cout <<"key x: " << key->info()->tensor_shape().x() << std::endl;
    std::cout <<"key y: " << key->info()->tensor_shape().y() << std::endl;
    std::cout <<"key z: " << key->info()->tensor_shape().z() << std::endl;
    std::cout << *reinterpret_cast<float *>(key->ptr_to_element(Coordinates(0,0)))  << std::endl;
    std::cout << *reinterpret_cast<float *>(key->ptr_to_element(Coordinates(767,6)))  << std::endl;

    std::cout <<"value x: " << value->info()->tensor_shape().x() << std::endl;
    std::cout <<"value y: " << value->info()->tensor_shape().y() << std::endl;
    std::cout <<"value z: " << value->info()->tensor_shape().z() << std::endl;
    std::cout << *reinterpret_cast<float *>(value->ptr_to_element(Coordinates(0,0)))  << std::endl;
    std::cout << *reinterpret_cast<float *>(value->ptr_to_element(Coordinates(767,6)))  << std::endl;


    CpuAuxTensorHandler reshaped_query(offset_int_vec(QueryReshape), _reshaped_query, tensors);
    CpuAuxTensorHandler permuted_query(offset_int_vec(QueryPermute), _permuted_query, tensors);
    CpuAuxTensorHandler reshaped_key(offset_int_vec(KeyReshape), _reshaped_key, tensors);
    CpuAuxTensorHandler permuted_key(offset_int_vec(KeyPermute), _permuted_key, tensors);
    CpuAuxTensorHandler transposed_key(offset_int_vec(KeyTranspose), _transposed_key, tensors);
    CpuAuxTensorHandler scaled_query_key(offset_int_vec(QueryKeyScale), _scaled_query_key, tensors);

    // Run Query multi-Head reshape 
    ITensorPack query_reshape_pack{{ACL_SRC_0, query},{ACL_DST, reshaped_query.get()}};
    const auto query_split_dimension = _query_reshape_kernel->get_split_dimension();
    NEScheduler::get().schedule_op(_query_reshape_kernel.get(), query_split_dimension, _query_reshape_kernel->window(), query_reshape_pack);

    ITensorPack query_permute_pack{{ACL_SRC, reshaped_query.get()},{ACL_DST, permuted_query.get()}};
    _query_permute_func->run(query_permute_pack);

    
    // Run Key multi-Head reshape 
    ITensorPack key_reshape_pack{{ACL_SRC_0, key},{ACL_DST, reshaped_key.get()}};
    const auto key_split_dimension = _key_reshape_kernel->get_split_dimension();
    NEScheduler::get().schedule_op(_key_reshape_kernel.get(), key_split_dimension, _key_reshape_kernel->window(), key_reshape_pack);

    ITensorPack key_permute_pack{{ACL_SRC, reshaped_key.get()},{ACL_DST, permuted_key.get()}};
    _key_permute_func->run(key_permute_pack);

    ITensorPack key_transpose_pack{{ACL_SRC, permuted_key.get()}, {ACL_DST, transposed_key.get()}};
    _key_transpose_func->run(key_transpose_pack);





    CpuAuxTensorHandler interleaved_query(offset_int_vec(InterleavedLHS), _tmp_query, tensors, true);
    CpuAuxTensorHandler transposed1xw_key(offset_int_vec(Transposed1xWRHS), _tmp_key, tensors, true);

    // Run interleave kernel
    ITensorPack interleave_pack{{ACL_SRC, permuted_query.get()}, {ACL_DST, interleaved_query.get()}};
    NEScheduler::get().schedule_op(_interleave_kernel.get(), Window::DimY, _interleave_kernel->window(),
                                    interleave_pack);

    // Run transpose1xw kernel
    ITensorPack transpose_pack{{ACL_SRC, transposed_key.get()}, {ACL_DST, transposed1xw_key.get()}};
    NEScheduler::get().schedule_op(_transpose1xW_key_kernel.get(), Window::DimY,
                                    _transpose1xW_key_kernel->window(), transpose_pack);


    // Run matrix multiply compute multi-head attention between Query and Key
    ITensorPack gemm_QK_pack{{ACL_SRC_0, interleaved_query.get()}, {ACL_SRC_1, transposed1xw_key.get()}, {ACL_DST, scaled_query_key.get()}};
    NEScheduler::get().schedule_op(_mm_kernel.get(),Window::DimZ,_mm_kernel->window(),gemm_QK_pack);



    std::cout <<"scaled_query_key.get() x: " << scaled_query_key.get()->info()->tensor_shape().x() << std::endl;
    std::cout <<"scaled_query_key.get() y: " << scaled_query_key.get()->info()->tensor_shape().y() << std::endl;
    std::cout <<"scaled_query_key.get() z: " << scaled_query_key.get()->info()->tensor_shape().z() << std::endl;
    std::cout << *reinterpret_cast<float *>(scaled_query_key.get()->ptr_to_element(Coordinates(0,0,0)))  << std::endl;
    std::cout << *reinterpret_cast<float *>(scaled_query_key.get()->ptr_to_element(Coordinates(0,1,0)))  << std::endl;
    std::cout << *reinterpret_cast<float *>(scaled_query_key.get()->ptr_to_element(Coordinates(0,0,1)))  << std::endl;
    std::cout << *reinterpret_cast<float *>(scaled_query_key.get()->ptr_to_element(Coordinates(6,0,0)))  << std::endl;
    std::cout << *reinterpret_cast<float *>(scaled_query_key.get()->ptr_to_element(Coordinates(7,0,0)))  << std::endl;


    /*
    const ITensor *key_to_use = key;

    CpuAuxTensorHandler pretransposed_key(offset_int_vec(PreTransposedRHS), _pretransposed_key, tensors);
    CpuAuxTensorHandler interleaved_query(offset_int_vec(InterleavedLHS), _tmp_query, tensors, true);
    CpuAuxTensorHandler transposed1xw_key(offset_int_vec(Transposed1xWRHS), _tmp_key, tensors, true);

    ITensorPack mm_pack{{ACL_SRC_0, query}, {ACL_SRC_1, key}, {ACL_DST, output}};
    std::cout << "src/cpu/operators/CpuScaleDotProduction.cpp " << std::endl;

    std::cout <<"key x: " << key->info()->tensor_shape().x() << std::endl;
    std::cout <<"key y: " << key->info()->tensor_shape().y() << std::endl;
    std::cout <<"key z: " << key->info()->tensor_shape().z() << std::endl;

    std::cout << *reinterpret_cast<float *>(key->ptr_to_element(Coordinates(0,0)))  << std::endl;
    std::cout << *reinterpret_cast<float *>(key->ptr_to_element(Coordinates(767,6)))  << std::endl;

    std::cout <<"query x: " << query->info()->tensor_shape().x() << std::endl;
    std::cout <<"query y: " << query->info()->tensor_shape().y() << std::endl;
    std::cout <<"query z: " << query->info()->tensor_shape().z() << std::endl;
    std::cout << *reinterpret_cast<float *>(query->ptr_to_element(Coordinates(0,0)))  << std::endl;
    std::cout << *reinterpret_cast<float *>(query->ptr_to_element(Coordinates(767,6)))  << std::endl;
    
    std::cout <<"value x: " << value->info()->tensor_shape().x() << std::endl;
    std::cout <<"value y: " << value->info()->tensor_shape().y() << std::endl;
    std::cout <<"value z: " << value->info()->tensor_shape().z() << std::endl;
    std::cout << *reinterpret_cast<float *>(value->ptr_to_element(Coordinates(0,0)))  << std::endl;
    std::cout << *reinterpret_cast<float *>(value->ptr_to_element(Coordinates(767,6)))  << std::endl;

    if (_run_interleave_transpose)
    {
        std::cout << "_run_interleave_transpose " << std::endl;
        // Run interleave kernel
        ITensorPack interleave_pack{{ACL_SRC, query}, {ACL_DST, interleaved_query.get()}};
        NEScheduler::get().schedule_op(_interleave_kernel.get(), Window::DimY, _interleave_kernel->window(),
                                        interleave_pack);
        // Use reshaped matrices
        mm_pack.add_const_tensor(ACL_SRC_0, interleaved_query.get());
    }

    if (_pretranspose_key_func && _run_pretranspose)
    {
        std::cout << "_pretranspose_key_func && _run_pretranspose " << std::endl;
        // Run pretranspose kernel
        ITensorPack pretranspose_pack{{ACL_SRC, key_to_use}, {ACL_DST, pretransposed_key.get()}};
        _pretranspose_key_func->run(pretranspose_pack);
        key_to_use = pretransposed_key.get();
    }

    if (_run_interleave_transpose)
    {
        std::cout << "_run_interleave_transpose " << std::endl;
        // Run transpose1xw kernel
        ITensorPack transpose_pack{{ACL_SRC, key_to_use}, {ACL_DST, transposed1xw_key.get()}};
        NEScheduler::get().schedule_op(_transpose1xW_key_kernel.get(), Window::DimY,
                                        _transpose1xW_key_kernel->window(), transpose_pack);
        key_to_use = transposed1xw_key.get();
    }

    // Use reshaped matrices
    mm_pack.add_const_tensor(ACL_SRC_1, key_to_use);

    NEScheduler::get().schedule_op(_mm_kernel.get(),
                                    _run_vector_matrix_multiplication ? Window::DimX : Window::DimY,
                                    _mm_kernel->window(), mm_pack);
    
    
    std::cout << *reinterpret_cast<const float *>(output->ptr_to_element(Coordinates(0,0))) << " "
              << *reinterpret_cast<const float *>(output->ptr_to_element(Coordinates(1,0))) << " " 
              << *reinterpret_cast<const float *>(output->ptr_to_element(Coordinates(2,0))) << " " 
              << *reinterpret_cast<const float *>(output->ptr_to_element(Coordinates(3,0))) << " " 
              
              << *reinterpret_cast<const float *>(output->ptr_to_element(Coordinates(767,0))) << " " 
              << *reinterpret_cast<const float *>(output->ptr_to_element(Coordinates(768,0))) << " " 
    << std::endl;

    std::cout << *reinterpret_cast<const float *>(output->ptr_to_element(Coordinates(0,0))) << " "
              << *reinterpret_cast<const float *>(output->ptr_to_element(Coordinates(0,1))) << " " 
              << *reinterpret_cast<const float *>(output->ptr_to_element(Coordinates(0,2))) << " " 
              << *reinterpret_cast<const float *>(output->ptr_to_element(Coordinates(0,3))) << " " 
              << *reinterpret_cast<const float *>(output->ptr_to_element(Coordinates(0,4))) << " "
              << *reinterpret_cast<const float *>(output->ptr_to_element(Coordinates(0,5))) << " " 
              << *reinterpret_cast<const float *>(output->ptr_to_element(Coordinates(0,6))) << " " 
              << *reinterpret_cast<const float *>(output->ptr_to_element(Coordinates(767,6))) << " "
    << std::endl; 
    */
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
