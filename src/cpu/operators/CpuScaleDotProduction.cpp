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
void CpuScaleDotProduction::configure(const ITensorInfo *key, const ITensorInfo *value, const ITensorInfo *query, ITensorInfo *output)
{
    ARM_COMPUTE_LOG_PARAMS(key, value, query, output);
    std::cout << "src/cpu/operators/CpuScaleDotProduction.cpp 1" << std::endl;
    /* Pretranspose Key, K=K^T*/
    _t_func  = std::make_unique<CpuTranspose>();
    _t_func->configure(key,&_buffer_t_info);
    std::cout << "src/cpu/operators/CpuScaleDotProduction.cpp 2" << std::endl;

    key = &_buffer_t_info;

    std::cout << "src/cpu/operators/CpuScaleDotProduction.cpp 3" << std::endl;

    /* Matrix multiply Query adn Key, QK */
    _mm_kernel = std::make_unique<cpu::kernels::CpuGemmMatrixMultiplyKernel>();
    _mm_kernel->configure(query,key,output,1.0,false);
    ARM_COMPUTE_UNUSED(value);

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

    std::cout << "src/cpu/operators/CpuScaleDotProduction.cpp" << std::endl;
    ARM_COMPUTE_ERROR_ON_MSG(tensors.empty(), "No inputs provided");
    prepare(tensors);
    std::cout << "1" << std::endl;

    //auto split_dimension = static_cast<kernels::CpuVectorizeKernel *>(_kernel.get())->get_split_dimension_hint();

    ARM_COMPUTE_UNUSED(tensors);


    //NEScheduler::get().schedule_op(_kernel.get(), split_dimension, _kernel->window(), tensors);
}

void CpuScaleDotProduction::prepare(ITensorPack &tensors)
{
    if(!_is_prepared)
    {
        const ITensor *key      = tensors.get_const_tensor(ACL_SRC_0);
        const ITensor *key_t    = key;

        CpuAuxTensorHandler pretransposed_key(
            offset_int_vec(KeyTransposeBuffer), _buffer_t_info, tensors,
            false /*pack_inject: no need to inject into tensors*/,
            _t_func ==
                nullptr /*bypass_alloc: no need to allocate if _t_kernel is not run*/);

        if (_t_func)
        {
            // Run pretranspose kernel
            ITensorPack pretranspose_pack{{ACL_SRC, key_t}, {ACL_DST, pretransposed_key.get()}};
            _t_func->run(pretranspose_pack);
            key_t = pretransposed_key.get();
        }
        
        _is_prepared = true;
    }
}

} // namespace cpu
} // namespace arm_compute
