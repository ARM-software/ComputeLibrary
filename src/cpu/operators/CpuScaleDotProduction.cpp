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
CpuScaleDotProduction::CpuScaleDotProduction():_buffer_t_info()
{
}

CpuScaleDotProduction::~CpuScaleDotProduction() = default;

void CpuScaleDotProduction::configure(const ITensorInfo *key, const ITensorInfo *value, const ITensorInfo *query, ITensorInfo *output)
{
    ARM_COMPUTE_LOG_PARAMS(key, value, query, output);
    std::cout << "src/cpu/operators/CpuScaleDotProduction.cpp 1" << std::endl;
    /* Pretranspose Key, K=K^T*/
    const ITensorInfo *key_to_use = key;
    _t_func  = std::make_unique<CpuTranspose>();
    _t_func->configure(key_to_use,&_buffer_t_info);
    
    
    experimental::MemoryLifetime lifetime = experimental::MemoryLifetime::Temporary;
    _aux_mem[KeyTransposeBuffer] =
        experimental::MemoryInfo(offset_int_vec(KeyTransposeBuffer), lifetime, _buffer_t_info.total_size());

    key_to_use = &_buffer_t_info;

    std::cout << "src/cpu/operators/CpuScaleDotProduction.cpp 2" << std::endl;


    std::cout << "src/cpu/operators/CpuScaleDotProduction.cpp 3" << std::endl;

    /* Matrix multiply Query adn Key, QK */
    //_mm_kernel = std::make_unique<cpu::kernels::CpuGemmMatrixMultiplyKernel>();
    //_mm_kernel->configure(query,key_to_use,output,1.0,false);
    ARM_COMPUTE_UNUSED(value);
    ARM_COMPUTE_UNUSED(query);
    ARM_COMPUTE_UNUSED(output);
    ARM_COMPUTE_UNUSED(key_to_use);

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
    ARM_COMPUTE_UNUSED(tensors);
}

experimental::MemoryRequirements CpuScaleDotProduction::workspace() const
{
    return _aux_mem;
}

} // namespace cpu
} // namespace arm_compute
