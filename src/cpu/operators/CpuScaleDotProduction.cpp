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

    _reshape_b_only_on_first_run = key->are_values_constant();

    /* Pretranspose Key, K=K^T*/
    const ITensorInfo *b_to_use = key;
    _pretranspose_b_func = std::make_unique<CpuTranspose>();
    _pretranspose_b_func->configure(b_to_use, &_pretransposed_b);

    experimental::MemoryLifetime lifetime = experimental::MemoryLifetime::Temporary;

    _aux_mem[PreTransposedRHS] =
                experimental::MemoryInfo(offset_int_vec(PreTransposedRHS), lifetime, _pretransposed_b.total_size());
            b_to_use = &_pretransposed_b;

    /* Matrix multiply Query adn Key, QK */
    _mm_kernel = std::make_unique<cpu::kernels::CpuGemmMatrixMultiplyKernel>();
    _mm_kernel->configure(query,b_to_use,output,1.0,false);
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

    ARM_COMPUTE_ERROR_ON_MSG(tensors.empty(), "No inputs provided"); 
    auto a = tensors.get_const_tensor(ACL_SRC_0);
    auto b = tensors.get_const_tensor(ACL_SRC_1);
    auto c = tensors.get_const_tensor(ACL_SRC_2);
    auto d = tensors.get_tensor(ACL_DST);

    CpuAuxTensorHandler pretransposed_b(offset_int_vec(PreTransposedRHS), _pretransposed_b, tensors);

    const ITensor *b_to_use = b;
    if (_pretranspose_b_func)
    {
        if (!_reshape_b_only_on_first_run)
        {
            // Run pretranspose kernel
            ITensorPack pretranspose_pack{{ACL_SRC, b_to_use}, {ACL_DST, pretransposed_b.get()}};
            _pretranspose_b_func->run(pretranspose_pack);
        }
        b_to_use = pretransposed_b.get();
    }

    ARM_COMPUTE_UNUSED(a);
    ARM_COMPUTE_UNUSED(c);
    ARM_COMPUTE_UNUSED(d);
}

experimental::MemoryRequirements CpuScaleDotProduction::workspace() const
{
    return _aux_mem;
}

} // namespace cpu
} // namespace arm_compute
