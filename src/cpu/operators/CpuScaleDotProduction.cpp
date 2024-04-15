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
                                      ITensorInfo *output)
{
    ARM_COMPUTE_LOG_PARAMS(key, value, query, output);

    // Pick b tensor in case pretranspose should be performed
    const ITensorInfo *key_to_use = key;

    /* Pretranspose Key, K=K^T*/
    _pretranspose_key_func = std::make_unique<CpuTranspose>();
    _pretranspose_key_func->configure(key_to_use, &_pretransposed_key);

    _aux_mem[PreTransposedRHS] =
                experimental::MemoryInfo(offset_int_vec(PreTransposedRHS), experimental::MemoryLifetime::Persistent, _pretransposed_key.total_size());
    key_to_use = &_pretransposed_key;

    /* Matrix multiply Query adn Key, QK */
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
    CpuAuxTensorHandler pretransposed_key(
                offset_int_vec(PreTransposedRHS), _pretransposed_key, tensors,
                false /*pack_inject: no need to inject into tensors*/,
                _pretranspose_key_func ==
                    nullptr /*bypass_alloc: no need to allocate if _pretranspose_b_func is not run*/);
    if (_pretranspose_key_func)
    {
        // Run pretranspose kernel
        ITensorPack pretranspose_pack{{ACL_SRC, key_to_use}, {ACL_DST, pretransposed_key.get()}};
        _pretranspose_key_func->run(pretranspose_pack);
        key_to_use = pretransposed_key.get();
    }
    
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
