#include "arm_compute/runtime/NEON/functions/NEScaleDotProductionAttentionLayer.h"

#include "arm_compute/core/Validate.h"

#include "src/cpu/operators/CpuScaleDotProduction.h"

namespace arm_compute
{

struct NEScaleDotProductionAttentionLayer::Impl
{
    const ITensor                      *key{nullptr};
    const ITensor                      *value{nullptr};
    const ITensor                      *query{nullptr};
    ITensor                            *dst{nullptr};
    IRuntimeContext                    *ctx{nullptr};
    std::unique_ptr<cpu::CpuScaleDotProduction> op{nullptr};
};

NEScaleDotProductionAttentionLayer::NEScaleDotProductionAttentionLayer(): _impl(std::make_unique<Impl>())
{
}

NEScaleDotProductionAttentionLayer::~NEScaleDotProductionAttentionLayer() = default;

void NEScaleDotProductionAttentionLayer::configure(ITensor *key, ITensor *value, ITensor *query, ITensor *output)
{
    _impl->key      = key;
    _impl->value    = value;
    _impl->query    = query;
    _impl->dst      = output;

    _impl->op  = std::make_unique<cpu::CpuScaleDotProduction>();
    _impl->op->configure(_impl->key->info(),_impl->value->info(),_impl->query->info(),_impl->dst->info());

}

void NEScaleDotProductionAttentionLayer::prepare()
{
}

void NEScaleDotProductionAttentionLayer::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC_0, _impl->key);
    pack.add_tensor(TensorType::ACL_SRC_1, _impl->value);
    pack.add_tensor(TensorType::ACL_SRC_2, _impl->query);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);

    std::cout << "src/runtime/NEON/functions/NEScaleDotProductionAttentionLayer.cpp RUNNNNNNNNN!!!!!!!!" << std::endl;
    _impl->op->run(pack);
}

} // namespace arm_compute