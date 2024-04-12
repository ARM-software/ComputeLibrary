#include "arm_compute/runtime/NEON/functions/NEScaleDotProductionAttentionLayer.h"

#include "arm_compute/core/Validate.h"

#include "src/cpu/operators/CpuScaleDotProduction.h"

namespace arm_compute
{

struct NEScaleDotProductionAttentionLayer::Impl
{

    ITensorPack                         run_pack{};
    IRuntimeContext                    *ctx{nullptr};

    std::unique_ptr<cpu::CpuScaleDotProduction> op{nullptr};

    bool is_prepared{false};
};

NEScaleDotProductionAttentionLayer::NEScaleDotProductionAttentionLayer(): _impl(std::make_unique<Impl>())
{
}

NEScaleDotProductionAttentionLayer::~NEScaleDotProductionAttentionLayer() = default;

void NEScaleDotProductionAttentionLayer::configure(ITensor *key, ITensor *value, ITensor *query, ITensor *output)
{
    _impl->op  = std::make_unique<cpu::CpuScaleDotProduction>();

    _impl->run_pack = {{ACL_SRC_0, key}, {ACL_SRC_1, value}, {ACL_SRC_2, query}, {ACL_DST, output}};

    _impl->op->configure(key->info(),value->info(),query->info(),output->info());

}

void NEScaleDotProductionAttentionLayer::prepare()
{
    if(!_impl->is_prepared)
    {
        _impl->op->prepare(_impl->run_pack);
    }
}

void NEScaleDotProductionAttentionLayer::run()
{
    ITensorPack pack;

    prepare();

    std::cout << "src/runtime/NEON/functions/NEScaleDotProductionAttentionLayer.cpp RUNNNNNNNNN!!!!!!!!" << std::endl;
    _impl->op->run(_impl->run_pack);
}

} // namespace arm_compute