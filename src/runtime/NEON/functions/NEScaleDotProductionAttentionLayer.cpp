#include "arm_compute/runtime/NEON/functions/NEScaleDotProductionAttentionLayer.h"

#include "arm_compute/core/Validate.h"
#include "src/core/helpers/MemoryHelpers.h"
#include "src/cpu/operators/CpuScaleDotProduction.h"
#include "src/cpu/operators/CpuGemm.h"

namespace arm_compute
{

struct NEScaleDotProductionAttentionLayer::Impl
{

    MemoryGroup                         memory_group{};

    ITensorPack                         scale_dot_pack{};

    IRuntimeContext                    *ctx{nullptr};

    std::unique_ptr<cpu::CpuScaleDotProduction> scale_dot_production_op{nullptr};

    bool is_prepared{false};
};

NEScaleDotProductionAttentionLayer::NEScaleDotProductionAttentionLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _impl(std::make_unique<Impl>())
{
    _impl->memory_group = MemoryGroup(std::move(memory_manager));
}

NEScaleDotProductionAttentionLayer::~NEScaleDotProductionAttentionLayer() = default;

void NEScaleDotProductionAttentionLayer::configure(const ITensor *query,
                                                   const ITensor *key,
                                                   const ITensor *value,
                                                   ITensor *output,
                                                   const ScaleDotProductionAttentionLayerInfo& info)
{
    /* Scale dot production of key and query */
    _impl->scale_dot_production_op  = std::make_unique<cpu::CpuScaleDotProduction>();
    _impl->scale_dot_production_op->configure(query->info(),key->info(),value->info(),output->info(),info);
    _impl->scale_dot_pack = {{ACL_SRC_0, query}, {ACL_SRC_1, key}, {ACL_SRC_2, value}, {ACL_DST, output}};

}

void NEScaleDotProductionAttentionLayer::run()
{
    ITensorPack pack;

    _impl->scale_dot_production_op->run(_impl->scale_dot_pack);

}

} // namespace arm_compute