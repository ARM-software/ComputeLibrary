#include "arm_compute/runtime/NEON/functions/NEScaleDotProductionAttentionLayer.h"

#include "arm_compute/core/Validate.h"
#include "src/core/helpers/MemoryHelpers.h"
#include "src/cpu/operators/CpuScaleDotProduction.h"
#include "src/cpu/operators/CpuSoftmax.h"

namespace arm_compute
{

struct NEScaleDotProductionAttentionLayer::Impl
{

    MemoryGroup                         memory_group{};
    ITensorPack                         scale_dot_pack{};
    ITensorPack                         softmax_pack{};
    IRuntimeContext                    *ctx{nullptr};

    std::unique_ptr<cpu::CpuScaleDotProduction> scale_dot_production_op{nullptr};
    std::unique_ptr<cpu::CpuSoftmaxGeneric>     softmax_op{nullptr};

    WorkspaceData<Tensor>            workspace{};
    experimental::MemoryRequirements aux_mem_req{};

    bool is_prepared{false};
};

NEScaleDotProductionAttentionLayer::NEScaleDotProductionAttentionLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _impl(std::make_unique<Impl>())
{
    _impl->memory_group = MemoryGroup(std::move(memory_manager));
}

NEScaleDotProductionAttentionLayer::~NEScaleDotProductionAttentionLayer() = default;

void NEScaleDotProductionAttentionLayer::configure(const ITensor *key,
                                                   const ITensor *value,
                                                   const ITensor *query,
                                                   ITensor *output,
                                                   const ScaleDotProductionAttentionLayerInfo& info)
{
    ITensor * production_to_softmax = output;
    _impl->scale_dot_production_op  = std::make_unique<cpu::CpuScaleDotProduction>();
    _impl->scale_dot_production_op->configure(key->info(),value->info(),query->info(),production_to_softmax->info(),info);
    _impl->aux_mem_req = _impl->scale_dot_production_op->workspace();
    _impl->scale_dot_pack = {{ACL_SRC_0, key}, {ACL_SRC_1, value}, {ACL_SRC_2, query}, {ACL_DST, production_to_softmax}};
    _impl->workspace =
        manage_workspace<Tensor>(_impl->aux_mem_req, _impl->memory_group, _impl->scale_dot_pack, _impl->scale_dot_pack);
    
    _impl->softmax_op = std::make_unique<cpu::CpuSoftmaxGeneric>();
    _impl->softmax_op->configure(production_to_softmax->info(),output->info());
    _impl->softmax_pack = {{ACL_SRC, production_to_softmax}, {ACL_DST, output}};
}

void NEScaleDotProductionAttentionLayer::run()
{
    ITensorPack pack;
    std::cout << "src/runtime/NEON/functions/NEScaleDotProductionAttentionLayer.cpp RUNNNNNNNNN!!!!!!!!" << std::endl;

    _impl->scale_dot_production_op->run(_impl->scale_dot_pack);
    _impl->softmax_op->run(_impl->softmax_pack);

    std::cout << "src/runtime/NEON/functions/NEScaleDotProductionAttentionLayer.cpp RUNNNNNNNNN!!!!!!!!" << std::endl;
}

} // namespace arm_compute