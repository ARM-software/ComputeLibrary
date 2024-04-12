#include "arm_compute/runtime/NEON/functions/NEScaleDotProductionAttentionLayer.h"

#include "arm_compute/core/Validate.h"
#include "src/core/helpers/MemoryHelpers.h"
#include "src/cpu/operators/CpuScaleDotProduction.h"

namespace arm_compute
{

struct NEScaleDotProductionAttentionLayer::Impl
{

    MemoryGroup                         memory_group{};
    ITensorPack                         run_pack{};
    IRuntimeContext                    *ctx{nullptr};

    std::unique_ptr<cpu::CpuScaleDotProduction> op{nullptr};
    WorkspaceData<Tensor>            workspace{};
    experimental::MemoryRequirements aux_mem_req{};

    bool is_prepared{false};
};

NEScaleDotProductionAttentionLayer::NEScaleDotProductionAttentionLayer(std::shared_ptr<IMemoryManager> memory_manager): _impl(std::make_unique<Impl>())
{
    _impl->memory_group = MemoryGroup(std::move(memory_manager));
}

NEScaleDotProductionAttentionLayer::~NEScaleDotProductionAttentionLayer() = default;

void NEScaleDotProductionAttentionLayer::configure(ITensor *key, ITensor *value, ITensor *query, ITensor *output)
{
    std::cout << " src/runtime/NEON/functions/NEScaleDotProductionAttentionLayer.cpp 1 " <<std::endl;
    _impl->op  = std::make_unique<cpu::CpuScaleDotProduction>();

    std::cout << " src/runtime/NEON/functions/NEScaleDotProductionAttentionLayer.cpp 2 " <<std::endl;
    _impl->run_pack = {{ACL_SRC_0, key}, {ACL_SRC_1, value}, {ACL_SRC_2, query}, {ACL_DST, output}};

    std::cout << " src/runtime/NEON/functions/NEScaleDotProductionAttentionLayer.cpp 3 " <<std::endl;

    _impl->op->configure(key->info(),value->info(),query->info(),output->info());
    _impl->aux_mem_req = _impl->op->workspace();
    _impl->workspace =
        manage_workspace<Tensor>(_impl->aux_mem_req, _impl->memory_group, _impl->run_pack, _impl->run_pack);

    std::cout << " src/runtime/NEON/functions/NEScaleDotProductionAttentionLayer.cpp 4 " <<std::endl;

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
    std::cout << "src/runtime/NEON/functions/NEScaleDotProductionAttentionLayer.cpp RUNNNNNNNNN!!!!!!!!" << std::endl;
    prepare();

    std::cout << "src/runtime/NEON/functions/NEScaleDotProductionAttentionLayer.cpp RUNNNNNNNNN!!!!!!!!" << std::endl;
    _impl->op->run(_impl->run_pack);
}

} // namespace arm_compute