#include "arm_compute/runtime/NEON/functions/NEScaleDotProductionAttentionLayer.h"

#include "arm_compute/core/Validate.h"
#include "src/core/helpers/MemoryHelpers.h"
#include "src/cpu/operators/CpuScaleDotProduction.h"
#include "src/cpu/operators/CpuSoftmax.h"
#include "src/cpu/operators/CpuGemm.h"

namespace arm_compute
{

struct NEScaleDotProductionAttentionLayer::Impl
{

    MemoryGroup                         memory_group{};

    ITensorPack                         scale_dot_pack{};
    ITensorPack                         softmax_pack{};
    ITensorPack                         value_gemm_pack{};

    IRuntimeContext                    *ctx{nullptr};

    std::unique_ptr<cpu::CpuScaleDotProduction> scale_dot_production_op{nullptr};
    std::unique_ptr<cpu::CpuSoftmaxGeneric>     softmax_op{nullptr};
    std::unique_ptr<cpu::CpuGemm>               value_gemm_op{nullptr};

    /*
    WorkspaceData<Tensor>            workspace{};
    experimental::MemoryRequirements aux_mem_req{};
    */

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
    ITensor * production_to_softmax = output;
    ITensor * softmax_to_gemm = output;

    /* Scale dot production of key and query */
    _impl->scale_dot_production_op  = std::make_unique<cpu::CpuScaleDotProduction>();
    _impl->scale_dot_production_op->configure(query->info(),key->info(),value->info(),production_to_softmax->info(),info);
    _impl->scale_dot_pack = {{ACL_SRC_0, query}, {ACL_SRC_1, key}, {ACL_SRC_2, value}, {ACL_DST, production_to_softmax}};
    
    /*
    _impl->aux_mem_req = _impl->scale_dot_production_op->workspace();
    _impl->workspace =
        manage_workspace<Tensor>(_impl->aux_mem_req, _impl->memory_group, _impl->scale_dot_pack, _impl->scale_dot_pack);
    */

    /*  Softmax of previous product */
    _impl->softmax_op = std::make_unique<cpu::CpuSoftmaxGeneric>();
    _impl->softmax_op->configure(production_to_softmax->info(),softmax_to_gemm->info());
    _impl->softmax_pack = {{ACL_SRC, production_to_softmax}, {ACL_DST, softmax_to_gemm}};

    /* Scale dot production of key and query */
    float scale = sqrt(info.d_model());
    _impl->value_gemm_op = std::make_unique<cpu::CpuGemm>();
    _impl->value_gemm_op->configure(softmax_to_gemm->info(),value->info(),nullptr,output->info(),scale,1.0);
    _impl->value_gemm_pack = {{ACL_SRC_0, production_to_softmax}, {ACL_SRC_1, value}, {ACL_DST, softmax_to_gemm}};

}

void NEScaleDotProductionAttentionLayer::run()
{
    ITensorPack pack;

    _impl->scale_dot_production_op->run(_impl->scale_dot_pack);
    _impl->softmax_op->run(_impl->softmax_pack);

    _impl->value_gemm_op->run(_impl->value_gemm_pack);

}

} // namespace arm_compute