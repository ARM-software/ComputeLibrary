#include "arm_compute/runtime/NEON/functions/NEEmbeddingSumLayer.h"

#include "arm_compute/core/Validate.h"

#include "src/cpu/operators/CpuTokenEmbed.h"
// operator to be added 

namespace arm_compute
{

struct NEEmbeddingSumLayer::Impl
{
    const ITensor                      *token{nullptr};
    const ITensor                      *segment{nullptr};
    const ITensor                      *position{nullptr};
    ITensor                            *dst{nullptr};
    IRuntimeContext                    *ctx{nullptr};
    std::unique_ptr<cpu::CpuTokenEmbed> op{nullptr};
};

NEEmbeddingSumLayer::NEEmbeddingSumLayer(): _impl(std::make_unique<Impl>())
{
}

NEEmbeddingSumLayer::~NEEmbeddingSumLayer() = default;

void NEEmbeddingSumLayer::configure(ITensor *token, ITensor *segment, ITensor *position, ITensor *output, const EmbeddingLayerInfo& emb_info)
{
    _impl->token      = token;
    _impl->segment    = segment;
    _impl->position   = segment;
    _impl->dst        = output;

    _impl->op  = std::make_unique<cpu::CpuTokenEmbed>();
    _impl->op->configure(_impl->token->info(),_impl->segment->info(), _impl->dst->info(), emb_info);
}

void NEEmbeddingSumLayer::prepare()
{
}

void NEEmbeddingSumLayer::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC_0, _impl->token);
    pack.add_tensor(TensorType::ACL_SRC_1, _impl->segment);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);

    _impl->op->run(pack);
}

} // namespace arm_compute