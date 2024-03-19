#include "arm_compute/runtime/NEON/functions/NETokenEmbeddingLayer.h"

#include "arm_compute/core/Validate.h"

#include "src/cpu/operators/CpuTokenEmbed.h"
// operator to be added 

namespace arm_compute
{

struct NETokenEmbeddingLayer::Impl
{
    const ITensor                      *src{nullptr};
    const ITensor                      *vocab{nullptr};
    ITensor                            *dst{nullptr};
    IRuntimeContext                    *ctx{nullptr};
    std::unique_ptr<cpu::CpuTokenEmbed> op{nullptr};
};

NETokenEmbeddingLayer::NETokenEmbeddingLayer(): _impl(std::make_unique<Impl>())
{
}

NETokenEmbeddingLayer::~NETokenEmbeddingLayer() = default;

void NETokenEmbeddingLayer::configure(ITensor *input, ITensor *vocab, ITensor *output, TokenEmbeddingLayerInfo tkemb_info)
{
    std::cout << "src/runtime/NEON/functions/NETokenEmbeddingLayer.cpp" << std::endl;
    _impl->src      = input;
    std::cout << _impl->src->info()->tensor_shape().x() << std::endl;
    _impl->vocab    = vocab;
    _impl->dst = output == nullptr ? input : output;
    _impl->op  = std::make_unique<cpu::CpuTokenEmbed>();
    _impl->op->configure(_impl->src->info(), _impl->dst->info(), tkemb_info);
}

void NETokenEmbeddingLayer::prepare()
{
}

void NETokenEmbeddingLayer::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC_0, _impl->src);
    pack.add_tensor(TensorType::ACL_SRC_1, _impl->vocab);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);
    _impl->op->run(pack);
}

} // namespace arm_compute