#include "arm_compute/runtime/NEON/functions/NETokenEmbeddingLayer.h"

#include "arm_compute/core/Validate.h"

#include "src/cpu/operators/CpuTokenEmbed.h"
// operator to be added 

namespace arm_compute
{

struct NETokenEmbeddingLayer::Impl
{
    const ITensor                      *src{nullptr};
    ITensor                            *dst{nullptr};
    IRuntimeContext                    *ctx{nullptr};
    std::unique_ptr<cpu::CpuTokenEmbed> op{nullptr};
};

NETokenEmbeddingLayer::NETokenEmbeddingLayer(): _impl(std::make_unique<Impl>())
{
}

NETokenEmbeddingLayer::~NETokenEmbeddingLayer() = default;

void NETokenEmbeddingLayer::configure(ITensor *input, ITensor *output, TokenEmbeddingLayerInfo tkemb_info)
{
    std::cout << " NETokenEmbeddingLayer::configure!!!!!!!!!!!!!!!  " << std::endl;
    std::cout << input->info()->tensor_shape().total_size() << std::endl;
    std::cout << output->info()->tensor_shape().total_size() << std::endl;
}


void NETokenEmbeddingLayer::run()
{
    std::cout << " NETokenEmbeddingLayer::run!!!!!!!!!!!!!!!  " << std::endl;
}

} // namespace arm_compute