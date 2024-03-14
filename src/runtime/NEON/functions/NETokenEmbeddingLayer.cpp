#include "arm_compute/runtime/NEON/functions/NETokenEmbeddingLayer.h"

#include "arm_compute/core/Validate.h"

#include "src/cpu/operators/CpuTokenEmbed.h"
// operator to be added 

namespace arm_compute
{

NETokenEmbeddingLayer::NETokenEmbeddingLayer()
{
}

NETokenEmbeddingLayer::~NETokenEmbeddingLayer() = default;

void NETokenEmbeddingLayer::configure(ITensor *input, ITensor *output, TokenEmbeddingLayerInfo tkemb_info)
{
        std::cout << " NETokenEmbeddingLayer::configure!!!!!!!!!!!!!!!  " << std::endl;
}

void NETokenEmbeddingLayer::run()
{
    std::cout << " NETokenEmbeddingLayer::run!!!!!!!!!!!!!!!  " << std::endl;
}

Status
NETokenEmbeddingLayer::validate(const ITensorInfo *input, const ITensorInfo *output, const TokenEmbeddingLayerInfo &tkemb_info)
{
    std::cout << "NETokenEmbeddingLayer::validate" << std::endl;
    std::cout << tkemb_info.d_model() << std::endl;
    return cpu::CpuTokenEmbed::validate(input, output);
}

} // namespace arm_compute