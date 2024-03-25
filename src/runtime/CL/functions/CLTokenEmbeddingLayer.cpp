#include "arm_compute/runtime/CL/functions/CLTokenEmbeddingLayer.h"

#include "arm_compute/core/Validate.h"

// operator to be added 

namespace arm_compute
{

CLTokenEmbeddingLayer::CLTokenEmbeddingLayer()
{
}
CLTokenEmbeddingLayer::~CLTokenEmbeddingLayer() = default;

void CLTokenEmbeddingLayer::configure(ITensor *input, ITensor *output, TokenEmbeddingLayerInfo tkemb_info)
{
        std::cout << " CLTokenEmbeddingLayer::configure!!!!!!!!!!!!!!!  " << std::endl;
}

void CLTokenEmbeddingLayer::run()
{
    std::cout << " CLTokenEmbeddingLayer::run!!!!!!!!!!!!!!!  " << std::endl;
}

} // namespace arm_compute