#include "arm_compute/runtime/NEON/functions/NETokenEmbeddingLayer.h"

#include "arm_compute/core/Validate.h"

// operator to be added 

namespace arm_compute
{
    
NETokenEmbeddingLayer::NETokenEmbeddingLayer()
{
}

NETokenEmbeddingLayer::~NETokenEmbeddingLayer() = default;

void NETokenEmbeddingLayer::run()
{
    std::cout << " NETokenEmbeddingLayer::run!!!!!!!!!!!!!!!  " << std::endl;
}

} // namespace arm_compute