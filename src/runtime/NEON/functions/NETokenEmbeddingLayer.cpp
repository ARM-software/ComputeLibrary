#include "arm_compute/runtime/NEON/functions/NETokenEmbeddingLayer.h"

#include "arm_compute/core/Validate.h"

// operator to be added 

namespace arm_compute
{
    void NETokenEmbeddingLayer::configure(ITensor *input, ITensor *output, TokenEmbeddingLayerInfo tkemb_info)
    {
         std::cout << " NETokenEmbeddingLayer::configure!!!!!!!!!!!!!!!  " << std::endl;
    }

    void NETokenEmbeddingLayer::run()
    {
        std::cout << " NETokenEmbeddingLayer::run!!!!!!!!!!!!!!!  " << std::endl;
    }
} // namespace arm_compute