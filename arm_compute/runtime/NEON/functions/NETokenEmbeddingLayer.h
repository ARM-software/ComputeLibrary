#ifndef ARM_COMPUTE_NETOKENEMBEDDINGLAYER_H
#define ARM_COMPUTE_NETOKENEMBEDDINGLAYER_H


#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/NEON/INEOperator.h"

namespace arm_compute
{
// Forward Declarations
class ITensor;

/** Basic function to run @ref NETokenEmbeddingLayerKernel */
class NETokenEmbeddingLayer : public IFunction
{
public:
    /** Default Constructor */
    NETokenEmbeddingLayer();
    /** Default Destructor */
    ~NETokenEmbeddingLayer();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NETokenEmbeddingLayer(const NETokenEmbeddingLayer &) = delete;
    /** Default move constructor */
    NETokenEmbeddingLayer(NETokenEmbeddingLayer &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NETokenEmbeddingLayer &operator=(const NETokenEmbeddingLayer &) = delete;
    /** Default move assignment operator */
    NETokenEmbeddingLayer &operator=(NETokenEmbeddingLayer &&);

    
    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace arm_compute

#endif /* ARM_COMPUTE_NETOKENEMBEDDINGLAYER_H */