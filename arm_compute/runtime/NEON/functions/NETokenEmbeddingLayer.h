#ifndef ARM_COMPUTE_NETOKENEMBEDDINGLAYER_H
#define ARM_COMPUTE_NETOKENEMBEDDINGLAYER_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"

#include <memory>

namespace arm_compute
{
// Forward declarations
class ITensor;
class ITensorInfo;

class NETokenEmbeddingLayer : public IFunction
{
public:
    /** Default Constructor */
    NETokenEmbeddingLayer();
    /** Default Destructor */
    ~NETokenEmbeddingLayer();

    // Inherited methods overridden:
    void run() override;
private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace arm_compute

#endif /* ARM_COMPUTE_NETOKENEMBEDDINGLAYER_H */