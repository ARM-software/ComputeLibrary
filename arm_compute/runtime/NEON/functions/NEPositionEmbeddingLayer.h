#ifndef ARM_COMPUTE_NEPOSITIONEMBEDDINGLAYER_H
#define ARM_COMPUTE_NEPOSITIONEMBEDDINGLAYER_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IRuntimeContext.h"

#include <memory>

namespace arm_compute
{
// Forward declarations
class ITensor;
class ITensorInfo;

class NEPositionEmbeddingLayer : public IFunction
{
public:
    /** Default Constructor */
    NEPositionEmbeddingLayer();
    /** Default Destructor */
    ~NEPositionEmbeddingLayer();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEPositionEmbeddingLayer(const NEPositionEmbeddingLayer &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEPositionEmbeddingLayer &operator=(const NEPositionEmbeddingLayer &) = delete;

    /** Set the input and output tensor.
     * 
     * @param[in]  input        Input tensor of char text, Data type supported: U8
     * @param[in]  Position     Const tenser of Position vector, Data type supported: F32
     * @param[out] output       Output tensor, shape (seq_len,d_model). Data type supported: F32
     */
    void configure(ITensor *input, ITensor *Position, ITensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref NEPositionEmbeddingLayer
     *
     * @param[in] output Destination tensor info. Data type supported: same as @p input
     *
     * @return a status
     */
    static Status validate(ITensor *output);

    void prepare() override;
    // Inherited methods overridden:
    void run() override;
private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace arm_compute

#endif /* ARM_COMPUTE_NEPOSITIONEMBEDDINGLAYER_H */