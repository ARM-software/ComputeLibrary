#ifndef ARM_COMPUTE_NEEMBEDDINGSUMLAYER_H
#define ARM_COMPUTE_NEEMBEDDINGSUMLAYER_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IRuntimeContext.h"


#include <memory>

namespace arm_compute
{
// Forward declarations
class ITensor;
class ITensorInfo;

class NEEmbeddingSumLayer : public IFunction
{
public:
    /** Default Constructor */
    NEEmbeddingSumLayer();
    /** Default Destructor */
    ~NEEmbeddingSumLayer();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEEmbeddingSumLayer(const NEEmbeddingSumLayer &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEEmbeddingSumLayer &operator=(const NEEmbeddingSumLayer &) = delete;

    /** Set the input and output tensor.
     * 
     * @param[in]  token        Token embedding input, Data type supported: F32
     * @param[in]  segemnt      Token embedding input, Data type supported: F32
     * @param[in]  position     Token embedding input, Data type supported: F32
     * @param[out] output       Output tensor, shape (seq_len,d_model). Data type supported: F32
     */
    void configure(ITensor *token, ITensor *segemnt, ITensor *position, ITensor *output, const EmbeddingLayerInfo& emb_info);
    /** Static function to check if given info will lead to a valid configuration of @ref NEEmbeddingSumLayer
     * 
     * @param[in]  token        Token embedding input, Data type supported: F32
     * @param[in]  segemnt      Token embedding input, Data type supported: F32
     * @param[in]  position     Token embedding input, Data type supported: F32
     * @param[out] output       Output tensor, shape (seq_len,d_model). Data type supported: F32
     *
     * @return a status
     */
    static Status validate(ITensor *token, ITensor *segemnt, ITensor *position, ITensor *output, const EmbeddingLayerInfo& emb_info);

    void prepare() override;
    // Inherited methods overridden:
    void run() override;
private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace arm_compute

#endif /* ARM_COMPUTE_NEEMBEDDINGSUMLAYER_H */