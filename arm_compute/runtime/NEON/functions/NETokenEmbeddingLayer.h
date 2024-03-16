#ifndef ARM_COMPUTE_NETOKENEMBEDDINGLAYER_H
#define ARM_COMPUTE_NETOKENEMBEDDINGLAYER_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IRuntimeContext.h"


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
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NETokenEmbeddingLayer(const NETokenEmbeddingLayer &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NETokenEmbeddingLayer &operator=(const NETokenEmbeddingLayer &) = delete;

    /** Set the input and output tensor.
     * 
     * @param[in]  input        Input tensor of char text, Data type supported: U8
     * @param[in]  vocab        Const tenser of char 2 vec, Data type supported: F32
     * @param[in]  tkemb_info   Token Embedding Layer Info.
     * @param[out] output       Output tensor, shape (seq_len,d_model). Data type supported: F32
     */
    void configure(ITensor *input, ITensor *vocab, ITensor *output, TokenEmbeddingLayerInfo tkemb_info);
    /** Static function to check if given info will lead to a valid configuration of @ref NETokenEmbeddingLayer
     *
     * @param[in] output Destination tensor info. Data type supported: same as @p input
     * @param[in] tkemb_info Token Embedding Layer Info.
     *
     * @return a status
     */
    static Status validate(ITensor *output, TokenEmbeddingLayerInfo &activation_info);

    void prepare() override;
    // Inherited methods overridden:
    void run() override;
private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace arm_compute

#endif /* ARM_COMPUTE_NETOKENEMBEDDINGLAYER_H */