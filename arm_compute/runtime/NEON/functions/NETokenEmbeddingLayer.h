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
    NETokenEmbeddingLayer(IRuntimeContext *ctx = nullptr);
    /** Default Destructor */
    ~NETokenEmbeddingLayer();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NETokenEmbeddingLayer(const NETokenEmbeddingLayer &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NETokenEmbeddingLayer &operator=(const NETokenEmbeddingLayer &) = delete;


    /** TODO: Move able ?*/

    
    /** Set the input and output tensor.
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src            |dst            |
     * |:--------------|:--------------|
     * |TBA            |TBA            |
     * |TBA            |TBA            |
     *
     * @param[in, out] input           Source tensor. In case of @p output tensor = nullptr, this tensor will store the result
     *                                 of the activation function. Data types supported: QASYMM8/QASYMM8_SIGNED/QSYMM16/F16/F32.
     * @param[out]     output          Destination tensor. Data type supported: same as @p input
     * @param[in]      activation_info Token embedding layer parameters.
     */
    void configure(ITensor *input, ITensor *output, TokenEmbeddingLayerInfo tkemb_info);
    /** Static function to check if given info will lead to a valid configuration of @ref NETokenEmbeddingLayer
     *
     * @param[in] input  Source tensor info. Data types supported: TBA
     * @param[in] alpha  Source alpha tensor info. Data types supported: same of @p input.
     * @param[in] output Destination tensor info. Data type supported: same as @p input
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const TokenEmbeddingLayerInfo &tkemb_info);

    // Inherited methods overridden:
    void run() override;
private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace arm_compute

#endif /* ARM_COMPUTE_NETOKENEMBEDDINGLAYER_H */