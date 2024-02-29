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
     * @param[in]  d_model  Model dimension. Data types supported: TBA
     * @param[out] output   Output tensor, shape (seq_len,d_model). Data type supported: TBA
     */
    void configure(const unsigned int d_model, ITensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref NETokenEmbeddingLayer
     *
     * @param[in] input  Source tensor info. Data types supported: TBA
     * @param[in] alpha  Source alpha tensor info. Data types supported: same of @p input.
     * @param[in] output Destination tensor info. Data type supported: same as @p input
     *
     * @return a status
     */
    static Status validate(const unsigned int d_model, ITensor *output);

    // Inherited methods overridden:
    void run() override;
private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace arm_compute

#endif /* ARM_COMPUTE_NETOKENEMBEDDINGLAYER_H */