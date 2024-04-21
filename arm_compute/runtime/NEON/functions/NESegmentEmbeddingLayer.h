#ifndef ARM_COMPUTE_NESEGMENTEMBEDDINGLAYER_H
#define ARM_COMPUTE_NESEGMENTEMBEDDINGLAYER_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IRuntimeContext.h"

#include <memory>

namespace arm_compute
{
// Forward declarations
class ITensor;
class ITensorInfo;

class NESegmentEmbeddingLayer : public IFunction
{
public:
    /** Default Constructor */
    NESegmentEmbeddingLayer();
    /** Default Destructor */
    ~NESegmentEmbeddingLayer();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NESegmentEmbeddingLayer(const NESegmentEmbeddingLayer &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NESegmentEmbeddingLayer &operator=(const NESegmentEmbeddingLayer &) = delete;

    /** Set the input and output tensor.
     * 
     * @param[in]  input        Input tensor of char text, Data type supported: U8
     * @param[in]  segment      Const tenser of segment vector, Data type supported: F32
     * @param[out] output       Output tensor, shape (seq_len,d_model). Data type supported: F32
     */
    void configure(ITensor *input, ITensor *segment, ITensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref NESegmentEmbeddingLayer
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

#endif /* ARM_COMPUTE_NESEGMENTEMBEDDINGLAYER_H */