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
class Status;

/** Basic function to execute concatenate tensors along a given axis */
class NETokenEmbeddingLayer : public IFunction
{
public:
    /** Default constructor */
    NETokenEmbeddingLayer();
    /** Destructor */
    ~NETokenEmbeddingLayer();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NETokenEmbeddingLayer(const NETokenEmbeddingLayer &) = delete;
    /** Default move constructor */
    NETokenEmbeddingLayer(NETokenEmbeddingLayer &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NETokenEmbeddingLayer &operator=(const NETokenEmbeddingLayer &) = delete;
    /** Default move assignment operator */
    NETokenEmbeddingLayer &operator=(NETokenEmbeddingLayer &&);
    /** Initialise the kernel's inputs vector and output.
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src            |dst            |
     * |:--------------|:--------------|
     * |QASYMM8        |QASYMM8        |
     * |QASYMM8_SIGNED |QASYMM8_SIGNED |
     * |F16            |F16            |
     * |F32            |F32            |
     *
     * @note Input and output tensor dimensions preconditions defer depending on the concatenation axis.
     * @note Preconditions can be found respectively at @ref cpu::kernels::CpuConcatenateWidthKernel, @ref cpu::kernels::CpuConcatenateHeightKernel,
     *       @ref cpu::kernels::CpuConcatenateDepthKernel and @ref cpu::kernels::CpuConcatenateBatchKernel.
     *
     * @param[in,out] inputs_vector The vectors containing all the tensors to concatenate. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[out]    output        Output tensor. Data types supported: Same as @p input.
     * @param[in]     axis          Concatenation axis. Supported underlying concatenation axis are 0, 1, 2 and 3.
     */
    void configure(std::vector<const ITensor *> inputs_vector, ITensor *output, size_t axis);
    /** Static function to check if given info will lead to a valid configuration of @ref NEConcatenateLayer
     *
     * @note Input and output tensor dimensions preconditions defer depending on the concatenation axis.
     * @note Preconditions can be found respectively at @ref cpu::kernels::CpuConcatenateWidthKernel, @ref cpu::kernels::CpuConcatenateHeightKernel,
     *       @ref cpu::kernels::CpuConcatenateDepthKernel and @ref cpu::kernels::CpuConcatenateBatchKernel.
     *
     * @param[in] inputs_vector The vectors containing all the tensors info to concatenate. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in] output        Output tensor info. Data types supported: Same as @p input.
     * @param[in] axis          Concatenation axis. Supported underlying concatenation axis are 0, 1, 2 and 3.
     *
     * @return a status
     */
    static Status
    validate(const std::vector<const ITensorInfo *> &inputs_vector, const ITensorInfo *output, size_t axis);

    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_NETOKENEMBEDDINGLAYER_H */
