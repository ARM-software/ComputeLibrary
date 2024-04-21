#ifndef ARM_COMPUTE_LAYER_NORM_LAYER_H
#define ARM_COMPUTE_LAYER_NORM_LAYER_H

#include "arm_compute/core/Types.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/runtime/IFunction.h"

#include <memory>

namespace arm_compute
{
// Forward declarations
class ITensor;
class ITensorInfo;

/** Perform basic layer normalization function */
class NELayerNormLayer : public IFunction
{
public:
    /** Constructor */
    NELayerNormLayer();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NELayerNormLayer(const NELayerNormLayer &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NELayerNormLayer(NELayerNormLayer &&) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NELayerNormLayer &operator=(const NELayerNormLayer &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NELayerNormLayer &operator=(NELayerNormLayer &&) = delete;
    /** Destructor */
    ~NELayerNormLayer();

    /** Initialise the kernel's inputs and output
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src0           |dst          |
     * |:--------------|:------------|
     * |F32            |F32          |
     *
     * @param[in]  input1 First tensor input. Data type supported: F32.
     * @param[out] output Output tensor. Data type supported: F32.
     */
    void configure(const ITensor *input, ITensor *output, const LayerNormLayerInfo& LayerNorm_info);
    /** Static function to check if given info will lead to a valid configuration of @ref NELayerNormLayer
     *
     * @param[in] input     First input tensor info. Data types supported: F32.
     * @param[in] output Output tensor info. Data type supported: F32.
     *
     * @return a status
     */
    static Status validate(const ITensor *input, ITensor *output, const LayerNormLayerInfo& LayerNorm_info);

    // Inherited methods overridden
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
    
};

} // namespace arm_compute

#endif /* ARM_COMPUTE_LAYER_NORM_LAYER_H */