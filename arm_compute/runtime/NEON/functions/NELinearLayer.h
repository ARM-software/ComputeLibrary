#ifndef ARM_COMPUTE_LINEAR_LAYER_H
#define ARM_COMPUTE_LINEAR_LAYER_H

#include "arm_compute/core/Types.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/runtime/IFunction.h"

#include <memory>

namespace arm_compute
{
// Forward declarations
class ITensor;
class ITensorInfo;

/** Perform basic linear function */
class NELinearLayer : public IFunction
{
public:
    /** Constructor */
    NELinearLayer();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NELinearLayer(const NELinearLayer &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NELinearLayer(NELinearLayer &&) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NELinearLayer &operator=(const NELinearLayer &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NELinearLayer &operator=(NELinearLayer &&) = delete;
    /** Destructor */
    ~NELinearLayer();

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
    void configure(const ITensor *input1, const ITensor *weight, const ITensor *bias, ITensor *output, const LinearLayerInfo& linear_info);
    /** Static function to check if given info will lead to a valid configuration of @ref NELinearLayer
     *
     * @param[in] input1 First input tensor info. Data types supported: F32.
     * @param[in] output Output tensor info. Data type supported: F32.
     *
     * @return a status
     */
    static Status validate(const ITensor *input, const ITensor *weight, const ITensor *bias, ITensor *output, const LinearLayerInfo& linear_info);

    // Inherited methods overridden
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace arm_compute

#endif /* ARM_COMPUTE_LINEAR_LAYER_H */