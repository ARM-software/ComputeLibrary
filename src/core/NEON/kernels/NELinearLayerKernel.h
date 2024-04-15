#ifndef ARM_COMPUTE_NELINEARLAYERKERNEL_H
#define ARM_COMPUTE_NELINEARLAYERKERNEL_H

#include "src/core/KernelTypes.h"
#include "src/core/NEON/INEKernel.h"

namespace arm_compute
{
namespace kernels
{
/** Interface for the kernel to perform linear operation for Value, Key, Query
 *
 * Supported logical operations:
 *  - Key
 *  - Value
 *  - Query
 */
class NELinearLayerKernel : public INEKernel
{
public:
    /** Initialise the kernel's inputs and output
     *
     * @param[in]  input1 An input tensor. Data type supported: f32.
     * @param[out] output Output tensor. Data type supported: F32.
     * @param[out] op     Logical operation to perform
     */
    void configure(const ITensorInfo *input1, ITensorInfo *output, LinearAttentionOperation op);
    /** Static function to check if given info will lead to a valid configuration of @ref NELinearLayerKernel
     *
     * @param[in] input1 An input tensor. Data type supported: F32.
     * @param[in] output Output tensor. Data type supported: F32..
     * @param[in] op     Logical operation to perform
     *
     * @return a status
     */
    static Status
    validate(const ITensorInfo *input1,  const ITensorInfo *output, LinearAttentionOperation op);

    // Inherited methods overridden:
    void        run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;
    const char *name() const override;

private:
    LinearAttentionOperation _op{};
};

} // namespace kernels
} // namespace arm_compute
#endif /* ARM_COMPUTE_NELINEARLAYERKERNEL_H */
