#ifndef ARM_COMPUTE_CPU_LAYER_NORM_H
#define ARM_COMPUTE_CPU_LAYER_NORM_H

#include "arm_compute/core/TensorInfo.h"

#include "src/cpu/ICpuOperator.h"
#include "src/cpu/kernels/CpuLayerNormKernel.h"

namespace arm_compute
{
namespace cpu
{

/** Basic function to run @ref kernels::CpuLayerNormKernel 
 * @note Performs LayerNorm function [alpha * A * B + beta * C]
*/
class CpuLayerNorm : public ICpuOperator
{
public:
    /** Initialise the kernel's inputs and output
     *
     * @param[in]  input      Input tensor. Data type supported: f32.
     * @param[out] output     Output tensor. Data type supported: F32.
     * @param[in]  info       (Optional)LayerNorm layer operation information
     */
    void configure(const ITensorInfo *input,
                   ITensorInfo       *output,
                   const LayerNormLayerInfo& info = LayerNormLayerInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref CpuLayerNormKernel
     *
     * Similar to @ref CpuGemm::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input,
                           ITensorInfo       *output,
                           const LayerNormLayerInfo& info = LayerNormLayerInfo());

    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;

private:
    std::unique_ptr<kernels::CpuLayerNormKernel> _layer_norm_kernel{nullptr};
};

} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_LAYER_NORM_H */
