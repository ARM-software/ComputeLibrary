#ifndef ARM_COMPUTE_CPU_LINEAR_H
#define ARM_COMPUTE_CPU_LINEAR_H

#include "src/cpu/ICpuOperator.h"

namespace arm_compute
{
namespace cpu
{

/** Basic function to run @ref kernels::CpuLinearKernel */
class CpuLinear : public ICpuOperator
{
public:
    /** Initialise the kernel's inputs and output
     *
     * @param[in]  input1 An input tensor. Data type supported: f32.
     * @param[out] output Output tensor. Data type supported: F32.
     * @param[out] op     Logical operation to perform
     */
    void configure(const ITensorInfo *input1, ITensorInfo *output, const LinearLayerInfo& info);
    /** Static function to check if given info will lead to a valid configuration of @ref CpuLinearKernel
     *
     * @param[in] input1 An input tensor. Data type supported: F32.
     * @param[in] output Output tensor. Data type supported: F32..
     * @param[in] op     Logical operation to perform
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1,  const ITensorInfo *output,const LinearLayerInfo& info);

    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;
};

} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_ACTIVATION_H */
