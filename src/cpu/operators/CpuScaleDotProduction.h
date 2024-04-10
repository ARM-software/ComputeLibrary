#ifndef ARM_COMPUTE_CPU_SCALE_DOT_PRODUCTION_H
#define ARM_COMPUTE_CPU_SCALE_DOT_PRODUCTION_H


#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"

#include "src/cpu/ICpuOperator.h"


namespace arm_compute
{
namespace cpu
{
/** Function implementation for scale dot production, uses kernels:
 * @ref kernels::CpuEmbedKernel
*/
class CpuScaleDotProduction : public ICpuOperator
{
public:
    /** Configure operator for a given list of arguments
     *
     * @param[in]  key             Attention key tensor info. Data types supported: U8.
     * @param[in]  value           Attention value tensor info. Data types supported: U8.
     * @param[in]  query           Attention key tensor info. Data types supported: U8.
     * @param[out] output          Destination tensor info. Data type supported: F32
     */
    void configure(const ITensorInfo *key, const ITensorInfo *value, const ITensorInfo *query, ITensorInfo *output);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref CpuScaleDotProduction::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *key, const ITensorInfo *value, const ITensorInfo *query, ITensorInfo *output);

    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;
};
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_SCALE_DOT_PRODUCTION_H */
