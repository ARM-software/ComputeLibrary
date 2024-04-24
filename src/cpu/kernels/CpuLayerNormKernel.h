#ifndef ARM_COMPUTE_NELAYERNORMKERNEL_H
#define ARM_COMPUTE_NELAYERNORMKERNEL_H

#include "src/core/common/Macros.h"

#include "src/core/KernelTypes.h"
#include "src/cpu/ICpuKernel.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
/** Interface for the kernel to perform layer normalization */
class CpuLayerNormKernel : public ICpuKernel<CpuLayerNormKernel>
{
private:
    using LayerNormKernelPtr =
        std::add_pointer<void(const ITensor *, ITensor *, const LayerNormLayerInfo &, const Window &)>::type;
public:
    /* Default Constructor */
    CpuLayerNormKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuLayerNormKernel);

    /** Initialise the kernel's inputs and output
     *
     * @param[in]  input1 An input tensor. Data type supported: f32.
     * @param[out] output Output tensor. Data type supported: F32.
     * @param[out] op     Logical operation to perform
     */
    void configure(const ITensorInfo *input, ITensorInfo *output, LayerNormLayerInfo info);
    /** Static function to check if given info will lead to a valid configuration of @ref CpuLayerNormKernel
     *
     * @param[in] input An input tensor. Data type supported: F32.
     * @param[in] output Output tensor. Data type supported: F32..
     * @param[in] op     Logical operation to perform
     *
     * @return a status
     */
    static Status
    validate(const ITensorInfo *input,  const ITensorInfo *output, LayerNormLayerInfo info);


    // Inherited methods overridden:
    void        run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;
    const char *name() const override;
    
private:
    LayerNormLayerInfo         _info{};
    LayerNormKernelPtr         _run_method{nullptr};
    std::string                _name{};
};

} // namespace kernels
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_NELAYERNORMKERNEL_H */
