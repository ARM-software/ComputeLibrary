#ifndef ARM_COMPUTE_NELINEARLAYERKERNEL_H
#define ARM_COMPUTE_NELINEARLAYERKERNEL_H

#include "src/core/common/Macros.h"

#include "src/core/KernelTypes.h"
#include "src/cpu/ICpuKernel.h"

namespace arm_compute
{
namespace cpu
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
class CpuLinearKernel : public ICpuKernel<CpuLinearKernel>
{
private:
    using LinearKernelPtr =
        std::add_pointer<void(const ITensor *, ITensor *, const LinearLayerInfo &, const Window &)>::type;
public:
    /* Default Constructor */
    CpuLinearKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuLinearKernel);

    /** Initialise the kernel's inputs and output
     *
     * @param[in]  input1 An input tensor. Data type supported: f32.
     * @param[out] output Output tensor. Data type supported: F32.
     * @param[out] op     Logical operation to perform
     */
    void configure(const ITensorInfo *input1, ITensorInfo *output, LinearLayerInfo info);
    /** Static function to check if given info will lead to a valid configuration of @ref CpuLinearKernel
     *
     * @param[in] input1 An input tensor. Data type supported: F32.
     * @param[in] output Output tensor. Data type supported: F32..
     * @param[in] op     Logical operation to perform
     *
     * @return a status
     */
    static Status
    validate(const ITensorInfo *input1,  const ITensorInfo *output, LinearLayerInfo info);


    // Inherited methods overridden:
    void        run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;
    const char *name() const override;

    struct LinearKernel
    {
        const char                                          *name;
        const TokenEmbedKernelDataTypeISASelectorDataPtr    is_selected;
        LinearKernelPtr                                      ukernel;
    };

    static const std::vector<LinearKernel> &get_available_kernels();

    
private:
    LinearLayerInfo         _info;
    LinearKernelPtr         _run_method{nullptr};
    size_t                  _split_dimension{Window::DimY};
    std::string             _name{};
};

} // namespace kernels
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_NELINEARLAYERKERNEL_H */
