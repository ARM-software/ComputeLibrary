#ifndef ARM_COMPUTE_CPU_POSITION_EMBEDDING_KERNEL_H
#define ARM_COMPUTE_CPU_POSITION_EMBEDDING_KERNEL_H

#include "src/core/common/Macros.h"
#include "src/cpu/ICpuKernel.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
/** Kernel to perform tensor position Embedding */
class CpuPositionEmbeddingKernel : public ICpuKernel<CpuPositionEmbeddingKernel>
{
public:
    /** Default constructor */
    CpuPositionEmbeddingKernel() = default;
    /** Default destructor */
    ~CpuPositionEmbeddingKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuPositionEmbeddingKernel);
    /** Configure kernel for a given list of arguments
     *
     * @note Arbitrary permutation vectors are supported with rank not greater than 4
     *
     * @param[in]  src  Srouce tensor to permute. Data types supported: All
     * @param[in]  pos  Pretrained position embedding. Data types supported: All
     * @param[out] dst  Destination tensor. Data types supported: Same as @p src
     * @param[in]  perm Permutation vector
     */
    void configure(const ITensorInfo *src, const ITensorInfo *pos, ITensorInfo *dst, const unsigned int d_model);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref CpuPositionEmbeddingKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *pos, const ITensorInfo *dst, const unsigned int d_model);

    // Inherited methods overridden:
    void        run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;
    const char *name() const override;

private:
    unsigned int _d_model{512U};
};
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_POSITION_EMBEDDING_KERNEL_H */
