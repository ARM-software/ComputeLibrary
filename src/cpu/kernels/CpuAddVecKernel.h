#ifndef ARM_COMPUTE_CPU_ADD_VEC_KERNEL_H
#define ARM_COMPUTE_CPU_ADD_VEC_KERNEL_H

#include "src/core/common/Macros.h"
#include "src/cpu/ICpuKernel.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
/** Interface for the kernel to perform addition between two tensors */
class CpuAddVecKernel : public ICpuKernel<CpuAddVecKernel>
{
private:
    using AddVecKernelPtr = std::add_pointer<void(
        const ITensor *, const ITensor *, ITensor *, const ConvertPolicy &, const Window &)>::type;

public:
    struct AddKernel
    {
        const char                                  *name;
        const CpuAddVecKernelDataTypeISASelectorDataPtr is_selected;
        AddVecKernelPtr                                 ukernel;
    };

    CpuAddVecKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuAddVecKernel);
    /** Initialise the kernel's input, dst and border mode.
     *
     * Valid configurations (src0,src1) -> dst :
     *
     *   - (U8,U8)           -> U8
     *   - (S16,S16)         -> S16
     *   - (S32,S32)         -> S32
     *   - (F16,F16)         -> F16
     *   - (F32,F32)         -> F32
     *   - (QASYMM8,QASYMM8) -> QASYMM8
     *   - (QASYMM8_SIGNED,QASYMM8_SIGNED) -> QASYMM8_SIGNED
     *   - (QSYMM16,QSYMM16) -> QSYMM16
     *
     * @param[in]  src0   First input tensor info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/F16/S32/F32
     * @param[in]  src1   Second input tensor info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/F16/S32/F32
     * @param[out] dst    The dst tensor info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/F16/S32/F32.
     * @param[in]  policy Overflow policy.
     */
    void configure(const ITensorInfo *src0, const ITensorInfo *src1, ITensorInfo *dst, ConvertPolicy policy);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to CpuAddVecKernel::configure()
     *
     * @return a status
     */
    static Status
    validate(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *dst, ConvertPolicy policy);

    // Inherited methods overridden:
    void        run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;
    const char *name() const override;

    /** Return minimum workload size of the relevant kernel
     *
     * @param[in] platform     The CPU platform used to create the context.
     * @param[in] thread_count Number of threads in the execution.
     *
     * @return[out] mws Minimum workload size for requested configuration.
     */
    size_t get_mws(const CPUInfo &platform, size_t thread_count) const override;

    static const std::vector<AddKernel> &get_available_kernels();

    size_t get_split_dimension() const
    {
        return _split_dimension;
    }

private:
    ConvertPolicy   _policy{};
    AddVecKernelPtr _run_method{nullptr};
    std::string     _name{};
    size_t          _split_dimension{Window::DimY};
};
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_ADD_KERNEL_H */
