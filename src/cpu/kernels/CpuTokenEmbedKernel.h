#ifndef SRC_CPU_KERNELS_CPU_TOKEN_EMBED_KERNEL_H
#define SRC_CPU_KERNELS_CPU_TOKEN_EMBED_KERNEL_H


#include "src/core/common/Macros.h"
#include "src/cpu/ICpuKernel.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
/** Interface for the token embedding kernel */
class CpuTokenEmbedKernel : public ICpuKernel<CpuTokenEmbedKernel>
{
private:
    using TKEMBKernelPtr =
        std::add_pointer<void(const ITensor *, ITensor *, const TokenEmbeddingLayerInfo &, const Window &)>::type;
public:
    /* Default Constructor */
    CpuTokenEmbedKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuTokenEmbedKernel);
    /** Configure kernel for a given list of arguments
     *
     * @param[in]   src             Source tensor info. Data types supported: U8.
     * @param[in]   vocab           Char 2 Vec const tensor info, Data type supported: F32
     * @param[out]  dst             Destination tensor info. Data type supported: F32
     * @param[in]   tkemb_info      Token embedding layer information.
     */
    void configure(const ITensorInfo *src, const ITensorInfo *vocab, ITensorInfo *dst, TokenEmbeddingLayerInfo tkemb_info);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to CpuTokenEmbedKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, ITensorInfo *dst, TokenEmbeddingLayerInfo tkemb_info);

    /** Return minimum workload size of the relevant kernel
     *
     * @param[in] platform     The CPU platform used to create the context.
     * @param[in] thread_count Number of threads in the execution.
     *
     * @return[out] small_network_mws          Minimum workload size for requsted configuration.
     */
    size_t get_mws(const CPUInfo &platform, size_t thread_count) const override;

    // Inherited methods overridden:
    void        run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;
    const char *name() const override;

    /** Get the preferred dimension in which the scheduler splits the work into multiple jobs.
     *
     * @return The split dimension hint.
     */
    size_t get_split_dimension_hint() const
    {
        return _split_dimension;
    }

    struct TKEMBKernel
    {
        const char                                          *name;
        const TokenEmbedKernelDataTypeISASelectorDataPtr    is_selected;
        TKEMBKernelPtr                                      ukernel;
    };

    static const std::vector<TKEMBKernel> &get_available_kernels();

    
private:
    TokenEmbeddingLayerInfo _tkemb_info{};
    TKEMBKernelPtr          _run_method{nullptr};
    size_t                  _split_dimension{Window::DimY};
    std::string             _name{};
};

} // kernels
} // namespace cpu
} // namespace arm_compute

#endif /* SRC_CPU_KERNELS_CPU_TOKEN_EMBED_KERNEL_H */