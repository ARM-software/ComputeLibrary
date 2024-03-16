#include "src/cpu/kernels/CpuTokenEmbedKernel.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"

#include "src/core/common/Registrars.h"
#include "src/core/CPP/Validate.h"
#include "src/cpu/kernels/tokenembed/list.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{

namespace
{
static const std::vector<CpuTokenEmbedKernel::TKEMBKernel> available_kernels = {
    /*
#ifdef ARM_COMPUTE_ENABLE_SVE
    // TBA
#endif // ARM_COMPUTE_ENABLE_SVE

#ifdef __aarch64__
    // TBA
#endif // __aarch64__
*/
    {"neon_fp32_token_embedding", [](const TokenEmbedKernelDataTypeISASelectorData &data) { return data.dt == DataType::F16; },
     REGISTER_FP32_NEON(arm_compute::cpu::neon_token_embed_char_2_float32)},
    {"neon_fp32_token_embedding", [](const TokenEmbedKernelDataTypeISASelectorData &data) { return data.dt == DataType::F32; },
     REGISTER_FP32_NEON(arm_compute::cpu::neon_token_embed_char_2_float32)}

};
}

void CpuTokenEmbedKernel::configure(const ITensorInfo *src, const ITensorInfo *vocab, ITensorInfo *dst, TokenEmbeddingLayerInfo tkemb_info)
{
    ARM_COMPUTE_UNUSED(dst);
    ARM_COMPUTE_ERROR_ON_NULLPTR(src);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, vocab, dst, tkemb_info));
    
    const auto uk = CpuTokenEmbedKernel::get_implementation(
        TokenEmbedKernelDataTypeISASelectorData{dst->data_type(), CPUInfo::get().get_isa()}
    );

    ARM_COMPUTE_ERROR_ON_NULLPTR(uk);

    _run_method = uk->ukernel;
    _name       = std::string("CpuTokenEmbedKernel").append("/").append(uk->name);

    std::cout << "src/cpu/kernels/CpuTokenEmbedKernel.cpp: neon_token_embed_char_2_float32" << std::endl;

    std::cout << src->id() << std::endl;
    std::cout << vocab->id() << std::endl;
    std::cout << dst->id() << std::endl;
    std::cout << tkemb_info.d_vocab() << std::endl;

}

Status CpuTokenEmbedKernel::validate(const ITensorInfo *src, ITensorInfo *dst, TokenEmbeddingLayerInfo tkemb_info)
{
    ARM_COMPUTE_UNUSED(tkemb_info);

    std::cout << "src/cpu/kernels/CpuTokenEmbedKernel.cpp: nvalidate" << std::endl;
    std::cout << src->id() << std::endl;
    std::cout << dst->id() << std::endl;
    std::cout << tkemb_info.d_vocab() << std::endl;

    return Status{};
}


size_t CpuTokenEmbedKernel::get_mws(const CPUInfo &platform, size_t thread_count) const
{
    ARM_COMPUTE_UNUSED(thread_count);
    ARM_COMPUTE_UNUSED(platform);

    if (_split_dimension == Window::DimX)
    {
        // Don't split the work load too small if the tensor has been reinterpreted as 1D.
        // This number is loosely chosen as threading overhead in each platform varies wildly.
        return 1536;
    }
    return default_mws;
}

void CpuTokenEmbedKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{

    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    ARM_COMPUTE_ERROR_ON(tensors.empty());
    ARM_COMPUTE_ERROR_ON(_run_method == nullptr);

    const ITensor *src   = tensors.get_const_tensor(TensorType::ACL_SRC_0);
    const ITensor *vocab = tensors.get_const_tensor(TensorType::ACL_SRC_1);
    ITensor       *dst   = tensors.get_tensor(TensorType::ACL_DST);

    std::cout << "src/cpu/kernels/CpuTokenEmbedKernel.cpp: run_op()!!!! " << std::endl;

    _run_method(src, vocab, dst, _tkemb_info, window);
}

const char *CpuTokenEmbedKernel::name() const
{
    return _name.c_str();
}

const std::vector<CpuTokenEmbedKernel::TKEMBKernel> &CpuTokenEmbedKernel::get_available_kernels()
{
    return available_kernels ;
}

} // namespace kernels
} // namespace cpu
} // namespace arm_compute