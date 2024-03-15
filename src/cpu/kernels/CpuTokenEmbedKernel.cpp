#include "src/cpu/kernels/CpuTokenEmbedKernel.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"

#include "src/core/common/Registrars.h"
#include "src/cpu/kernels/tokenembed/list.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{

namespace
{
static const std::vector<CpuTokenEmbedKernel::TKEMBKernel> avaiable_kernels = {
    /*
#ifdef ARM_COMPUTE_ENABLE_SVE
    // TBA
#endif // ARM_COMPUTE_ENABLE_SVE

#ifdef __aarch64__
    // TBA
#endif // __aarch64__
*/
    {"neon_fp32_token_embedding", [](const TokenEmbedKernelDataTypeISASelectorData &data) { return data.dt == DataType::F32; },
     REGISTER_FP32_NEON(arm_compute::cpu::neon_token_embed_char_2_float32)}

};
}

void CpuTokenEmbedKernel::configure(const ITensorInfo *src, ITensorInfo *dst, TokenEmbeddingLayerInfo tkemb_info)
{
    ARM_COMPUTE_UNUSED(dst);
    ARM_COMPUTE_ERROR_ON_NULLPTR(src);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, dst, tkemb_info));

    std::cout << "src/cpu/kernels/CpuTokenEmbedKernel.cpp: neon_token_embed_char_2_float32" << std::endl;

    std::cout << src->id() << std::endl;
    std::cout << dst->id() << std::endl;
    std::cout << tkemb_info.d_vocab() << std::endl;

}

} // namespace kernels
} // namespace cpu
} // namespace arm_compute