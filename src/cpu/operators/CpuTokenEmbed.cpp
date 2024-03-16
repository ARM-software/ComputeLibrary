#include "src/cpu/operators/CpuTokenEmbed.h"

#include "arm_compute/runtime/NEON/NEScheduler.h"

#include "src/common/IOperator.h"
#include "src/common/utils/LegacySupport.h"
#include "src/common/utils/Log.h"
#include "src/cpu/CpuContext.h"
#include "src/cpu/kernels/CpuTokenEmbedKernel.h"

namespace arm_compute
{
namespace cpu
{
void CpuTokenEmbed::configure(const ITensorInfo *input, const ITensorInfo *vocab, ITensorInfo *output, const TokenEmbeddingLayerInfo &tkemb_info)
{
    ARM_COMPUTE_LOG_PARAMS(input, output, tkemb_info);
    auto k = std::make_unique<kernels::CpuTokenEmbedKernel>();
    k->configure(input, vocab, output, tkemb_info);
    _kernel = std::move(k);
    
}

Status
CpuTokenEmbed::validate(const ITensorInfo *input, const ITensorInfo *vocab, const ITensorInfo *output,const TokenEmbeddingLayerInfo &tkemb_info)
{
    std::cout << "src/cpu/operators/CpuTokenEmbed.cpp: validate " << std::endl;
    std::cout << input->id() << std::endl;
    std::cout << vocab->id() << std::endl;
    std::cout << output->id() << std::endl;
    std::cout << tkemb_info.d_vocab() << std::endl;
    return Status{};
}

void CpuTokenEmbed::run(ITensorPack &tensors)
{
    ARM_COMPUTE_ERROR_ON_MSG(tensors.empty(), "No inputs provided");
    auto split_dimension = static_cast<kernels::CpuTokenEmbedKernel *>(_kernel.get())->get_split_dimension_hint();
    NEScheduler::get().schedule_op(_kernel.get(), split_dimension, _kernel->window(), tensors);
}


} // namespace cpu
} // namespace arm_compute
