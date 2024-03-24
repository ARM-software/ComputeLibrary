#include "src/cpu/operators/CpuTokenEmbed.h"

#include "arm_compute/runtime/NEON/NEScheduler.h"

#include "src/common/IOperator.h"
#include "src/common/utils/LegacySupport.h"
#include "src/common/utils/Log.h"
#include "src/cpu/CpuContext.h"
#include "src/cpu/kernels/CpuTokenEmbedKernel.h"
#include "src/cpu/utils/CpuAuxTensorHandler.h"
#include "src/core/helpers/MemoryHelpers.h"

#include <memory>

using namespace arm_compute::experimental;

namespace arm_compute
{
namespace cpu
{
void CpuTokenEmbed::configure(const ITensorInfo *input,  ITensorInfo *output, const TokenEmbeddingLayerInfo &tkemb_info)
{
    ARM_COMPUTE_LOG_PARAMS(input, output, tkemb_info);
    auto k = std::make_unique<kernels::CpuTokenEmbedKernel>();
    k->configure(input, &_tmp_t2p, tkemb_info);
    _kernel = std::move(k);

    _PE_kernel = std::make_unique<kernels::CpuPositionalEncodingKernel>();
    _PE_kernel->configure(&_tmp_t2p,output,tkemb_info.d_model());
}

Status
CpuTokenEmbed::validate(const ITensorInfo *input, const ITensorInfo *vocab, const ITensorInfo *output,const TokenEmbeddingLayerInfo &tkemb_info)
{
    ARM_COMPUTE_UNUSED(input);
    ARM_COMPUTE_UNUSED(vocab);
    ARM_COMPUTE_UNUSED(output);
    ARM_COMPUTE_UNUSED(tkemb_info);
    return Status{};
}

void CpuTokenEmbed::run(ITensorPack &tensors)
{
    ARM_COMPUTE_ERROR_ON_MSG(tensors.empty(), "No inputs provided");
    auto split_dimension = static_cast<kernels::CpuTokenEmbedKernel *>(_kernel.get())->get_split_dimension_hint();

    const ITensor *src   = tensors.get_const_tensor(TensorType::ACL_SRC_0);
    const ITensor *vocab = tensors.get_const_tensor(TensorType::ACL_SRC_1);
    ITensor       *dst   = tensors.get_tensor(TensorType::ACL_DST);

    NEScheduler::get().schedule_op(_kernel.get(), split_dimension, _kernel->window(), tensors);
    std::cout << "src/cpu/operators/CpuTokenEmbed.cpp 4  " << std::endl;
    NEScheduler::get().schedule_op(_PE_kernel.get(),Window::DimY,_PE_kernel->window(), tensors);
    std::cout << "src/cpu/operators/CpuTokenEmbed.cpp 5  " << std::endl;
}


} // namespace cpu
} // namespace arm_compute
