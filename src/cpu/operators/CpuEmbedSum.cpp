#include "src/cpu/operators/CpuEmbedSum.h"

#include "arm_compute/runtime/NEON/NEScheduler.h"

#include "src/common/IOperator.h"
#include "src/common/utils/LegacySupport.h"
#include "src/common/utils/Log.h"
#include "src/cpu/CpuContext.h"

#include "src/core/helpers/MemoryHelpers.h"
#include "src/cpu/utils/CpuAuxTensorHandler.h"


namespace arm_compute
{
namespace cpu
{
void CpuEmbedSum::configure(const ITensorInfo *token,
                            const ITensorInfo *segemnt,
                            const ITensorInfo *position,
                            ITensorInfo *output,
                            const EmbeddingLayerInfo &emb_info)
{
    _add_kernel_1 = std::make_unique<kernels::CpuAddKernel>();
    _add_kernel_2 = std::make_unique<kernels::CpuAddKernel>();

    _add_kernel_1->configure(token,segemnt,&_tmp_token_segment,emb_info.c_policy());

    _aux_mem[TokenSegmentOutput] =
                experimental::MemoryInfo(offset_int_vec(TokenSegmentOutput),
                                         experimental::MemoryLifetime::Persistent,
                                         _tmp_token_segment.total_size());
    
    _add_kernel_2->configure(&_tmp_token_segment,position,output,emb_info.c_policy());
}

Status
CpuEmbedSum::validate(const ITensorInfo *token,
                      const ITensorInfo *segemnt,
                      const ITensorInfo *position,
                      ITensorInfo *output,
                      const EmbeddingLayerInfo &emb_info)
{
    ARM_COMPUTE_UNUSED(token);
    ARM_COMPUTE_UNUSED(segemnt);
    ARM_COMPUTE_UNUSED(position);
    ARM_COMPUTE_UNUSED(output);
    ARM_COMPUTE_UNUSED(emb_info);
    return Status{};
}

void CpuEmbedSum::run(ITensorPack &tensors)
{
    ARM_COMPUTE_ERROR_ON_MSG(tensors.empty(), "No inputs provided");
    auto token      = tensors.get_const_tensor(ACL_SRC_0);
    auto segment    = tensors.get_const_tensor(ACL_SRC_1);
    auto position   = tensors.get_const_tensor(ACL_SRC_2);
    auto output     = tensors.get_tensor(ACL_DST);

    CpuAuxTensorHandler aux_token_segemnt(offset_int_vec(TokenSegmentOutput), _tmp_token_segment, tensors, true);

    ITensorPack run_pack{{ACL_SRC_0, token}, {ACL_SRC_1, segment}, {ACL_DST, aux_token_segemnt.get()}};

    // Reshape window if tensor valid region has been reshaped
    Window win = _add_kernel_1->window();

    size_t reshaped_x = token->info()->valid_region().shape.x() <  segment->info()->valid_region().shape.x()
                        ? token->info()->valid_region().shape.x() : segment->info()->valid_region().shape.x();
    reshaped_x = reshaped_x < position->info()->valid_region().shape.x() 
                        ? reshaped_x : position->info()->valid_region().shape.x();

    win.set(Window::DimX, Window::Dimension(0,reshaped_x,1));
    NEScheduler::get().schedule_op(_add_kernel_1.get(), Window::DimY, win, run_pack);

    run_pack.add_const_tensor(ACL_SRC_0,aux_token_segemnt.get());
    run_pack.add_const_tensor(ACL_SRC_1,position);
    run_pack.add_tensor(ACL_DST,output);

    NEScheduler::get().schedule_op(_add_kernel_2.get(), Window::DimY, win, run_pack);
}


} // namespace cpu
} // namespace arm_compute
