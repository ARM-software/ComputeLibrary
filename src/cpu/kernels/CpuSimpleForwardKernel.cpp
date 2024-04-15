#include "src/cpu/kernels/CpuSimpleForwardKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include "src/core/common/Registrars.h"
#include "src/core/CPP/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/Utils.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/cpu/kernels/softmax/list.h"

#include <vector>

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace
{

} // namespace

void CpuSimpleForwardKernel::configure(const ITensorInfo *src1,
                                       const ITensorInfo *src2,
                                       const ITensorInfo *src3,
                                       ITensorInfo *dst1,
                                       ITensorInfo *dst2,
                                       ITensorInfo *dst3)
{
    auto_init_if_empty(*dst1, src1->clone()->set_tensor_shape(src1->tensor_shape()));
    auto_init_if_empty(*dst2, src2->clone()->set_tensor_shape(src2->tensor_shape()));
    auto_init_if_empty(*dst3, src3->clone()->set_tensor_shape(src3->tensor_shape()));

    Window win;

    win = calculate_max_window(*dst1, Steps());
    ICPPKernel::configure(win);
}


void CpuSimpleForwardKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_UNUSED(window);
    ARM_COMPUTE_UNUSED(tensors);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    
    const auto src1 = tensors.get_const_tensor(TensorType::ACL_SRC_0);
    auto       dst1 = tensors.get_tensor(TensorType::ACL_DST_0);
    const auto src2 = tensors.get_const_tensor(TensorType::ACL_SRC_1);
    auto       dst2 = tensors.get_tensor(TensorType::ACL_DST_1);
    const auto src3 = tensors.get_const_tensor(TensorType::ACL_SRC_2);
    auto       dst3 = tensors.get_tensor(TensorType::ACL_DST_2);

    std::cout << "src/cpu/kernels/CpuSimpleForwardKernel.cpp Runnnn" << std::endl;
    std::cout << src1->info()->total_size() <<std::endl;
    std::cout << src2->info()->total_size() <<std::endl;
    std::cout << src3->info()->total_size() <<std::endl;

    std::cout << dst1->info()->total_size() <<std::endl;
    std::cout << dst2->info()->total_size() <<std::endl;
    std::cout << dst3->info()->total_size() <<std::endl;

    dst1->copy_from(*src1);
    dst2->copy_from(*src2);
    dst3->copy_from(*src3);
    std::cout << "src/cpu/kernels/CpuSimpleForwardKernel.cpp Runnnn" << std::endl;

    std::cout << src1->info()->tensor_shape().x() <<std::endl;
    std::cout << src1->info()->tensor_shape().y() <<std::endl;
    std::cout << src1->info()->tensor_shape().z() <<std::endl;

    std::cout << dst1->info()->tensor_shape().x() <<std::endl;
    std::cout << dst1->info()->tensor_shape().y() <<std::endl;
    std::cout << dst1->info()->tensor_shape().z() <<std::endl;

    std::cout << src1->is_used() << "  " << dst1->is_used() << std::endl;
    std::cout << src2->is_used() << "  " << dst2->is_used() << std::endl;
    std::cout << src3->is_used() << "  " << dst3->is_used() << std::endl;

    std::cout << "src/cpu/kernels/CpuSimpleForwardKernel.cpp Runnnn" << std::endl;
    std::cout << src1->ptr_to_element(Coordinates(0,0)) <<std::endl;
    std::cout << src2->ptr_to_element(Coordinates(0,0)) <<std::endl;
    std::cout << src3->ptr_to_element(Coordinates(0,0)) <<std::endl;

    std::cout << dst1->ptr_to_element(Coordinates(0,0)) <<std::endl;
    std::cout << dst2->ptr_to_element(Coordinates(0,0)) <<std::endl;
    std::cout << dst3->ptr_to_element(Coordinates(0,0)) <<std::endl;
    std::cout << "src/cpu/kernels/CpuSimpleForwardKernel.cpp Runnnn" << std::endl;
    
}

const char *CpuSimpleForwardKernel::name() const
{
    return "CpuSimpleForwardKernel";
}

} // namespace kernels
} // namespace cpu
} // namespace arm_compute
