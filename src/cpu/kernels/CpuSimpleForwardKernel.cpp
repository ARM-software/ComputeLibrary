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

void CpuSimpleForwardKernel::configure(unsigned int total_nodes)
{
    _total_nodes = total_nodes;
}


void CpuSimpleForwardKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    std::cout << "src/cpu/kernels/CpuSimpleForwardKernel.cpp" << std::endl;
    for(unsigned int idx = 0; idx < _total_nodes; idx++){
        ITensor *src = tensors.get_tensor(TensorType::ACL_SRC_0+idx);
        ITensor *dst = tensors.get_tensor(TensorType::ACL_DST_0+idx);
    }
}

const char *CpuSimpleForwardKernel::name() const
{
    return _name.c_str();
}

} // namespace kernels
} // namespace cpu
} // namespace arm_compute
