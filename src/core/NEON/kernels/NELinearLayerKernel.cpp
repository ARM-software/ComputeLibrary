#include "src/core/NEON/kernels/NELinearLayerKernel.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Validate.h"

#include "src/common/utils/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

namespace arm_compute
{
namespace kernels
{

void NELinearLayerKernel::configure(const ITensorInfo *input1,
                                    ITensorInfo       *output,
                                    LinearAttentionOperation   op)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input1, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate(input1, output, op));

    _op = op;

    Window      win       = calculate_max_window(*input1, Steps());
    TensorShape out_shape = input1->tensor_shape();

    ICPPKernel::configure(win);

    // Auto initialize if empty
    set_shape_if_empty(*output, out_shape);
    set_data_type_if_unknown(*output, input1->data_type());
}

Status NELinearLayerKernel::validate(const ITensorInfo *input1,
                                     const ITensorInfo *output,
                                     LinearAttentionOperation   op)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input1, 1, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON(op == LinearAttentionOperation::Unknown);

    TensorShape out_shape = input1->tensor_shape();
    if (op != LinearAttentionOperation::Key)
    {
        out_shape = TensorShape::broadcast_shape(input1->tensor_shape());
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(out_shape.total_size() == 0, "Inputs are not broadcast compatible");
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input1);
    }

    // Checks performed when output is configured
    if ((output != nullptr) && (output->total_size() != 0))
    {
        ARM_COMPUTE_RETURN_ERROR_ON(detail::have_different_dimensions(out_shape, output->tensor_shape(), 0));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input1, output);
    }

    return Status{};
}

void NELinearLayerKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(tensors.empty());

    const ITensor *src = tensors.get_const_tensor(TensorType::ACL_SRC_0);
    //ITensor       *dst  = tensors.get_tensor(TensorType::ACL_DST);

    Window win = window;
    win.use_tensor_dimensions(src);

    Iterator src_iter(src,win);
    const auto src_ptr      = reinterpret_cast<float *>(src_iter.ptr());

    std::cout << "src/core/NEON/kernels/NELinearLayerKernel.cpp" <<std::endl; 
    execute_window_loop(win,
    [&](const Coordinates &id){
        std::cout << *(src_ptr + id[0]) << std::endl;
        

    },src_iter);
    
}

const char *NELinearLayerKernel::name() const
{
    return "NELinearLayerKernel";
}

} // namespace kernels
} // namespace arm_compute
