#ifndef ARM_COMPUTE_CPU_SCALE_DOT_PRODUCTION_H
#define ARM_COMPUTE_CPU_SCALE_DOT_PRODUCTION_H


#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"

#include "src/cpu/ICpuOperator.h"
#include "src/cpu/kernels/CpuScaleKernel.h"
#include "src/cpu/kernels/CpuSoftmaxKernel.h"
#include "src/cpu/operators/CpuTranspose.h"
#include "src/cpu/kernels/CpuGemmMatrixMultiplyKernel.h"
#include "src/cpu/operators/CpuActivation.h"
#include "src/cpu/kernels/CpuGemmInterleave4x4Kernel.h"
#include "src/cpu/kernels/CpuGemmTranspose1xWKernel.h"

#include "src/cpu/kernels/CpuReshapeKernel.h"
#include "src/cpu/operators/CpuPermute.h"
#include "src/cpu/operators/CpuSoftmax.h"
#include "src/cpu/operators/CpuGemm.h"

#include <memory>

namespace arm_compute
{
namespace cpu
{
/** Function implementation for scale dot production, uses kernels:
 * @ref kernels::CpuEmbedKernel
*/
class CpuScaleDotProduction : public ICpuOperator
{
public:
    /** Constructor */
    CpuScaleDotProduction() = default;
    /** Destructor */
    ~CpuScaleDotProduction() = default;
    
    /** Configure operator for a given list of arguments
     * 
     * @param[in]  query           Attention key tensor info. Data types supported: F32.
     * @param[in]  key             Attention key tensor info. Data types supported: F32.
     * @param[in]  value           Attention value tensor info. Data types supported: F32.
     * @param[out] output          Destination tensor info. Data type supported: F32
     */
    void configure( const ITensorInfo *query, const ITensorInfo *key, const ITensorInfo *value, ITensorInfo *output, const ScaleDotProductionAttentionLayerInfo& info);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref CpuScaleDotProduction::configure()
     *
     * @return a status
     */
    static Status validate( const ITensorInfo *query, const ITensorInfo *key, const ITensorInfo *value, ITensorInfo *output);

    void transpose(ITensorPack &tensors);

    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;
    experimental::MemoryRequirements workspace() const override;

private:
    enum AuxTensorIdx
    {
        /* Slots 0 - 2 reserved for CpuGemmAssemblyDispatch */
        InterleavedLHS = 3,
        Transposed1xWRHS,
        InterleavedProduct,
        Transposed1xWValue,
        QueryReshape,
        QueryPermute,
        KeyReshape,
        KeyPermute,
        ValueReshape,
        ValuePermute,
        KeyTranspose,
        QueryKeyScale,
        Softmax,
        GemmedContext,
        ConcatPermute,
        Count
    };

    std::unique_ptr<kernels::CpuGemmInterleave4x4Kernel>    _query_interleave_kernel{nullptr};
    std::unique_ptr<kernels::CpuGemmMatrixMultiplyKernel>   _product_mm_kernel{nullptr};
    std::unique_ptr<kernels::CpuGemmTranspose1xWKernel>     _key_transpose1xW_kernel{nullptr};
    std::unique_ptr<CpuActivation>                          _scale_func{nullptr};
    std::unique_ptr<kernels::CpuGemmInterleave4x4Kernel>    _product_interleave_kernel{nullptr};
    std::unique_ptr<kernels::CpuGemmMatrixMultiplyKernel>   _context_mm_kernel{nullptr};
    std::unique_ptr<kernels::CpuGemmTranspose1xWKernel>     _value_transpose1xW_kernel{nullptr};
    

    std::unique_ptr<kernels::CpuReshapeKernel>              _query_reshape_kernel{nullptr};
    std::unique_ptr<CpuPermute>                             _query_permute_func{nullptr};
    std::unique_ptr<kernels::CpuReshapeKernel>              _key_reshape_kernel{nullptr};
    std::unique_ptr<CpuPermute>                             _key_permute_func{nullptr};
    std::unique_ptr<kernels::CpuReshapeKernel>              _value_reshape_kernel{nullptr};
    std::unique_ptr<CpuPermute>                             _value_permute_func{nullptr};
    std::unique_ptr<kernels::CpuReshapeKernel>              _concat_reshape_kernel{nullptr};
    std::unique_ptr<CpuPermute>                             _concat_permute_func{nullptr};
    std::unique_ptr<CpuTranspose>                           _key_transpose_func{nullptr};
    std::unique_ptr<CpuSoftmaxGeneric>                      _softmax_func{nullptr};

    

    TensorInfo _tmp_query{};
    TensorInfo _tmp_key{};
    TensorInfo _interleaved_product{};
    TensorInfo _transposed1xW_value{};

    TensorInfo _reshaped_query{};
    TensorInfo _permuted_query{};
    TensorInfo _reshaped_key{};
    TensorInfo _permuted_key{};
    TensorInfo _reshaped_value{};
    TensorInfo _permuted_value{};
    TensorInfo _permuted_concat{};
    TensorInfo _transposed_key{};
    TensorInfo _scaled_query_key{};
    TensorInfo _softmaxed_product{};
    TensorInfo _gemmed_context{};

    bool _run_pretranspose{false};
    bool _run_scale{false};
    bool _run_vector_matrix_multiplication{false};
    bool _run_interleave_transpose{
        true}; /**< If we run CpuGemmInterleave4x4Kernel on lhs and CpuGemmTranspose1xWKernel on rhs */

    experimental::MemoryRequirements _aux_mem{Count};

};
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_SCALE_DOT_PRODUCTION_H */
