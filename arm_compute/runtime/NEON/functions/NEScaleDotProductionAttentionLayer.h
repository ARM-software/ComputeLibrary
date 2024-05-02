#ifndef ARM_COMPUTE_NESCALEDOTPRODUCTIONATTENTIONLAYER_H
#define ARM_COMPUTE_NESCALEDOTPRODUCTIONATTENTIONLAYER_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IRuntimeContext.h"
#include "arm_compute/runtime/Tensor.h"

#include <memory>

namespace arm_compute
{
// Forward declarations
class ITensor;
class ITensorInfo;

class NEScaleDotProductionAttentionLayer : public IFunction
{
public:
    /** Default Constructor */
    NEScaleDotProductionAttentionLayer(std::shared_ptr<IMemoryManager> memory_manager);
    /** Default Destructor */
    ~NEScaleDotProductionAttentionLayer();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEScaleDotProductionAttentionLayer(const NEScaleDotProductionAttentionLayer &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEScaleDotProductionAttentionLayer &operator=(const NEScaleDotProductionAttentionLayer &) = delete;

    /** Set the input and output tensor.
     * 
     * @param[in]  query      Input tenser of Attention Query, Data type supported: F32
     * @param[in]  key        Input tensor of Attention Key, Data type supported: F32
     * @param[in]  value      Input tenser of Attention Value, Data type supported: F32
     * @param[out] output     Output tensor, shape (d_model,d_model). Data type supported: F32
     */
    void configure(const ITensor *query,const ITensor *key,const ITensor *value, ITensor *output, const ScaleDotProductionAttentionLayerInfo& info);
    /** Static function to check if given info will lead to a valid configuration of @ref NEScaleDotProductionAttentionLayer
     *
     * @param[in] output Destination tensor info. Data type supported: same as @p input
     *
     * @return a status
     */
    static Status validate(ITensor *output);

    // Inherited methods overridden:
    void run() override;
private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace arm_compute

#endif /* ARM_COMPUTE_NESCALEDOTPRODUCTIONATTENTIONLAYER_H */