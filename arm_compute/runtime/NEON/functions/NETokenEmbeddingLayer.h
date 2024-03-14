#ifndef ARM_COMPUTE_NETOKENEMBEDDINGLAYER_H
#define ARM_COMPUTE_NETOKENEMBEDDINGLAYER_H


#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/NEON/INEOperator.h"

namespace arm_compute
{
// Forward Declarations
class ITensor;

/** Basic function to run @ref NETokenEmbeddingLayerKernel */
class NETokenEmbeddingLayer : public IFunction
{
public:
    /** Default Constructor */
    NETokenEmbeddingLayer();
    /** Default Destructor */
    ~NETokenEmbeddingLayer();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NETokenEmbeddingLayer(const NETokenEmbeddingLayer &) = delete;
    /** Default move constructor */
    NETokenEmbeddingLayer(NETokenEmbeddingLayer &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NETokenEmbeddingLayer &operator=(const NETokenEmbeddingLayer &) = delete;
    /** Default move assignment operator */
    NETokenEmbeddingLayer &operator=(NETokenEmbeddingLayer &&);

    /** Configure kernel
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src    |dst    |
     * |:------|:------|
     * |All    |All    |
     *
     * @note Supported tensor rank: up to 4
     *
     * @param[in]  input            Source tensor. Data type supported: All
     * @param[out] output           Destination tensor. Data type supported: Same as @p input
     * @param[in]  starts           The starts of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     * @param[in]  ends             The ends of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     * @param[in]  strides          The strides of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     * @param[in]  begin_mask       (Optional) If the ith bit of begin_mask is set, starts[i] is ignored and the fullest possible range in that dimension is used instead.
     * @param[in]  end_mask         (Optional) If the ith bit of end_mask is set, ends[i] is ignored and the fullest possible range in that dimension is used instead.
     * @param[in]  shrink_axis_mask (Optional) If the ith bit of shrink_axis_mask is set, it implies that the ith specification shrinks the dimensionality by 1.
     *                              A slice of size 1 starting from starts[i] in the dimension must be preserved.
     */
    void configure(const ITensor     *input,
                   ITensor           *output,
                   const Coordinates &starts,
                   const Coordinates &ends,
                   const BiStrides   &strides,
                   int32_t            begin_mask       = 0,
                   int32_t            end_mask         = 0,
                   int32_t            shrink_axis_mask = 0);

    /** Static function to check if given info will lead to a valid configuration of @ref NETokenEmbeddingLayer
     *
     * @note Supported tensor rank: up to 4
     *
     * @param[in] input            Source tensor info. Data type supported: All
     * @param[in] output           Destination tensor info. Data type supported: Same as @p input
     * @param[in] starts           The starts of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     * @param[in] ends             The ends of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     * @param[in] strides          The strides of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     * @param[in] begin_mask       (Optional) If the ith bit of begin_mask is set, starts[i] is ignored and the fullest possible range in that dimension is used instead.
     * @param[in] end_mask         (Optional) If the ith bit of end_mask is set, ends[i] is ignored and the fullest possible range in that dimension is used instead.
     * @param[in] shrink_axis_mask (Optional) If the ith bit of shrink_axis_mask is set, it implies that the ith specification shrinks the dimensionality by 1.
     *                             A slice of size 1 starting from starts[i] in the dimension must be preserved.
     */
    static Status validate(const ITensorInfo *input,
                           const ITensorInfo *output,
                           const Coordinates &starts,
                           const Coordinates &ends,
                           const BiStrides   &strides,
                           int32_t            begin_mask       = 0,
                           int32_t            end_mask         = 0,
                           int32_t            shrink_axis_mask = 0);

    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace arm_compute

#endif /* ARM_COMPUTE_NETOKENEMBEDDINGLAYER_H */