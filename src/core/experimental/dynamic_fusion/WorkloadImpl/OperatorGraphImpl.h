/*
 * Copyright (c) 2022 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#ifdef ENABLE_EXPERIMENTAL_DYNAMIC_FUSION
#ifndef ARM_COMPUTE_EXPERIMENTAL_DYNAMICFUSION_OPERATORGRAPHIMPL
#define ARM_COMPUTE_EXPERIMENTAL_DYNAMICFUSION_OPERATORGRAPHIMPL

#include "arm_compute/core/experimental/ClWorkload.h"
#include "src/core/experimental/dynamic_fusion/WorkloadImpl/ITensorDescPack.h"

#include "support/Cast.h"
#include "support/DeepCopy.h"

#include <map>
#include <tuple>
#include <type_traits>

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
enum class OperatorComplexity
{
    Complex = 0,
    Simple
};

struct ClKernelGraph;
struct OpTensorContent
{
public:
    using Id          = DependencyGraph::Id;
    OpTensorContent() = default;
    OpTensorContent(Id id)
        : id{ id }, desc{}
    {
    }
    OpTensorContent(Id id, ITensorInfo *desc)
        : id{ id }, desc{ desc }
    {
    }
    ~OpTensorContent()                       = default;
    OpTensorContent(const OpTensorContent &) = default;
    OpTensorContent &operator=(const OpTensorContent &) = default;
    OpTensorContent(OpTensorContent &&)                 = default;
    OpTensorContent &operator=(OpTensorContent &&) = default;
    bool operator==(const OpTensorContent &other) const
    {
        return desc == other.desc;
    }

    const ITensorInfo *get_tensor_info() const
    {
        return desc;
    }
    ITensorInfo *get_tensor_info()
    {
        return desc;
    }

    Id           id{};
    ITensorInfo *desc{};
};

struct OperatorContent
{
public:
    using Id          = DependencyGraph::Id;
    OperatorContent() = default;
    OperatorContent(const OperatorGraph::Implementation *graph, Id id, const ITensorDescPack<OpTensorContent> &tensors)
        : _graph{ graph }, _id{ id }, _tensors{ tensors }
    {
    }
    OperatorContent(const OperatorContent &op) = default;
    OperatorContent &operator=(const OperatorContent &op) = default;
    OperatorContent(OperatorContent &&op)                 = default;
    OperatorContent &operator=(OperatorContent &&op)            = default;
    virtual ~OperatorContent()                                  = default;
    virtual OperatorComplexity complexity() const               = 0;
    virtual bool operator==(const OperatorContent &other) const = 0;
    virtual Status translate(ClKernelGraph &kernel_graph) const = 0;

protected:
    const OperatorGraph::Implementation *_graph {};
    Id                                   _id{};
    ITensorDescPack<OpTensorContent>     _tensors{};
};

struct Conv2dContent : public OperatorContent
{
public:
    Conv2dContent() = default;
    Conv2dContent(const OperatorGraph::Implementation *graph, Id id, const Conv2dDescriptor &desc, const ITensorDescPack<OpTensorContent> &tensors)
        : OperatorContent(graph, id, tensors), desc(desc), forced_method(), forced_method_enabled(false)
    {
    }
    // Temporary. Do not need to pass ConvolutionMethod
    Conv2dContent(const OperatorGraph::Implementation *graph, Id id, const Conv2dDescriptor &desc, const ITensorDescPack<OpTensorContent> &tensors, ConvolutionMethod method)
        : OperatorContent(graph, id, tensors), desc(desc), forced_method(method), forced_method_enabled(true)
    {
    }
    ~Conv2dContent()                     = default;
    Conv2dContent(const Conv2dContent &) = default;
    Conv2dContent &operator=(const Conv2dContent &) = default;
    Conv2dContent(Conv2dContent &&)                 = default;
    Conv2dContent &operator=(Conv2dContent &&) = default;
    bool operator==(const OperatorContent &other) const override;
    OperatorComplexity complexity() const override
    {
        return OperatorComplexity::Complex;
    }
    void set_method(ConvolutionMethod method)
    {
        forced_method_enabled = true;
        forced_method         = method;
    }

    Status translate(ClKernelGraph &kernel_graph) const override;
    /** Replicate heuristics of @ref ClConv2d::get_convolution_method(), except that non-supported data types and data layouts are removed from the heuristics
     *
     * @param src
     * @param weights
     * @param dst
     * @param conv2d_desc
     * @param gpu_target
     * @return ConvolutionMethod
     */
    static ConvolutionMethod select_conv_method(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *dst, const Conv2dDescriptor &conv2d_desc, const GPUTarget gpu_target);

    Conv2dDescriptor  desc{};
    ConvolutionMethod forced_method{ ConvolutionMethod::GEMM_CONV2D };
    bool              forced_method_enabled{ false };

private:
    Status translate_direct_conv2d(ClKernelGraph &kernel_graph) const;
};

class AddContent : public OperatorContent
{
public:
    AddContent() = default;
    AddContent(const OperatorGraph::Implementation *graph, Id id, const AddDescriptor &desc, const ITensorDescPack<OpTensorContent> &tensors)
        : OperatorContent(graph, id, tensors), desc(desc)
    {
    }
    ~AddContent()                  = default;
    AddContent(const AddContent &) = default;
    AddContent &operator=(const AddContent &) = default;
    AddContent(AddContent &&)                 = default;
    AddContent &operator=(AddContent &&) = default;
    bool operator==(const OperatorContent &other) const override;
    OperatorComplexity complexity() const override
    {
        return OperatorComplexity::Simple;
    }
    Status translate(ClKernelGraph &kernel_graph) const override;

private:
    AddDescriptor desc{};
};

struct OperatorGraph::Implementation
{
public:
    template <typename ContentT, typename... Args>
    void add_node(Operator::Id id, Args &&... args)
    {
        operators[id] = utils::memory::make_deep_unique<OperatorContent, ContentT>(this, id, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void add_tensor(OpTensor::Id id, Args &&... args)
    {
        tensors[id] = utils::memory::make_deep_unique<OpTensorContent, OpTensorContent>(id, std::forward<Args>(args)...);
    }

    using Dependency  = DependencyGraph;
    using OperatorMap = std::map<Operator::Id, utils::memory::deep_unique_ptr<OperatorContent>>;
    using OpTensorMap = std::map<OpTensor::Id, utils::memory::deep_unique_ptr<OpTensorContent>>;

    Implementation()  = default;
    ~Implementation() = default;

    friend bool operator==(const OperatorGraph::Implementation &graph0, const OperatorGraph::Implementation &graph1)
    {
        return graph0.graph == graph1.graph && graph0.operators == graph1.operators && graph0.tensors == graph1.tensors;
    }

    Dependency  graph{};
    OperatorMap operators{};
    OpTensorMap tensors{};
    Status      status{};
};

std::vector<const OperatorContent *> traverse(const OperatorGraph::Implementation &graph);

std::vector<OperatorContent *> traverse(OperatorGraph::Implementation &graph);

Status translate(ClKernelGraph &kernel_graph, const OperatorGraph::Implementation &op_graph);

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute

#endif //ARM_COMPUTE_EXPERIMENTAL_DYNAMICFUSION_OPERATORGRAPHIMPL
#endif /* ENABLE_EXPERIMENTAL_DYNAMIC_FUSION */