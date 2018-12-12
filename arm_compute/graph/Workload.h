/*
 * Copyright (c) 2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_GRAPH_WORKLOAD_H__
#define __ARM_COMPUTE_GRAPH_WORKLOAD_H__

#include "arm_compute/graph/GraphContext.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryGroup.h"

#include <functional>
#include <memory>
#include <vector>

namespace arm_compute
{
namespace graph
{
// Forward declarations
class ITensorHandle;
class INode;
class Tensor;
class Graph;

struct ExecutionTask;

void execute_task(ExecutionTask &task);

/** Task executor */
class TaskExecutor final
{
private:
    /** Default constructor **/
    TaskExecutor();

public:
    /** Task executor accessor
     *
     * @return Task executor instance
     */
    static TaskExecutor &get();
    /** Function that is responsible for executing tasks */
    std::function<decltype(execute_task)> execute_function;
};

/** Execution task
 *
 * Contains all the information required to execute a given task
 */
struct ExecutionTask
{
    ExecutionTask(std::unique_ptr<arm_compute::IFunction> &&f, INode *n)
        : task(std::move(f)), node(n)
    {
    }
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    ExecutionTask(const ExecutionTask &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    ExecutionTask &operator=(const ExecutionTask &) = delete;
    /** Default Move Constructor. */
    ExecutionTask(ExecutionTask &&) noexcept = default;
    /** Default move assignment operator */
    ExecutionTask &operator=(ExecutionTask &&) noexcept = default;
    /** Default destructor */
    ~ExecutionTask() = default;
    // TODO (geopin01) : Support vector of functions?
    std::unique_ptr<arm_compute::IFunction> task = {}; /**< Task to execute */
    INode                                  *node = {}; /**< Node bound to this workload */

    /** Function operator */
    void operator()();

    /** Prepare execution task */
    void prepare();
};

/** Execution workload */
struct ExecutionWorkload
{
    std::vector<Tensor *>      inputs  = {};          /**< Input handles */
    std::vector<Tensor *>      outputs = {};          /**< Output handles */
    std::vector<ExecutionTask> tasks   = {};          /**< Execution workload */
    Graph                     *graph   = { nullptr }; /**< Graph bound to the workload */
    GraphContext              *ctx     = { nullptr }; /**< Graph execution context */
};
} // namespace graph
} // namespace arm_compute
#endif /* __ARM_COMPUTE_GRAPH_WORKLOAD_H__ */
