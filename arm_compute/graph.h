/*
 * Copyright (c) 2018-2019 Arm Limited.
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
#ifndef ARM_COMPUTE_GRAPH_H
#define ARM_COMPUTE_GRAPH_H

// IR
#include "arm_compute/graph/Edge.h"
#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/GraphBuilder.h"
#include "arm_compute/graph/IDeviceBackend.h"
#include "arm_compute/graph/IGraphMutator.h"
#include "arm_compute/graph/IGraphPrinter.h"
#include "arm_compute/graph/INode.h"
#include "arm_compute/graph/INodeVisitor.h"
#include "arm_compute/graph/Logger.h"
#include "arm_compute/graph/Tensor.h"
#include "arm_compute/graph/TensorDescriptor.h"
#include "arm_compute/graph/TypePrinter.h"
#include "arm_compute/graph/Types.h"

// Nodes
#include "arm_compute/graph/nodes/Nodes.h"

// Algorithms, Mutators, Printers
#include "arm_compute/graph/algorithms/Algorithms.h"
#include "arm_compute/graph/mutators/GraphMutators.h"
#include "arm_compute/graph/printers/Printers.h"

// Frontend
#include "arm_compute/graph/frontend/IStreamOperators.h"
#include "arm_compute/graph/frontend/Layers.h"
#include "arm_compute/graph/frontend/Stream.h"
#include "arm_compute/graph/frontend/SubStream.h"
#include "arm_compute/graph/frontend/Types.h"

#endif /* ARM_COMPUTE_GRAPH_H */
