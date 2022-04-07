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

/* Depthwise kernel drivers commonly require a per-thread blob of working space
 * in which to store parameters required by the depthwise implementations. The
 * composition of this working space varies with the driver, kernel, and data
 * types -- but the tasks of requesting sufficient space, allocating buffer
 * space, and performing initialisation of the working space are common.
 *
 * The classes in this file consist of a number of working space "Elements"
 * (which are logical units of functionality) and a Workspace type which allows
 * for compile time composition of elements into a single working space type.
 *
 * Creating a workspace
 * ====================
 *
 * A new workspace type can be created by combining Elements as an argument to
 * the Workspace class. For instance:
 *
 *   Workspace<
 *     depthwise_depthfirst::InputArrayElement<float>,
 *     InputBufferElement<float>,
 *     OutputArrayElement<float>
 *   >
 *
 * Creates a new Workspace consisting of the given elements. The workspace type
 * contained within this class (`Workspace<...>::WorkspaceType`) is equivalent to:
 *
 *   struct WorkspaceType
 *   {
 *     const float **inptr_array;  // From InputArrayElement<float>
 *     float *input_buffer;  // From InputBufferElement<float>
 *     float **outptr_array;  // From OutputArrayElement<float>
 *     float *output_buffer;  // From OutputArrayElement<float>
 *   };
 *
 * Calling `Workspace<...>::get_sizeof_workspace(...)` will return the amount
 * of space required to store the above struct and the elements contained
 * within it. Once this space has been allocated, the workspace can be
 * initialised by calling `Workspace<...>::initialise` with a pointer to the
 * buffer and the same arguments. This will place a struct of type
 * `Workspace<...>::WorkspaceType` at the start of the buffer, and share the
 * remaining space between the specified elements. As this is all done at
 * compile time, later code can access elements from the `WorkspaceType` by
 * name.
 *
 * Writing a new element
 * =====================
 *
 * Each Element must provide:
 *  - A struct called "Workspace" containing the variables contained within
 *    this portion of the workspace.
 *  - A static method called `get_element_size` which returns the amount of
 *    buffer space required by this element of the workspace (NOT including the
 *    size of the Workspace struct). For example, an element which stores a
 *    vector of pointers will return the amount of space required top store the
 *    vector.
 *  - A static method called `initialise` which accepts a pointer to a struct
 *    which will be composed of the Element's `Workspace` struct (along with
 *    other elements), a pointer to the start of the buffer allocated for this
 *    portion of the workspace, and arguments to be used to initialise the
 *    workspace. The Element should consume as much of the buffer as it
 *    requires, initialise the Workspace, and then return the pointer to the
 *    next free byte of the buffer.
 *
 * See the below elements for an example of how this should work.
 */

#pragma once

#include "depthwise.hpp"
#include "depthfirst_driver.hpp"
#include "src/core/NEON/kernels/arm_gemm/utils.hpp"

namespace arm_conv {
namespace depthwise {
namespace {  // anonymous because we expect this to appear in several compilation units

/* Arguments to use to size and initialise a workspace.
 */
template <class StratType, class OutputStage=Nothing>
struct WorkspaceArgs
{
  const StratType *strategy;
  const DepthwiseArgs &depthwise_args;
  const OutputStage &output_stage;

  WorkspaceArgs(const StratType *strat, const DepthwiseArgs &dwargs, const OutputStage &os = {})
  : strategy(strat), depthwise_args(dwargs), output_stage(os)
  {
  }
};


/* Sometimes we use templated structs to fill in workspace types, the Empty
 * element can be useful for when a blank element is required for some sets of
 * parameters.
 */
struct EmptyElement
{
  struct Workspace {};

  template <class StratType, class OutputStage>
  static size_t get_element_size(const WorkspaceArgs<StratType, OutputStage> &) { return 0; }

  template <class WorkspaceType, class StratType, class OutputStage>
  static void *initialise(WorkspaceType *, void *buffer, const WorkspaceArgs<StratType, OutputStage> &)
  {
    return buffer;
  }
};


/* Store fused activations for a kernel.
 *
 * Activations are set based on the DepthwiseArgs.
 */
template <typename T, class OutputStage=Nothing>
class ActivationsElement
{
  public:
  struct Workspace
  {
    T activation_min, activation_max;
  };

  template <typename StratType>
  static size_t get_element_size(const WorkspaceArgs<StratType, OutputStage> &)
  {
    return 0;
  }

  template <class WorkspaceType, class StratType>
  static void *initialise(WorkspaceType *ws, void *buffer, const WorkspaceArgs<StratType, OutputStage> &args)
  {
    ws->activation_min = static_cast<T>(-std::numeric_limits<float>::infinity());
    ws->activation_max = static_cast<T>(std::numeric_limits<float>::infinity());

    switch (args.depthwise_args.activation.type)
    {
      case arm_gemm::Activation::Type::BoundedReLU:
        ws->activation_max = static_cast<T>(args.depthwise_args.activation.param1);
        // Fall through
      case arm_gemm::Activation::Type::ReLU:
        ws->activation_min = static_cast<T>(0);
        break;
      default:
        break;
    }

    return buffer;
  }
};

/* Activation clamps are contained within `arm_gemm::Requantize32`, so if the
 * output stage is one of these we substitute in an empty workspace element.
 */
template <typename T>
class ActivationsElement<T, arm_gemm::Requantize32> : public EmptyElement
{
};


/* Get the value with which to fill an input buffer. This defaults to `0`
 * (which we return as a `char` since it gets used by `memset`).
 */
template <typename OutputStage>
char get_input_buffer_fill_value(const OutputStage &)
{
  return 0;
}

/* In the case of kernels operating on quantized data, we need to fill the
 * input buffer with the zero offset of the input tensor.
 */
template <> char get_input_buffer_fill_value(const arm_gemm::Requantize32 &qp) __attribute__ ((unused));
template <> char get_input_buffer_fill_value(const arm_gemm::Requantize32 &qp)
{
  return qp.a_offset;
}


/* Container for a vector of padding values which can be safely consumed by the
 * depthwise kernel. The padding values are initialised to either `0` or the
 * zero offset of the input tensor (if quantized).
 */
template <typename T>
class InputBufferElement
{
  public:
  struct Workspace
  {
    T *input_buffer;
  };

  template <typename StratType, typename OutputStage>
  static size_t get_element_size(const WorkspaceArgs<StratType, OutputStage> &args)
  {
    return sizeof(T) * args.depthwise_args.input_channels;
  }

  template <class WorkspaceType, typename StratType, typename OutputStage>
  static void *initialise(WorkspaceType *ws, void *buffer, const WorkspaceArgs<StratType, OutputStage> &args)
  {
    ws->input_buffer = reinterpret_cast<T*>(buffer);
    memset(ws->input_buffer, get_input_buffer_fill_value(args.output_stage), get_element_size(args));
    return reinterpret_cast<char *>(buffer) + get_element_size(args);
  }
};


/* Container for an array of output pointers, and a buffer which can be used as
 * a destination for unnecessary writes.
 */
template <typename T>
class OutputArrayElement
{
  public:
  struct Workspace
  {
    T **outptr_array;
    T *output_buffer;
  };

  template <typename OutputStage>
  static size_t get_element_size(const WorkspaceArgs<IDepthfirstStrategy, OutputStage> &args)
  {
    return sizeof_outptr_array(args) + sizeof_output_buffer(args);
  }

  template <class WorkspaceType, typename OutputStage>
  static void *initialise(WorkspaceType *ws, void *buffer, const WorkspaceArgs<IDepthfirstStrategy, OutputStage> &args)
  {
    char *buffer_bytes = reinterpret_cast<char *>(buffer);

    ws->outptr_array = reinterpret_cast<T **>(buffer_bytes);
    buffer_bytes += sizeof_outptr_array(args);

    ws->output_buffer = reinterpret_cast<T *>(buffer_bytes);
    buffer_bytes += sizeof_output_buffer(args);

    return buffer_bytes;
  }

  protected:
  template <typename OutputStage>
  static size_t sizeof_outptr_array(const WorkspaceArgs<IDepthfirstStrategy, OutputStage> &args)
  {
    return sizeof(T **) * args.strategy->get_output_rows() * args.strategy->get_output_cols();
  }

  template <typename OutputStage>
  static size_t sizeof_output_buffer(const WorkspaceArgs<IDepthfirstStrategy, OutputStage> &args)
  {
    return sizeof(T) * args.depthwise_args.input_channels * args.depthwise_args.channel_multiplier;
  }
};


/* Container for requantization parameters.
 *
 * This removes the distinction between per-layer and per-channel
 * requantization parameters by providing a vector of requantization parameters
 * regardless of whether per-layer or per-channel is selected.
 */
class RequantizationParametersElement
{
  public:
  struct Workspace
  {
    const int32_t *bias, *requant_muls, *requant_shifts;
  };

  template <typename StratType>
  static size_t get_element_size(const WorkspaceArgs<StratType, arm_gemm::Requantize32> &args)
  {
    return sizeof_bias(args) + sizeof_requant_muls(args) + sizeof_requant_shifts(args);
  }

  template <typename WorkspaceType, typename StratType>
  static void *initialise(WorkspaceType *ws, void *buffer, const WorkspaceArgs<StratType, arm_gemm::Requantize32> &args)
  {
    const auto n_output_channels = args.depthwise_args.input_channels * args.depthwise_args.channel_multiplier;
    char *buffer_bytes = reinterpret_cast<char *>(buffer);

    ws->bias = args.output_stage.bias;
    ws->requant_muls = args.output_stage.per_channel_muls;
    ws->requant_shifts = args.output_stage.per_channel_right_shifts;

    if (ws->bias == nullptr)
    {
      ws->bias = reinterpret_cast<const int32_t *>(buffer_bytes);
      memset(buffer_bytes, 0, sizeof_bias(args));
      buffer_bytes += sizeof_bias(args);
    }

    if (ws->requant_muls == nullptr)
    {
      ws->requant_muls = reinterpret_cast<const int32_t *>(buffer_bytes);
      auto muls = reinterpret_cast<int32_t *>(buffer_bytes);
      buffer_bytes += sizeof_requant_muls(args);

      for (auto n = 0u; n < n_output_channels; n++)
      {
        muls[n] = args.output_stage.per_layer_mul;
      }
    }

    if (ws->requant_shifts == nullptr)
    {
      ws->requant_shifts = reinterpret_cast<int32_t *>(buffer_bytes);
      auto shifts = reinterpret_cast<int32_t *>(buffer_bytes);
      buffer_bytes += sizeof_requant_shifts(args);

      for (auto n = 0u; n < n_output_channels; n++)
      {
        shifts[n] = args.output_stage.per_layer_right_shift;
      }
    }

    return buffer_bytes;
  }

  protected:
  template <typename StratType>
  static size_t sizeof_bias(const WorkspaceArgs<StratType, arm_gemm::Requantize32> &args)
  {
    return args.output_stage.bias != nullptr ?
      0 : sizeof(int32_t) * args.depthwise_args.channel_multiplier * args.depthwise_args.input_channels;
  }

  template <typename StratType>
  static size_t sizeof_requant_muls(const WorkspaceArgs<StratType, arm_gemm::Requantize32> &args)
  {
    return args.output_stage.per_channel_muls != nullptr ?
      0 : sizeof(int32_t) * args.depthwise_args.channel_multiplier * args.depthwise_args.input_channels;
  }

  template <typename StratType>
  static size_t sizeof_requant_shifts(const WorkspaceArgs<StratType, arm_gemm::Requantize32> &args)
  {
    return args.output_stage.per_channel_right_shifts != nullptr ?
      0 : sizeof(int32_t) * args.depthwise_args.channel_multiplier * args.depthwise_args.input_channels;
  }
};


template <typename ...Elements>
class Workspace;

template <typename Element, typename ...Elements>
class Workspace<Element, Elements...>
{
  public:
  struct WorkspaceType : Element::Workspace, Workspace<Elements...>::WorkspaceType
  {
  };

  template <class S, class T>
  static void initialise(void *buffer, const WorkspaceArgs<S, T> &args)
  {
    // Allocate sufficient space for the struct, then initialise each of the
    // elements in turn.
    auto ws = reinterpret_cast<WorkspaceType *>(buffer);
    initialise_elements(ws, ws + 1, args);
  }

  template <class S, class T=Nothing>
  static size_t get_sizeof_workspace(const WorkspaceArgs<S, T> &args)
  {
    return sizeof(WorkspaceType) + get_element_sizes(args);
  }

  template <class S, class T>
  static inline size_t get_element_sizes(const WorkspaceArgs<S, T> &args)
  {
    return Element::get_element_size(args) + Workspace<Elements...>::get_element_sizes(args);
  }

  template <class WorkspaceType, class S, class T>
  static void initialise_elements(WorkspaceType *ws, void *buffer, const WorkspaceArgs<S, T> &args)
  {
    buffer = Element::initialise(ws, buffer, args);  // Get the next buffer
    Workspace<Elements...>::initialise_elements(ws, buffer, args);
  }
};

template <>
class Workspace<>
{
  public:
  struct WorkspaceType
  {
  };

  template <class S, class T>
  static inline size_t get_element_sizes(const WorkspaceArgs<S, T> &)
  {
    return 0;
  }

  template <class WorkspaceType, class S, class T>
  static void initialise_elements(WorkspaceType *, void *, const WorkspaceArgs<S, T> &)
  {
  }
};

}  // namespace {anonymous}
}  // namespace depthwise
}  // namespace arm_conv
