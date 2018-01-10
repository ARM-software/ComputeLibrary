/*
 * Copyright (c) 2017 ARM Limited.
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
#include "winograd_gemm.hpp"
#include "batched_blocked_gemm.hpp"
using namespace winograd;

/** Get the output shape of a convolution. */
template <int kr, int kc, int itr, int itc>
template <typename TOut, typename TIn>
Tensor4DShape WinogradGEMM<kr, kc, itr, itc>::Convolution<TOut, TIn>::get_output_shape(
  const KernelShape &kernel_shape,
  const Tensor4DShape &in_shape,
  const PaddingType padding
)
{
  // TODO Accept different kernel sizes
  return Tensor4DShape {
    in_shape.n_batches,
    (padding == PADDING_SAME) ? in_shape.n_rows : in_shape.n_rows - 2,
    (padding == PADDING_SAME) ? in_shape.n_cols : in_shape.n_cols - 2,
    kernel_shape.n_output_channels,
    in_shape.ordering
  };
}

/* Get the memory required to transform the kernel.
 */
template <int kernel_rows, int kernel_cols,
          int output_tile_rows, int output_tile_cols>
template <typename TOut, typename TIn>
size_t WinogradGEMM<kernel_rows, kernel_cols, output_tile_rows, output_tile_cols>::Convolution<TOut, TIn>::get_kernel_transform_working_size(const KernelShape &shape)
{
  if (shape.ordering == HWIO)
  {
    // Kernel is already in the correct order, so no additional memory is
    // required.
    return 0;
  }
  else
  {
    // Need to re-order the kernel into HWIO form, require enough space to
    // represent the tensor.
    return sizeof(TIn) * shape.size();
  }
}

/** Get the memory required to store the kernel transformed into the
 * Winograd domain.
 */
template <int kernel_rows, int kernel_cols, int output_tile_rows, int output_tile_cols>
template <typename TOut, typename TIn>
size_t WinogradGEMM<kernel_rows, kernel_cols, output_tile_rows, output_tile_cols>::Convolution<TOut, TIn>::get_kernel_storage_size(const KernelShape &shape)
{
  return N_GEMMS * get_kernel_matrix_size(shape);
}


template <int kernel_rows, int kernel_cols, int output_tile_rows, int output_tile_cols>
template <typename TOut, typename TIn>
size_t WinogradGEMM<kernel_rows, kernel_cols, output_tile_rows, output_tile_cols>::Convolution<TOut, TIn>::get_input_storage_size(
  const KernelShape &kernel_shape,
  const Tensor4DShape &input_shape,
  const PaddingType padding
)
{
  return N_GEMMS * get_input_matrix_size(kernel_shape, input_shape, padding);
}


template <int kernel_rows, int kernel_cols, int output_tile_rows, int output_tile_cols>
template <typename TOut, typename TIn>
size_t WinogradGEMM<kernel_rows, kernel_cols, output_tile_rows, output_tile_cols>::Convolution<TOut, TIn>::get_output_storage_size(
  const KernelShape &kernel_shape,
  const Tensor4DShape &input_shape,
  const PaddingType padding
)
{
  return N_GEMMS * get_output_matrix_size(kernel_shape, input_shape, padding);
}


/** Get the memory required to apply a Winograd operator to some input.
 */
template <int kernel_rows, int kernel_cols, int output_tile_rows, int output_tile_cols>
template <typename TOut, typename TIn>
size_t WinogradGEMM<kernel_rows, kernel_cols, output_tile_rows, output_tile_cols>::Convolution<TOut, TIn>::get_working_space_size(
  const KernelShape &kernel_shape,
  const Tensor4DShape &input_shape,
  const PaddingType padding_type
)
{
  const auto output_shape = get_output_shape(kernel_shape, input_shape, padding_type);

  // Get the memory required to store the matrices
  const size_t matrix_sizes = N_GEMMS * (
    get_input_matrix_size(kernel_shape, input_shape, padding_type) +
    get_output_matrix_size(kernel_shape, input_shape, padding_type)
  );

  // Add additional space to re-order the input and output if the input tensor
  // is not in NHWC format.
  if (input_shape.ordering == NHWC)
  {
    return matrix_sizes;  // No extra spacing required
  }
  else  // NCHW, must reorder the input and output tensors
  {
    // We only need to re-order the input or output at any one time, so request
    // enough memory to do the largest of these.
    const size_t extra_memory = std::max(
      sizeof(TIn) * input_shape.size(),
      sizeof(TOut) * output_shape.size()
    );
    return matrix_sizes + extra_memory;
  }
}


/* Get the memory required by a single "input" matrix.
 */
template <int kernel_rows, int kernel_cols, int output_tile_rows, int output_tile_cols>
template <typename TOut, typename TIn>
size_t WinogradGEMM<kernel_rows, kernel_cols, output_tile_rows, output_tile_cols>::Convolution<TOut, TIn>::get_input_matrix_size(
  const KernelShape &kernel_shape,
  const Tensor4DShape &input_shape,
  const PaddingType padding_type
)
{
  return get_input_matrix_stride(kernel_shape, input_shape, padding_type) * sizeof(TIn);
}

template <int kernel_rows, int kernel_cols, int output_tile_rows, int output_tile_cols>
template <typename TOut, typename TIn>
int WinogradGEMM<kernel_rows, kernel_cols, output_tile_rows, output_tile_cols>::Convolution<TOut, TIn>::get_input_matrix_stride(
  const KernelShape &kernel_shape,
  const Tensor4DShape &input_shape,
  const PaddingType padding_type
)
{
  // Compute shape for the GEMM
  const auto output_shape = get_output_shape(kernel_shape, input_shape, padding_type);
  const int tile_rows = iceildiv(output_shape.n_rows, output_tile_rows);
  const int tile_cols = iceildiv(output_shape.n_cols, output_tile_cols);
  const int M = roundup(input_shape.n_batches * tile_rows * tile_cols, M_BLOCK);
  const int K = kernel_shape.n_input_channels;

  return M * K;
}


/* Get the memory required by a single "output" matrix.
 */
template <int kernel_rows, int kernel_cols, int output_tile_rows, int output_tile_cols>
template <typename TOut, typename TIn>
size_t WinogradGEMM<kernel_rows, kernel_cols, output_tile_rows, output_tile_cols>::Convolution<TOut, TIn>::get_output_matrix_size(
    const KernelShape &kernel_shape,
    const Tensor4DShape &input_shape,
    const PaddingType padding_type
)
{
  return get_output_matrix_stride(kernel_shape, input_shape, padding_type) * sizeof(TOut);
}


template <int kernel_rows, int kernel_cols, int output_tile_rows, int output_tile_cols>
template <typename TOut, typename TIn>
int WinogradGEMM<kernel_rows, kernel_cols, output_tile_rows, output_tile_cols>::Convolution<TOut, TIn>::get_output_matrix_stride(
    const KernelShape &kernel_shape,
    const Tensor4DShape &input_shape,
    const PaddingType padding_type
)
{
  // Compute shape for the GEMM
  const auto output_shape = get_output_shape(kernel_shape, input_shape, padding_type);
  const int tile_rows = iceildiv(output_shape.n_rows, output_tile_rows);
  const int tile_cols = iceildiv(output_shape.n_cols, output_tile_cols);
  const int M = roundup(tile_rows * tile_cols, M_BLOCK);
  const int N = roundup(kernel_shape.n_output_channels, N_BLOCK);

  return input_shape.n_batches * M * N;
}


/* Get the memory required by a single "kernel" matrix.
 */
template <int kernel_rows, int kernel_cols, int output_tile_rows, int output_tile_cols>
template <typename TOut, typename TIn>
size_t WinogradGEMM<kernel_rows, kernel_cols, output_tile_rows, output_tile_cols>::Convolution<TOut, TIn>::get_kernel_matrix_size(const KernelShape &shape)
{
  return sizeof(TIn) * get_kernel_matrix_stride(shape);
}

template <int kernel_rows, int kernel_cols, int output_tile_rows, int output_tile_cols>
template <typename TOut, typename TIn>
int WinogradGEMM<kernel_rows, kernel_cols, output_tile_rows, output_tile_cols>::Convolution<TOut, TIn>::get_kernel_matrix_stride(const KernelShape &shape)
{
  const int K = shape.n_input_channels;
  const int N = roundup(shape.n_output_channels, N_BLOCK);
  return K * N;
}


/** Create a new Winograd operator. */
template <int output_tile_rows, int output_tile_cols,
          int kernel_rows, int kernel_cols>
template <typename TOut, typename TIn>
WinogradGEMM<output_tile_rows, output_tile_cols, kernel_rows, kernel_cols>::Convolution<TOut, TIn>::Convolution(
  const KernelShape &kernel_shape,
  const Tensor4DShape &input_shape,
  const PaddingType padding,
  void *kernel_storage
) : kernel_shape(kernel_shape),  // Store the kernel shape
    kernel_matrix_row_stride(roundup(kernel_shape.n_output_channels, N_BLOCK)),
    manage_kernel_storage(kernel_storage == NULL),
    _kernel_storage(manage_kernel_storage ?
                      ALLOCATE(get_kernel_storage_size(kernel_shape)) :
                      kernel_storage),
    input_shape(input_shape),
    padding(padding),
    output_shape(get_output_shape(kernel_shape, input_shape, padding)),
    tile_rows(iceildiv(output_shape.n_rows, output_tile_rows)),
    tile_cols(iceildiv(output_shape.n_cols, output_tile_cols)),
    M(input_shape.n_batches * tile_rows * tile_cols),
    K(kernel_shape.n_input_channels),
    N(kernel_shape.n_output_channels),
    prof()
{
  // Create pointers to the kernel matrices
  const int kernel_matrix_size_bytes = get_kernel_matrix_size(kernel_shape);
  int8_t* const ks_bytes = reinterpret_cast<int8_t *>(_kernel_storage);
  for (int i = 0; i < N_GEMMS; i++) {
    kernel_matrices[i] = reinterpret_cast<TIn *>(
      ks_bytes + i*kernel_matrix_size_bytes);
  }
}


/** Create a new Winograd operator and initialise the weights. */
template <int output_tile_rows, int output_tile_cols,
          int kernel_rows, int kernel_cols>
template <typename TOut, typename TIn>
WinogradGEMM<output_tile_rows, output_tile_cols, kernel_rows, kernel_cols>::Convolution<TOut, TIn>::Convolution(
  const KernelShape &kernel_shape,
  const Tensor4DShape &input_shape,
  const PaddingType padding,
  const TIn* const kernel,
  void *kernel_storage,
  void *transform_working_space
) : Convolution(kernel_shape, input_shape, padding, kernel_storage)
{
  transform_weights(kernel, transform_working_space);
}


/** Clean up a convolution engine. */
template <int output_tile_rows, int output_tile_cols, int kernel_rows, int kernel_cols>
template <typename TOut, typename TIn>
WinogradGEMM<output_tile_rows, output_tile_cols, kernel_rows, kernel_cols>::
Convolution<TOut, TIn>::~Convolution()
{
  // If we were responsible for managing kernel storage ensure that it is
  // freed.
  if (manage_kernel_storage)
  {
    free(_kernel_storage);
  }
}


/** Transform weights into the Winograd domain and store them for later use/reuse. */
template <int output_tile_rows, int output_tile_cols, int kernel_rows, int kernel_cols>
template <typename TOut, typename TIn>
template <typename WeightsTransformT>
void WinogradGEMM<output_tile_rows, output_tile_cols, kernel_rows, kernel_cols>::
Convolution<TOut, TIn>::transform_weights(
  const TIn* const kernel,
  void *transform_working_space
)
{
  // Allocate working space if it is required
  bool allocated_working_space = false;
  if (transform_working_space == NULL &&  // If no memory has been provided
      get_kernel_transform_working_size(kernel_shape) != 0)  // And we need the space
  {
    allocated_working_space = true;
    transform_working_space = ALLOCATE(
      get_kernel_transform_working_size(kernel_shape)
    );
  }

  // The transformation methods only work on weights laid out in HWIO form, if
  // the weights are not in this form then we need to re-order them.
  const TIn *kernel_hwio = kernel;
  if (kernel_shape.ordering != HWIO)
  {
    kernel_hwio = reinterpret_cast<TIn *>(transform_working_space);

    // Re-order the weights from OIHW to HWIO
    this->prof(
      "Weight reorder",
      [&kernel, &kernel_hwio, this] () {
        reorder::ofm_ifm_h_w_to_h_w_ifm_ofm(
          kernel, const_cast<TIn *>(kernel_hwio),
          kernel_shape.n_output_channels,
          kernel_shape.n_input_channels,
          kernel_shape.n_rows,
          kernel_shape.n_cols
        );
      },
      kernel_shape.size() * sizeof(TIn),
      0,
      kernel_shape.size() * sizeof(TIn)
    );
  }

  const int kernel_matrix_size_bytes = get_kernel_matrix_size(kernel_shape);
  WeightsTransformT weights_transform(
    kernel_hwio, kernel_matrices[0],
    kernel_matrix_size_bytes / sizeof(TIn),
    kernel_matrix_row_stride,
    kernel_shape.n_output_channels,
    kernel_shape.n_input_channels
  );

  // Transform the weights into the Winograd domain
  auto kernel_prep = [&] ()
  {
    weights_transform.run(0, weights_transform.get_window());
  };

  prof(
    "Kernel Prep", kernel_prep,
    WeightsTransformT::bytes_read(kernel_shape),
    WeightsTransformT::ops_performed(kernel_shape),
    WeightsTransformT::bytes_written(kernel_shape)
  );

  // Free memory if we allocated it
  if (allocated_working_space)
  {
    free(transform_working_space);
  }
}


/** Perform a convolution. */
template <int output_tile_rows, int output_tile_cols,
          int kernel_rows, int kernel_cols>
template <typename TOut, typename TIn>
void WinogradGEMM<output_tile_rows, output_tile_cols, kernel_rows, kernel_cols>::
Convolution<TOut, TIn>::execute(
  TOut* const output,
  const TIn* const input,
  void *working_space,
  const int n_threads
)
{
  const auto padding_type = padding;
  const auto input_shape = this->input_shape;

  // Allocate working space if none has been provided
  const bool manage_working_space = (working_space == NULL);
  if (manage_working_space)
  {
    const size_t ws_size = get_working_space_size(
      kernel_shape, input_shape, padding_type
    );
    working_space = ALLOCATE(ws_size * sizeof(int8_t));
    memset(working_space, 0x00, ws_size);
  }
  int8_t* const ws_bytes = reinterpret_cast<int8_t *>(working_space);

  // Split the working space into that required for 16 input matrices and
  // output matrices.
  TIn *input_matrices[N_GEMMS];
  TOut *output_matrices[N_GEMMS];
  const int in_matrix_stride_bytes = get_input_matrix_size(kernel_shape, input_shape, padding_type);
  const int out_matrix_stride_bytes = get_output_matrix_size(kernel_shape, input_shape, padding_type);

  for (int i = 0; i < N_GEMMS; i++)
  {
    input_matrices[i] = reinterpret_cast<TIn *>(
        ws_bytes + i*in_matrix_stride_bytes);
    output_matrices[i] = reinterpret_cast<TIn *>(
        ws_bytes + N_GEMMS*in_matrix_stride_bytes + i*out_matrix_stride_bytes);
  }

  // If we need to re-order the input and output tensors then the final chunk
  // of the working space can be used for this purpose.
  // TODO  - Overlay the input reorder on top of the output matrices
  //       - Overlay the output reorder on top of the input matrices
  // Reorder the input input form if it was not provided in this ordering.
  const TIn* input_nhwc = input;
  if (input_shape.ordering == NCHW)
  {
    input_nhwc = reinterpret_cast<TIn *>(
      ws_bytes + N_GEMMS*(in_matrix_stride_bytes + out_matrix_stride_bytes)
    );

    this->prof(
      "NCHW -> NHWC",
      [input, input_shape, input_nhwc] () {
        reorder::nchw_to_nhwc(
          input, const_cast<TIn *>(input_nhwc),
          input_shape.n_batches,
          input_shape.n_channels,
          input_shape.n_rows,
          input_shape.n_cols
        );
      },
      input_shape.size(), 0, input_shape.size()
    );
  }

  // Compute shape for the GEMM
  const auto output_shape = this->output_shape;
  int M = this->M;
  int K = this->K;
  int N = this->N;

  const int in_matrix_row_stride = K;
  const int out_matrix_row_stride = kernel_matrix_row_stride;

  InputTransform<TIn> input_transform(
    input_nhwc,
    input_shape.n_batches,
    input_shape.n_rows,
    input_shape.n_cols,
    input_shape.n_channels,
    padding_type,
    input_matrices[0],
    in_matrix_stride_bytes / sizeof(TIn),
    in_matrix_row_stride
  );

  // Transform the input into the Winograd domain
  auto input_prep = [&] () {
    input_transform.run(0, input_transform.get_window());
  };
  prof(
    "Input Prep", input_prep,
    InputTransform<TIn>::bytes_read(input_shape),
    InputTransform<TIn>::ops_performed(input_shape),
    InputTransform<TIn>::bytes_written(input_shape)
  );

  // Perform the GEMMs
  const int kernel_matrix_stride_bytes = get_kernel_matrix_size(kernel_shape);
  BatchedBlockedGemm<M_BLOCK, N_BLOCK, TOut, TIn> gemms(
    N_GEMMS, M, K, N,
    in_matrix_stride_bytes / sizeof(TIn),
    in_matrix_row_stride,
    kernel_matrix_stride_bytes / sizeof(TIn),
    kernel_matrix_row_stride,
    out_matrix_stride_bytes / sizeof(TOut),
    out_matrix_row_stride,
    input_matrices[0],
    kernel_matrices[0],
    output_matrices[0]
  );
  gemms.run(0, gemms.get_window());

  // If the output tensor needs to be in NCHW form then store the NHWC output
  // tensor in temporary storage and then reorder. If the output tensor needs
  // to be in NHWC then just write straight to the output tensor.
  TOut *output_nhwc = output;
  if (input_shape.ordering == NCHW)
  {
    output_nhwc = reinterpret_cast<TOut *>(
      ws_bytes + N_GEMMS*(in_matrix_stride_bytes + out_matrix_stride_bytes)
    );
  }

  // Transform the output tensor from the Winograd domain to the spatial
  // domain.
  OutputTransform<TOut> output_transform(
    output_matrices[0],
    out_matrix_stride_bytes / sizeof(TOut),
    out_matrix_row_stride,
    output_nhwc,
    output_shape.n_batches,
    output_shape.n_rows,
    output_shape.n_cols,
    output_shape.n_channels
  );
  auto output_prep = [&] () {
    output_transform.run(0, output_transform.get_window());
  };
  prof(
    "Output Comp", output_prep,
    OutputTransform<TOut>::bytes_read(output_shape),
    OutputTransform<TOut>::ops_performed(output_shape),
    OutputTransform<TOut>::bytes_written(output_shape)
  );

  // Reorder the output tensor if it is required to be in NCHW form.
  if (input_shape.ordering == NCHW)
  {
    prof(
      "NHWC -> NCHW",
      [output_nhwc, output_shape, output] () {
        reorder::nhwc_to_nchw(
          output_nhwc, output,
          output_shape.n_batches,
          output_shape.n_rows,
          output_shape.n_cols,
          output_shape.n_channels
        );
      },
      output_shape.size(), 0, output_shape.size()
    );
  }

  // Free working space if we were responsible for allocating it
  if (manage_working_space)
  {
    free(working_space);
  }
}


/** Perform a convolution. */
template <int output_tile_rows, int output_tile_cols,
          int kernel_rows, int kernel_cols>
template <typename TOut, typename TIn>
void WinogradGEMM<output_tile_rows, output_tile_cols, kernel_rows, kernel_cols>::
Convolution<TOut, TIn>::execute(
  TOut* const output,
  const TIn* const input,
  const int n_threads
)
{
  execute(output, input, NULL, n_threads);
}


// Instantiate required implementations
template class WinogradGEMM<2, 2, 3, 3>::Convolution<float, float>;
template class WinogradGEMM<4, 4, 3, 3>::Convolution<float, float>;
