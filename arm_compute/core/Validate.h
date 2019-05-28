/*
 * Copyright (c) 2016-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_VALIDATE_H__
#define __ARM_COMPUTE_VALIDATE_H__

#include "arm_compute/core/Error.h"
#include "arm_compute/core/HOGInfo.h"
#include "arm_compute/core/IKernel.h"
#include "arm_compute/core/IMultiHOG.h"
#include "arm_compute/core/IMultiImage.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/MultiImageInfo.h"
#include "arm_compute/core/Window.h"

#include <algorithm>

namespace arm_compute
{
namespace detail
{
/* Check whether two dimension objects differ.
 *
 * @param[in] dim1      First object to be compared.
 * @param[in] dim2      Second object to be compared.
 * @param[in] upper_dim The dimension from which to check.
 *
 * @return Return true if the two objects are different.
 */
template <typename T>
inline bool have_different_dimensions(const Dimensions<T> &dim1, const Dimensions<T> &dim2, unsigned int upper_dim)
{
    for(unsigned int i = upper_dim; i < arm_compute::Dimensions<T>::num_max_dimensions; ++i)
    {
        if(dim1[i] != dim2[i])
        {
            return true;
        }
    }

    return false;
}

/** Function to compare two @ref Dimensions objects and throw an error on mismatch.
 *
 * @param[in] dim      Object to compare against.
 * @param[in] function Function in which the error occurred.
 * @param[in] file     File in which the error occurred.
 * @param[in] line     Line in which the error occurred.
 */
template <typename T>
class compare_dimension
{
public:
    /** Construct a comparison function.
     *
     * @param[in] dim      Dimensions to compare.
     * @param[in] function Source function. Used for error reporting.
     * @param[in] file     Source code file. Used for error reporting.
     * @param[in] line     Source code line. Used for error reporting.
     */
    compare_dimension(const Dimensions<T> &dim, const char *function, const char *file, int line)
        : _dim{ dim }, _function{ function }, _file{ file }, _line{ line }
    {
    }

    /** Compare the given object against the stored one.
     *
     * @param[in] dim To be compared object.
     *
     * @return a status.
     */
    arm_compute::Status operator()(const Dimensions<T> &dim)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_LOC_MSG(have_different_dimensions(_dim, dim, 0), _function, _file, _line,
                                            "Objects have different dimensions");
        return arm_compute::Status{};
    }

private:
    const Dimensions<T> &_dim;
    const char *const    _function;
    const char *const    _file;
    const int            _line;
};

template <typename F>
inline arm_compute::Status for_each_error(F &&)
{
    return arm_compute::Status{};
}

template <typename F, typename T, typename... Ts>
inline arm_compute::Status for_each_error(F &&func, T &&arg, Ts &&... args)
{
    ARM_COMPUTE_RETURN_ON_ERROR(func(arg));
    ARM_COMPUTE_RETURN_ON_ERROR(for_each_error(func, args...));
    return arm_compute::Status{};
}

/** Get the info for a tensor, dummy struct */
template <typename T>
struct get_tensor_info_t;
/** Get the info for a tensor */
template <>
struct get_tensor_info_t<ITensorInfo *>
{
    /** Get the info for a tensor.
     *
     * @param[in] tensor Tensor.
     *
     * @return tensor info.
     */
    ITensorInfo *operator()(const ITensor *tensor)
    {
        return tensor->info();
    }
};
} // namespace detail

/** Create an error if one of the pointers is a nullptr.
 *
 * @param[in] function Function in which the error occurred.
 * @param[in] file     Name of the file where the error occurred.
 * @param[in] line     Line on which the error occurred.
 * @param[in] pointers Pointers to check against nullptr.
 *
 * @return Status
 */
template <typename... Ts>
inline arm_compute::Status error_on_nullptr(const char *function, const char *file, const int line, Ts &&... pointers)
{
    const std::array<const void *, sizeof...(Ts)> pointers_array{ { std::forward<Ts>(pointers)... } };
    bool has_nullptr = std::any_of(pointers_array.begin(), pointers_array.end(), [&](const void *ptr)
    {
        return (ptr == nullptr);
    });
    ARM_COMPUTE_RETURN_ERROR_ON_LOC_MSG(has_nullptr, function, file, line, "Nullptr object!");
    return arm_compute::Status{};
}
#define ARM_COMPUTE_ERROR_ON_NULLPTR(...) \
    ARM_COMPUTE_ERROR_THROW_ON(::arm_compute::error_on_nullptr(__func__, __FILE__, __LINE__, __VA_ARGS__))
#define ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(...) \
    ARM_COMPUTE_RETURN_ON_ERROR(::arm_compute::error_on_nullptr(__func__, __FILE__, __LINE__, __VA_ARGS__))

/** Return an error if the passed window is invalid.
 *
 * The subwindow is invalid if:
 * - It is not a valid window.
 * - Its dimensions don't match the full window's ones
 * - The step for each of its dimension is not identical to the corresponding one of the full window.
 *
 * @param[in] function Function in which the error occurred.
 * @param[in] file     Name of the file where the error occurred.
 * @param[in] line     Line on which the error occurred.
 * @param[in] full     Full size window
 * @param[in] win      Window to validate.
 *
 * @return Status
 */
arm_compute::Status error_on_mismatching_windows(const char *function, const char *file, const int line,
                                                 const Window &full, const Window &win);
#define ARM_COMPUTE_ERROR_ON_MISMATCHING_WINDOWS(f, w) \
    ARM_COMPUTE_ERROR_THROW_ON(::arm_compute::error_on_mismatching_windows(__func__, __FILE__, __LINE__, f, w))
#define ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_WINDOWS(f, w) \
    ARM_COMPUTE_RETURN_ON_ERROR(::arm_compute::error_on_mismatching_windows(__func__, __FILE__, __LINE__, f, w))

/** Return an error if the passed subwindow is invalid.
 *
 * The subwindow is invalid if:
 * - It is not a valid window.
 * - It is not fully contained inside the full window
 * - The step for each of its dimension is not identical to the corresponding one of the full window.
 *
 * @param[in] function Function in which the error occurred.
 * @param[in] file     Name of the file where the error occurred.
 * @param[in] line     Line on which the error occurred.
 * @param[in] full     Full size window
 * @param[in] sub      Sub-window to validate.
 *
 * @return Status
 */
arm_compute::Status error_on_invalid_subwindow(const char *function, const char *file, const int line,
                                               const Window &full, const Window &sub);
#define ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(f, s) \
    ARM_COMPUTE_ERROR_THROW_ON(::arm_compute::error_on_invalid_subwindow(__func__, __FILE__, __LINE__, f, s))
#define ARM_COMPUTE_RETURN_ERROR_ON_INVALID_SUBWINDOW(f, s) \
    ARM_COMPUTE_RETURN_ON_ERROR(::arm_compute::error_on_invalid_subwindow(__func__, __FILE__, __LINE__, f, s))

/** Return an error if the window can't be collapsed at the given dimension.
 *
 * The window cannot be collapsed if the given dimension not equal to the full window's dimension or not start from 0.
 *
 * @param[in] function Function in which the error occurred.
 * @param[in] file     Name of the file where the error occurred.
 * @param[in] line     Line on which the error occurred.
 * @param[in] full     Full size window
 * @param[in] window   Window to be collapsed.
 * @param[in] dim      Dimension need to be checked.
 *
 * @return Status
 */
arm_compute::Status error_on_window_not_collapsable_at_dimension(const char *function, const char *file, const int line,
                                                                 const Window &full, const Window &window, const int dim);
#define ARM_COMPUTE_ERROR_ON_WINDOW_NOT_COLLAPSABLE_AT_DIMENSION(f, w, d) \
    ARM_COMPUTE_ERROR_THROW_ON(::arm_compute::error_on_window_not_collapsable_at_dimension(__func__, __FILE__, __LINE__, f, w, d))
#define ARM_COMPUTE_RETURN_ERROR_ON_WINDOW_NOT_COLLAPSABLE_AT_DIMENSION(f, w, d) \
    ARM_COMPUTE_RETURN_ON_ERROR(::arm_compute::error_on_window_not_collapsable_at_dimension(__func__, __FILE__, __LINE__, f, w, d))

/** Return an error if the passed coordinates have too many dimensions.
 *
 * The coordinates have too many dimensions if any of the dimensions greater or equal to max_dim is different from 0.
 *
 * @param[in] function Function in which the error occurred.
 * @param[in] file     Name of the file where the error occurred.
 * @param[in] line     Line on which the error occurred.
 * @param[in] pos      Coordinates to validate
 * @param[in] max_dim  Maximum number of dimensions allowed.
 *
 * @return Status
 */
arm_compute::Status error_on_coordinates_dimensions_gte(const char *function, const char *file, const int line,
                                                        const Coordinates &pos, unsigned int max_dim);
#define ARM_COMPUTE_ERROR_ON_COORDINATES_DIMENSIONS_GTE(p, md) \
    ARM_COMPUTE_ERROR_THROW_ON(::arm_compute::error_on_coordinates_dimensions_gte(__func__, __FILE__, __LINE__, p, md))
#define ARM_COMPUTE_RETURN_ERROR_ON_COORDINATES_DIMENSIONS_GTE(p, md) \
    ARM_COMPUTE_RETURN_ON_ERROR(::arm_compute::error_on_coordinates_dimensions_gte(__func__, __FILE__, __LINE__, p, md))

/** Return an error if the passed window has too many dimensions.
 *
 * The window has too many dimensions if any of the dimension greater or equal to max_dim is different from 0.
 *
 * @param[in] function Function in which the error occurred.
 * @param[in] file     Name of the file where the error occurred.
 * @param[in] line     Line on which the error occurred.
 * @param[in] win      Window to validate
 * @param[in] max_dim  Maximum number of dimensions allowed.
 *
 * @return Status
 */
arm_compute::Status error_on_window_dimensions_gte(const char *function, const char *file, const int line,
                                                   const Window &win, unsigned int max_dim);
#define ARM_COMPUTE_ERROR_ON_WINDOW_DIMENSIONS_GTE(w, md) \
    ARM_COMPUTE_ERROR_THROW_ON(::arm_compute::error_on_window_dimensions_gte(__func__, __FILE__, __LINE__, w, md))
#define ARM_COMPUTE_RETURN_ERROR_ON_WINDOW_DIMENSIONS_GTE(w, md) \
    ARM_COMPUTE_RETURN_ON_ERROR(::arm_compute::error_on_window_dimensions_gte(__func__, __FILE__, __LINE__, w, md))

/** Return an error if the passed dimension objects differ.
 *
 * @param[in] function Function in which the error occurred.
 * @param[in] file     Name of the file where the error occurred.
 * @param[in] line     Line on which the error occurred.
 * @param[in] dim1     The first object to be compared.
 * @param[in] dim2     The second object to be compared.
 * @param[in] dims     (Optional) Further allowed objects.
 *
 * @return Status
 */
template <typename T, typename... Ts>
arm_compute::Status error_on_mismatching_dimensions(const char *function, const char *file, int line,
                                                    const Dimensions<T> &dim1, const Dimensions<T> &dim2, Ts &&... dims)
{
    ARM_COMPUTE_RETURN_ON_ERROR(detail::for_each_error(detail::compare_dimension<T>(dim1, function, file, line), dim2, std::forward<Ts>(dims)...));
    return arm_compute::Status{};
}
#define ARM_COMPUTE_ERROR_ON_MISMATCHING_DIMENSIONS(...) \
    ARM_COMPUTE_ERROR_THROW_ON(::arm_compute::error_on_mismatching_dimensions(__func__, __FILE__, __LINE__, __VA_ARGS__))
#define ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(...) \
    ARM_COMPUTE_RETURN_ON_ERROR(::arm_compute::error_on_mismatching_dimensions(__func__, __FILE__, __LINE__, __VA_ARGS__))

/** Return an error if the passed tensor objects are not even.
 *
 * @param[in] function Function in which the error occurred.
 * @param[in] file     Name of the file where the error occurred.
 * @param[in] line     Line on which the error occurred.
 * @param[in] format   Format to check if odd shape is allowed
 * @param[in] tensor1  The first object to be compared for odd shape.
 * @param[in] tensors  (Optional) Further allowed objects.
 *
 * @return Status
 */
template <typename... Ts>
arm_compute::Status error_on_tensors_not_even(const char *function, const char *file, int line,
                                              const Format &format, const ITensor *tensor1, Ts... tensors)
{
    ARM_COMPUTE_RETURN_ERROR_ON_LOC(tensor1 == nullptr, function, file, line);
    ARM_COMPUTE_RETURN_ON_ERROR(::arm_compute::error_on_nullptr(function, file, line, std::forward<Ts>(tensors)...));
    const std::array < const ITensor *, 1 + sizeof...(Ts) > tensors_info_array{ { tensor1, std::forward<Ts>(tensors)... } };
    ARM_COMPUTE_RETURN_ERROR_ON_LOC_MSG(std::any_of(tensors_info_array.cbegin(), tensors_info_array.cend(), [&](const ITensor * tensor)
    {
        const TensorShape correct_shape = adjust_odd_shape(tensor->info()->tensor_shape(), format);
        return detail::have_different_dimensions(tensor->info()->tensor_shape(), correct_shape, 2);
    }),
    function, file, line, "Tensor shape has odd dimensions");
    return arm_compute::Status{};
}

#define ARM_COMPUTE_ERROR_ON_TENSORS_NOT_EVEN(...) \
    ARM_COMPUTE_ERROR_THROW_ON(::arm_compute::error_on_tensors_not_even(__func__, __FILE__, __LINE__, __VA_ARGS__))
#define ARM_COMPUTE_RETURN_ERROR_ON_TENSORS_NOT_EVEN(...) \
    ARM_COMPUTE_RETURN_ON_ERROR(::arm_compute::error_on_tensors_not_even(__func__, __FILE__, __LINE__, __VA_ARGS__))

/** Return an error if the passed tensor objects are not sub-sampled.
 *
 * @param[in] function Function in which the error occurred.
 * @param[in] file     Name of the file where the error occurred.
 * @param[in] line     Line on which the error occurred.
 * @param[in] format   Format to check if sub-sampling allowed.
 * @param[in] shape    The tensor shape to calculate sub-sampling from.
 * @param[in] tensor1  The first object to be compared.
 * @param[in] tensors  (Optional) Further allowed objects.
 *
 * @return Status
 */
template <typename... Ts>
arm_compute::Status error_on_tensors_not_subsampled(const char *function, const char *file, int line,
                                                    const Format &format, const TensorShape &shape, const ITensor *tensor1, Ts... tensors)
{
    ARM_COMPUTE_RETURN_ERROR_ON_LOC(tensor1 == nullptr, function, file, line);
    ARM_COMPUTE_RETURN_ON_ERROR(::arm_compute::error_on_nullptr(function, file, line, std::forward<Ts>(tensors)...));
    const TensorShape sub2_shape = calculate_subsampled_shape(shape, format);
    const std::array < const ITensor *, 1 + sizeof...(Ts) > tensors_info_array{ { tensor1, std::forward<Ts>(tensors)... } };
    ARM_COMPUTE_RETURN_ERROR_ON_LOC_MSG(std::any_of(tensors_info_array.cbegin(), tensors_info_array.cend(), [&](const ITensor * tensor)
    {
        return detail::have_different_dimensions(tensor->info()->tensor_shape(), sub2_shape, 2);
    }),
    function, file, line, "Tensor shape has mismatch dimensions for sub-sampling");
    return arm_compute::Status{};
}

#define ARM_COMPUTE_ERROR_ON_TENSORS_NOT_SUBSAMPLED(...) \
    ARM_COMPUTE_ERROR_THROW_ON(::arm_compute::error_on_tensors_not_subsampled(__func__, __FILE__, __LINE__, __VA_ARGS__))
#define ARM_COMPUTE_RETURN_ERROR_ON_TENSORS_NOT_SUBSAMPLED(...) \
    ARM_COMPUTE_RETURN_ON_ERROR(::arm_compute::error_on_tensors_not_subsampled(__func__, __FILE__, __LINE__, __VA_ARGS__))

/** Return an error if the passed two tensor infos have different shapes from the given dimension
 *
 * @param[in] function      Function in which the error occurred.
 * @param[in] file          Name of the file where the error occurred.
 * @param[in] line          Line on which the error occurred.
 * @param[in] tensor_info_1 The first tensor info to be compared.
 * @param[in] tensor_info_2 The second tensor info to be compared.
 * @param[in] tensor_infos  (Optional) Further allowed tensor infos.
 *
 * @return Status
 */
template <typename... Ts>
inline arm_compute::Status error_on_mismatching_shapes(const char *function, const char *file, const int line,
                                                       const ITensorInfo *tensor_info_1, const ITensorInfo *tensor_info_2, Ts... tensor_infos)
{
    return error_on_mismatching_shapes(function, file, line, 0U, tensor_info_1, tensor_info_2, std::forward<Ts>(tensor_infos)...);
}
/** Return an error if the passed two tensors have different shapes from the given dimension
 *
 * @param[in] function Function in which the error occurred.
 * @param[in] file     Name of the file where the error occurred.
 * @param[in] line     Line on which the error occurred.
 * @param[in] tensor_1 The first tensor to be compared.
 * @param[in] tensor_2 The second tensor to be compared.
 * @param[in] tensors  (Optional) Further allowed tensors.
 *
 * @return Status
 */
template <typename... Ts>
inline arm_compute::Status error_on_mismatching_shapes(const char *function, const char *file, const int line,
                                                       const ITensor *tensor_1, const ITensor *tensor_2, Ts... tensors)
{
    return error_on_mismatching_shapes(function, file, line, 0U, tensor_1, tensor_2, std::forward<Ts>(tensors)...);
}
/** Return an error if the passed two tensors have different shapes from the given dimension
 *
 * @param[in] function      Function in which the error occurred.
 * @param[in] file          Name of the file where the error occurred.
 * @param[in] line          Line on which the error occurred.
 * @param[in] upper_dim     The dimension from which to check.
 * @param[in] tensor_info_1 The first tensor info to be compared.
 * @param[in] tensor_info_2 The second tensor info to be compared.
 * @param[in] tensor_infos  (Optional) Further allowed tensor infos.
 *
 * @return Status
 */
template <typename... Ts>
inline arm_compute::Status error_on_mismatching_shapes(const char *function, const char *file, const int line,
                                                       unsigned int upper_dim, const ITensorInfo *tensor_info_1, const ITensorInfo *tensor_info_2, Ts... tensor_infos)
{
    ARM_COMPUTE_RETURN_ERROR_ON_LOC(tensor_info_1 == nullptr, function, file, line);
    ARM_COMPUTE_RETURN_ERROR_ON_LOC(tensor_info_2 == nullptr, function, file, line);
    ARM_COMPUTE_RETURN_ON_ERROR(::arm_compute::error_on_nullptr(function, file, line, std::forward<Ts>(tensor_infos)...));

    const std::array < const ITensorInfo *, 2 + sizeof...(Ts) > tensors_info_array{ { tensor_info_1, tensor_info_2, std::forward<Ts>(tensor_infos)... } };
    ARM_COMPUTE_RETURN_ERROR_ON_LOC_MSG(std::any_of(std::next(tensors_info_array.cbegin()), tensors_info_array.cend(), [&](const ITensorInfo * tensor_info)
    {
        return detail::have_different_dimensions((*tensors_info_array.cbegin())->tensor_shape(), tensor_info->tensor_shape(), upper_dim);
    }),
    function, file, line, "Tensors have different shapes");
    return arm_compute::Status{};
}
/** Return an error if the passed two tensors have different shapes from the given dimension
 *
 * @param[in] function  Function in which the error occurred.
 * @param[in] file      Name of the file where the error occurred.
 * @param[in] line      Line on which the error occurred.
 * @param[in] upper_dim The dimension from which to check.
 * @param[in] tensor_1  The first tensor to be compared.
 * @param[in] tensor_2  The second tensor to be compared.
 * @param[in] tensors   (Optional) Further allowed tensors.
 *
 * @return Status
 */
template <typename... Ts>
inline arm_compute::Status error_on_mismatching_shapes(const char *function, const char *file, const int line,
                                                       unsigned int upper_dim, const ITensor *tensor_1, const ITensor *tensor_2, Ts... tensors)
{
    ARM_COMPUTE_RETURN_ERROR_ON_LOC(tensor_1 == nullptr, function, file, line);
    ARM_COMPUTE_RETURN_ERROR_ON_LOC(tensor_2 == nullptr, function, file, line);
    ARM_COMPUTE_RETURN_ON_ERROR(::arm_compute::error_on_nullptr(function, file, line, std::forward<Ts>(tensors)...));
    ARM_COMPUTE_RETURN_ON_ERROR(::arm_compute::error_on_mismatching_shapes(function, file, line, upper_dim, tensor_1->info(), tensor_2->info(),
                                                                           detail::get_tensor_info_t<ITensorInfo *>()(tensors)...));
    return arm_compute::Status{};
}
#define ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(...) \
    ARM_COMPUTE_ERROR_THROW_ON(::arm_compute::error_on_mismatching_shapes(__func__, __FILE__, __LINE__, __VA_ARGS__))
#define ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(...) \
    ARM_COMPUTE_RETURN_ON_ERROR(::arm_compute::error_on_mismatching_shapes(__func__, __FILE__, __LINE__, __VA_ARGS__))

/** Return an error if the passed tensor infos have different data layouts
 *
 * @param[in] function     Function in which the error occurred.
 * @param[in] file         Name of the file where the error occurred.
 * @param[in] line         Line on which the error occurred.
 * @param[in] tensor_info  The first tensor info to be compared.
 * @param[in] tensor_infos (Optional) Further allowed tensor infos.
 *
 * @return Status
 */
template <typename... Ts>
inline arm_compute::Status error_on_mismatching_data_layouts(const char *function, const char *file, const int line,
                                                             const ITensorInfo *tensor_info, Ts... tensor_infos)
{
    ARM_COMPUTE_RETURN_ERROR_ON_LOC(tensor_info == nullptr, function, file, line);
    ARM_COMPUTE_RETURN_ON_ERROR(::arm_compute::error_on_nullptr(function, file, line, std::forward<Ts>(tensor_infos)...));

    DataLayout &&tensor_data_layout = tensor_info->data_layout();
    const std::array<const ITensorInfo *, sizeof...(Ts)> tensors_infos_array{ { std::forward<Ts>(tensor_infos)... } };
    ARM_COMPUTE_RETURN_ERROR_ON_LOC_MSG(std::any_of(tensors_infos_array.begin(), tensors_infos_array.end(), [&](const ITensorInfo * tensor_info_obj)
    {
        return tensor_info_obj->data_layout() != tensor_data_layout;
    }),
    function, file, line, "Tensors have different data layouts");
    return arm_compute::Status{};
}
/** Return an error if the passed tensors have different data layouts
 *
 * @param[in] function Function in which the error occurred.
 * @param[in] file     Name of the file where the error occurred.
 * @param[in] line     Line on which the error occurred.
 * @param[in] tensor   The first tensor to be compared.
 * @param[in] tensors  (Optional) Further allowed tensors.
 *
 * @return Status
 */
template <typename... Ts>
inline arm_compute::Status error_on_mismatching_data_layouts(const char *function, const char *file, const int line,
                                                             const ITensor *tensor, Ts... tensors)
{
    ARM_COMPUTE_RETURN_ERROR_ON_LOC(tensor == nullptr, function, file, line);
    ARM_COMPUTE_RETURN_ON_ERROR(::arm_compute::error_on_nullptr(function, file, line, std::forward<Ts>(tensors)...));
    ARM_COMPUTE_RETURN_ON_ERROR(::arm_compute::error_on_mismatching_data_layouts(function, file, line, tensor->info(),
                                                                                 detail::get_tensor_info_t<ITensorInfo *>()(tensors)...));
    return arm_compute::Status{};
}
#define ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_LAYOUT(...) \
    ARM_COMPUTE_ERROR_THROW_ON(::arm_compute::error_on_mismatching_data_layouts(__func__, __FILE__, __LINE__, __VA_ARGS__))
#define ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(...) \
    ARM_COMPUTE_RETURN_ON_ERROR(::arm_compute::error_on_mismatching_data_layouts(__func__, __FILE__, __LINE__, __VA_ARGS__))

/** Return an error if the passed two tensor infos have different data types
 *
 * @param[in] function     Function in which the error occurred.
 * @param[in] file         Name of the file where the error occurred.
 * @param[in] line         Line on which the error occurred.
 * @param[in] tensor_info  The first tensor info to be compared.
 * @param[in] tensor_infos (Optional) Further allowed tensor infos.
 *
 * @return Status
 */
template <typename... Ts>
inline arm_compute::Status error_on_mismatching_data_types(const char *function, const char *file, const int line,
                                                           const ITensorInfo *tensor_info, Ts... tensor_infos)
{
    ARM_COMPUTE_RETURN_ERROR_ON_LOC(tensor_info == nullptr, function, file, line);
    ARM_COMPUTE_RETURN_ON_ERROR(::arm_compute::error_on_nullptr(function, file, line, std::forward<Ts>(tensor_infos)...));

    DataType &&tensor_data_type = tensor_info->data_type();
    const std::array<const ITensorInfo *, sizeof...(Ts)> tensors_infos_array{ { std::forward<Ts>(tensor_infos)... } };
    ARM_COMPUTE_RETURN_ERROR_ON_LOC_MSG(std::any_of(tensors_infos_array.begin(), tensors_infos_array.end(), [&](const ITensorInfo * tensor_info_obj)
    {
        return tensor_info_obj->data_type() != tensor_data_type;
    }),
    function, file, line, "Tensors have different data types");
    return arm_compute::Status{};
}
/** Return an error if the passed two tensors have different data types
 *
 * @param[in] function Function in which the error occurred.
 * @param[in] file     Name of the file where the error occurred.
 * @param[in] line     Line on which the error occurred.
 * @param[in] tensor   The first tensor to be compared.
 * @param[in] tensors  (Optional) Further allowed tensors.
 *
 * @return Status
 */
template <typename... Ts>
inline arm_compute::Status error_on_mismatching_data_types(const char *function, const char *file, const int line,
                                                           const ITensor *tensor, Ts... tensors)
{
    ARM_COMPUTE_RETURN_ERROR_ON_LOC(tensor == nullptr, function, file, line);
    ARM_COMPUTE_RETURN_ON_ERROR(::arm_compute::error_on_nullptr(function, file, line, std::forward<Ts>(tensors)...));
    ARM_COMPUTE_RETURN_ON_ERROR(::arm_compute::error_on_mismatching_data_types(function, file, line, tensor->info(),
                                                                               detail::get_tensor_info_t<ITensorInfo *>()(tensors)...));
    return arm_compute::Status{};
}
#define ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(...) \
    ARM_COMPUTE_ERROR_THROW_ON(::arm_compute::error_on_mismatching_data_types(__func__, __FILE__, __LINE__, __VA_ARGS__))
#define ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(...) \
    ARM_COMPUTE_RETURN_ON_ERROR(::arm_compute::error_on_mismatching_data_types(__func__, __FILE__, __LINE__, __VA_ARGS__))

/** Return an error if the passed tensor infos have different asymmetric quantized data types or different quantization info
 *
 * @note: If the first tensor info doesn't have asymmetric quantized data type, the function returns without throwing an error
 *
 * @param[in] function      Function in which the error occurred.
 * @param[in] file          Name of the file where the error occurred.
 * @param[in] line          Line on which the error occurred.
 * @param[in] tensor_info_1 The first tensor info to be compared.
 * @param[in] tensor_info_2 The second tensor info to be compared.
 * @param[in] tensor_infos  (Optional) Further allowed tensor infos.
 *
 * @return Status
 */
template <typename... Ts>
inline arm_compute::Status error_on_mismatching_quantization_info(const char *function, const char *file, const int line,
                                                                  const ITensorInfo *tensor_info_1, const ITensorInfo *tensor_info_2, Ts... tensor_infos)
{
    DataType             &&first_data_type         = tensor_info_1->data_type();
    const QuantizationInfo first_quantization_info = tensor_info_1->quantization_info();

    if(!is_data_type_quantized(first_data_type))
    {
        return arm_compute::Status{};
    }

    const std::array < const ITensorInfo *, 1 + sizeof...(Ts) > tensor_infos_array{ { tensor_info_2, std::forward<Ts>(tensor_infos)... } };
    ARM_COMPUTE_RETURN_ERROR_ON_LOC_MSG(std::any_of(tensor_infos_array.begin(), tensor_infos_array.end(), [&](const ITensorInfo * tensor_info)
    {
        return tensor_info->data_type() != first_data_type;
    }),
    function, file, line, "Tensors have different asymmetric quantized data types");
    ARM_COMPUTE_RETURN_ERROR_ON_LOC_MSG(std::any_of(tensor_infos_array.begin(), tensor_infos_array.end(), [&](const ITensorInfo * tensor_info)
    {
        return tensor_info->quantization_info() != first_quantization_info;
    }),
    function, file, line, "Tensors have different quantization information");

    return arm_compute::Status{};
}
/** Return an error if the passed tensor have different asymmetric quantized data types or different quantization info
 *
 * @note: If the first tensor doesn't have asymmetric quantized data type, the function returns without throwing an error
 *
 * @param[in] function Function in which the error occurred.
 * @param[in] file     Name of the file where the error occurred.
 * @param[in] line     Line on which the error occurred.
 * @param[in] tensor_1 The first tensor to be compared.
 * @param[in] tensor_2 The second tensor to be compared.
 * @param[in] tensors  (Optional) Further allowed tensors.
 *
 * @return Status
 */
template <typename... Ts>
inline arm_compute::Status error_on_mismatching_quantization_info(const char *function, const char *file, const int line,
                                                                  const ITensor *tensor_1, const ITensor *tensor_2, Ts... tensors)
{
    ARM_COMPUTE_RETURN_ON_ERROR(::arm_compute::error_on_mismatching_quantization_info(function, file, line, tensor_1->info(), tensor_2->info(),
                                                                                      detail::get_tensor_info_t<ITensorInfo *>()(tensors)...));
    return arm_compute::Status{};
}
#define ARM_COMPUTE_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(...) \
    ARM_COMPUTE_ERROR_THROW_ON(::arm_compute::error_on_mismatching_quantization_info(__func__, __FILE__, __LINE__, __VA_ARGS__))
#define ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(...) \
    ARM_COMPUTE_RETURN_ON_ERROR(::arm_compute::error_on_mismatching_quantization_info(__func__, __FILE__, __LINE__, __VA_ARGS__))

/** Throw an error if the format of the passed tensor/multi-image does not match any of the formats provided.
 *
 * @param[in] function Function in which the error occurred.
 * @param[in] file     Name of the file where the error occurred.
 * @param[in] line     Line on which the error occurred.
 * @param[in] object   Tensor/multi-image to validate.
 * @param[in] format   First format allowed.
 * @param[in] formats  (Optional) Further allowed formats.
 */
template <typename T, typename F, typename... Fs>
void error_on_format_not_in(const char *function, const char *file, const int line,
                            const T *object, F &&format, Fs &&... formats)
{
    ARM_COMPUTE_ERROR_ON_LOC(object == nullptr, function, file, line);

    Format &&object_format = object->info()->format();
    ARM_COMPUTE_UNUSED(object_format);

    ARM_COMPUTE_ERROR_ON_LOC(object_format == Format::UNKNOWN, function, file, line);

    const std::array<F, sizeof...(Fs)> formats_array{ { std::forward<Fs>(formats)... } };
    ARM_COMPUTE_UNUSED(formats_array);

    ARM_COMPUTE_ERROR_ON_LOC_MSG(object_format != format && std::none_of(formats_array.begin(), formats_array.end(), [&](const F & f)
    {
        return f == object_format;
    }),
    function, file, line, "Format %s not supported by this kernel", string_from_format(object_format).c_str());
}
#define ARM_COMPUTE_ERROR_ON_FORMAT_NOT_IN(t, ...) ::arm_compute::error_on_format_not_in(__func__, __FILE__, __LINE__, t, __VA_ARGS__)

/** Return an error if the data type of the passed tensor info does not match any of the data types provided.
 *
 * @param[in] function    Function in which the error occurred.
 * @param[in] file        Name of the file where the error occurred.
 * @param[in] line        Line on which the error occurred.
 * @param[in] tensor_info Tensor info to validate.
 * @param[in] dt          First data type allowed.
 * @param[in] dts         (Optional) Further allowed data types.
 *
 * @return Status
 */
template <typename T, typename... Ts>
inline arm_compute::Status error_on_data_type_not_in(const char *function, const char *file, const int line,
                                                     const ITensorInfo *tensor_info, T &&dt, Ts &&... dts)
{
    ARM_COMPUTE_RETURN_ERROR_ON_LOC(tensor_info == nullptr, function, file, line);

    const DataType &tensor_dt = tensor_info->data_type(); //NOLINT
    ARM_COMPUTE_RETURN_ERROR_ON_LOC(tensor_dt == DataType::UNKNOWN, function, file, line);

    const std::array<T, sizeof...(Ts)> dts_array{ { std::forward<Ts>(dts)... } };
    ARM_COMPUTE_RETURN_ERROR_ON_LOC_MSG(tensor_dt != dt && std::none_of(dts_array.begin(), dts_array.end(), [&](const T & d)
    {
        return d == tensor_dt;
    }),
    function, file, line, "ITensor data type %s not supported by this kernel", string_from_data_type(tensor_dt).c_str());
    return arm_compute::Status{};
}
/** Return an error if the data type of the passed tensor does not match any of the data types provided.
 *
 * @param[in] function Function in which the error occurred.
 * @param[in] file     Name of the file where the error occurred.
 * @param[in] line     Line on which the error occurred.
 * @param[in] tensor   Tensor to validate.
 * @param[in] dt       First data type allowed.
 * @param[in] dts      (Optional) Further allowed data types.
 *
 * @return Status
 */
template <typename T, typename... Ts>
inline arm_compute::Status error_on_data_type_not_in(const char *function, const char *file, const int line,
                                                     const ITensor *tensor, T &&dt, Ts &&... dts)
{
    ARM_COMPUTE_RETURN_ERROR_ON_LOC(tensor == nullptr, function, file, line);
    ARM_COMPUTE_RETURN_ON_ERROR(::arm_compute::error_on_data_type_not_in(function, file, line, tensor->info(), std::forward<T>(dt), std::forward<Ts>(dts)...));
    return arm_compute::Status{};
}
#define ARM_COMPUTE_ERROR_ON_DATA_TYPE_NOT_IN(t, ...) \
    ARM_COMPUTE_ERROR_THROW_ON(::arm_compute::error_on_data_type_not_in(__func__, __FILE__, __LINE__, t, __VA_ARGS__))
#define ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(t, ...) \
    ARM_COMPUTE_RETURN_ON_ERROR(::arm_compute::error_on_data_type_not_in(__func__, __FILE__, __LINE__, t, __VA_ARGS__))

/** Return an error if the data layout of the passed tensor info does not match any of the data layouts provided.
 *
 * @param[in] function    Function in which the error occurred.
 * @param[in] file        Name of the file where the error occurred.
 * @param[in] line        Line on which the error occurred.
 * @param[in] tensor_info Tensor info to validate.
 * @param[in] dl          First data layout allowed.
 * @param[in] dls         (Optional) Further allowed data layouts.
 *
 * @return Status
 */
template <typename T, typename... Ts>
inline arm_compute::Status error_on_data_layout_not_in(const char *function, const char *file, const int line,
                                                       const ITensorInfo *tensor_info, T &&dl, Ts &&... dls)
{
    ARM_COMPUTE_RETURN_ERROR_ON_LOC(tensor_info == nullptr, function, file, line);

    const DataLayout &tensor_dl = tensor_info->data_layout(); //NOLINT
    ARM_COMPUTE_RETURN_ERROR_ON_LOC(tensor_dl == DataLayout::UNKNOWN, function, file, line);

    const std::array<T, sizeof...(Ts)> dls_array{ { std::forward<Ts>(dls)... } };
    ARM_COMPUTE_RETURN_ERROR_ON_LOC_MSG(tensor_dl != dl && std::none_of(dls_array.begin(), dls_array.end(), [&](const T & l)
    {
        return l == tensor_dl;
    }),
    function, file, line, "ITensor data layout %s not supported by this kernel", string_from_data_layout(tensor_dl).c_str());
    return arm_compute::Status{};
}
/** Return an error if the data layout of the passed tensor does not match any of the data layout provided.
 *
 * @param[in] function Function in which the error occurred.
 * @param[in] file     Name of the file where the error occurred.
 * @param[in] line     Line on which the error occurred.
 * @param[in] tensor   Tensor to validate.
 * @param[in] dl       First data layout allowed.
 * @param[in] dls      (Optional) Further allowed data layouts.
 *
 * @return Status
 */
template <typename T, typename... Ts>
inline arm_compute::Status error_on_data_layout_not_in(const char *function, const char *file, const int line,
                                                       const ITensor *tensor, T &&dl, Ts &&... dls)
{
    ARM_COMPUTE_RETURN_ERROR_ON_LOC(tensor == nullptr, function, file, line);
    ARM_COMPUTE_RETURN_ON_ERROR(::arm_compute::error_on_data_layout_not_in(function, file, line, tensor->info(), std::forward<T>(dl), std::forward<Ts>(dls)...));
    return arm_compute::Status{};
}
#define ARM_COMPUTE_ERROR_ON_DATA_LAYOUT_NOT_IN(t, ...) \
    ARM_COMPUTE_ERROR_THROW_ON(::arm_compute::error_on_data_layout_not_in(__func__, __FILE__, __LINE__, t, __VA_ARGS__))
#define ARM_COMPUTE_RETURN_ERROR_ON_DATA_LAYOUT_NOT_IN(t, ...) \
    ARM_COMPUTE_RETURN_ON_ERROR(::arm_compute::error_on_data_layout_not_in(__func__, __FILE__, __LINE__, t, __VA_ARGS__))

/** Return an error if the data type or the number of channels of the passed tensor info does not match any of the data types and number of channels provided.
 *
 * @param[in] function     Function in which the error occurred.
 * @param[in] file         Name of the file where the error occurred.
 * @param[in] line         Line on which the error occurred.
 * @param[in] tensor_info  Tensor info to validate.
 * @param[in] num_channels Number of channels to check
 * @param[in] dt           First data type allowed.
 * @param[in] dts          (Optional) Further allowed data types.
 *
 * @return Status
 */
template <typename T, typename... Ts>
inline arm_compute::Status error_on_data_type_channel_not_in(const char *function, const char *file, const int line,
                                                             const ITensorInfo *tensor_info, size_t num_channels, T &&dt, Ts &&... dts)
{
    ARM_COMPUTE_RETURN_ON_ERROR(::arm_compute::error_on_data_type_not_in(function, file, line, tensor_info, std::forward<T>(dt), std::forward<Ts>(dts)...));
    const size_t tensor_nc = tensor_info->num_channels();
    ARM_COMPUTE_RETURN_ERROR_ON_LOC_MSG(tensor_nc != num_channels, function, file, line, "Number of channels %d. Required number of channels %d", tensor_nc, num_channels);
    return arm_compute::Status{};
}
/** Return an error if the data type or the number of channels of the passed tensor does not match any of the data types and number of channels provided.
 *
 * @param[in] function     Function in which the error occurred.
 * @param[in] file         Name of the file where the error occurred.
 * @param[in] line         Line on which the error occurred.
 * @param[in] tensor       Tensor to validate.
 * @param[in] num_channels Number of channels to check
 * @param[in] dt           First data type allowed.
 * @param[in] dts          (Optional) Further allowed data types.
 *
 * @return Status
 */
template <typename T, typename... Ts>
inline arm_compute::Status error_on_data_type_channel_not_in(const char *function, const char *file, const int line,
                                                             const ITensor *tensor, size_t num_channels, T &&dt, Ts &&... dts)
{
    ARM_COMPUTE_RETURN_ERROR_ON_LOC(tensor == nullptr, function, file, line);
    ARM_COMPUTE_RETURN_ON_ERROR(error_on_data_type_channel_not_in(function, file, line, tensor->info(), num_channels, std::forward<T>(dt), std::forward<Ts>(dts)...));
    return arm_compute::Status{};
}
#define ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(t, c, ...) \
    ARM_COMPUTE_ERROR_THROW_ON(::arm_compute::error_on_data_type_channel_not_in(__func__, __FILE__, __LINE__, t, c, __VA_ARGS__))
#define ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(t, c, ...) \
    ARM_COMPUTE_RETURN_ON_ERROR(::arm_compute::error_on_data_type_channel_not_in(__func__, __FILE__, __LINE__, t, c, __VA_ARGS__))

/** Return an error if the data type of the passed tensor info is FP16 and FP16 extension is not supported by the device.
 *
 * @param[in] function          Function in which the error occurred.
 * @param[in] file              Name of the file where the error occurred.
 * @param[in] line              Line on which the error occurred.
 * @param[in] tensor_info       Tensor info to validate.
 * @param[in] is_fp16_supported Is fp16 supported by the device.
 *
 * @return Status
 */
inline arm_compute::Status error_on_unsupported_fp16(const char *function, const char *file, const int line,
                                                     const ITensorInfo *tensor_info, bool is_fp16_supported)
{
    ARM_COMPUTE_RETURN_ERROR_ON_LOC(tensor_info == nullptr, function, file, line);
    ARM_COMPUTE_RETURN_ERROR_ON_LOC_MSG((tensor_info->data_type() == DataType::F16 && !is_fp16_supported),
                                        function, file, line, "FP16 not supported by the device");
    return arm_compute::Status{};
}

/** Return an error if the data type of the passed tensor is FP16 and FP16 extension is not supported by the device.
 *
 * @param[in] function          Function in which the error occurred.
 * @param[in] file              Name of the file where the error occurred.
 * @param[in] line              Line on which the error occurred.
 * @param[in] tensor            Tensor to validate.
 * @param[in] is_fp16_supported Is fp16 supported by the device.
 *
 * @return Status
 */
inline arm_compute::Status error_on_unsupported_fp16(const char *function, const char *file, const int line,
                                                     const ITensor *tensor, bool is_fp16_supported)
{
    ARM_COMPUTE_RETURN_ERROR_ON_LOC(tensor == nullptr, function, file, line);
    ARM_COMPUTE_RETURN_ON_ERROR(::arm_compute::error_on_unsupported_fp16(function, file, line, tensor->info(), is_fp16_supported));
    return arm_compute::Status{};
}

/** Return an error if the tensor is not 2D.
 *
 * @param[in] function Function in which the error occurred.
 * @param[in] file     Name of the file where the error occurred.
 * @param[in] line     Line on which the error occurred.
 * @param[in] tensor   Tensor to validate.
 *
 * @return Status
 */
arm_compute::Status error_on_tensor_not_2d(const char *function, const char *file, const int line,
                                           const ITensor *tensor);

/** Return an error if the tensor info is not 2D.
 *
 * @param[in] function Function in which the error occurred.
 * @param[in] file     Name of the file where the error occurred.
 * @param[in] line     Line on which the error occurred.
 * @param[in] tensor   Tensor info to validate.
 *
 * @return Status
 */
arm_compute::Status error_on_tensor_not_2d(const char *function, const char *file, const int line,
                                           const ITensorInfo *tensor);

#define ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(t) \
    ARM_COMPUTE_ERROR_THROW_ON(::arm_compute::error_on_tensor_not_2d(__func__, __FILE__, __LINE__, t))
#define ARM_COMPUTE_RETURN_ERROR_ON_TENSOR_NOT_2D(t) \
    ARM_COMPUTE_RETURN_ON_ERROR(::arm_compute::error_on_tensor_not_2d(__func__, __FILE__, __LINE__, t))

/** Return an error if the channel is not in channels.
 *
 * @param[in] function Function in which the error occurred.
 * @param[in] file     Name of the file where the error occurred.
 * @param[in] line     Line on which the error occurred.
 * @param[in] cn       Input channel
 * @param[in] channel  First channel allowed.
 * @param[in] channels (Optional) Further allowed channels.
 *
 * @return Status
 */
template <typename T, typename... Ts>
inline arm_compute::Status error_on_channel_not_in(const char *function, const char *file, const int line,
                                                   T cn, T &&channel, Ts &&... channels)
{
    ARM_COMPUTE_RETURN_ERROR_ON_LOC(cn == Channel::UNKNOWN, function, file, line);

    const std::array<T, sizeof...(Ts)> channels_array{ { std::forward<Ts>(channels)... } };
    ARM_COMPUTE_RETURN_ERROR_ON_LOC(channel != cn && std::none_of(channels_array.begin(), channels_array.end(), [&](const T & f)
    {
        return f == cn;
    }),
    function, file, line);
    return arm_compute::Status{};
}
#define ARM_COMPUTE_ERROR_ON_CHANNEL_NOT_IN(c, ...) \
    ARM_COMPUTE_ERROR_THROW_ON(::arm_compute::error_on_channel_not_in(__func__, __FILE__, __LINE__, c, __VA_ARGS__))
#define ARM_COMPUTE_RETURN_ERROR_ON_CHANNEL_NOT_IN(c, ...) \
    ARM_COMPUTE_RETURN_ON_ERROR(::arm_compute::error_on_channel_not_in(__func__, __FILE__, __LINE__, c, __VA_ARGS__))

/** Return an error if the channel is not in format.
 *
 * @param[in] function Function in which the error occurred.
 * @param[in] file     Name of the file where the error occurred.
 * @param[in] line     Line on which the error occurred.
 * @param[in] fmt      Input channel
 * @param[in] cn       First channel allowed.
 *
 * @return Status
 */
arm_compute::Status error_on_channel_not_in_known_format(const char *function, const char *file, const int line,
                                                         Format fmt, Channel cn);
#define ARM_COMPUTE_ERROR_ON_CHANNEL_NOT_IN_KNOWN_FORMAT(f, c) \
    ARM_COMPUTE_ERROR_THROW_ON(::arm_compute::error_on_channel_not_in_known_format(__func__, __FILE__, __LINE__, f, c))
#define ARM_COMPUTE_RETURN_ERROR_ON_CHANNEL_NOT_IN_KNOWN_FORMAT(f, c) \
    ARM_COMPUTE_RETURN_ON_ERROR(::arm_compute::error_on_channel_not_in_known_format(__func__, __FILE__, __LINE__, f, c))

/** Return an error if the @ref IMultiHOG container is invalid
 *
 * An @ref IMultiHOG container is invalid if:
 *
 * -# it is a nullptr
 * -# it doesn't contain models
 * -# it doesn't have the HOG data objects with the same phase_type, normalization_type and l2_hyst_threshold (if normalization_type == L2HYS_NORM)
 *
 * @param[in] function  Function in which the error occurred.
 * @param[in] file      Name of the file where the error occurred.
 * @param[in] line      Line on which the error occurred.
 * @param[in] multi_hog IMultiHOG container to validate
 *
 * @return Status
 */
arm_compute::Status error_on_invalid_multi_hog(const char *function, const char *file, const int line,
                                               const IMultiHOG *multi_hog);
#define ARM_COMPUTE_ERROR_ON_INVALID_MULTI_HOG(m) \
    ARM_COMPUTE_ERROR_THROW_ON(::arm_compute::error_on_invalid_multi_hog(__func__, __FILE__, __LINE__, m))
#define ARM_COMPUTE_RETURN_ERROR_ON_INVALID_MULTI_HOG(m) \
    ARM_COMPUTE_RETURN_ON_ERROR(::arm_compute::error_on_invalid_multi_hog(__func__, __FILE__, __LINE__, m))

/** Return an error if the kernel is not configured.
 *
 * @param[in] function Function in which the error occurred.
 * @param[in] file     Name of the file where the error occurred.
 * @param[in] line     Line on which the error occurred.
 * @param[in] kernel   Kernel to validate.
 *
 * @return Status
 */
arm_compute::Status error_on_unconfigured_kernel(const char *function, const char *file, const int line,
                                                 const IKernel *kernel);
#define ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(k) \
    ARM_COMPUTE_ERROR_THROW_ON(::arm_compute::error_on_unconfigured_kernel(__func__, __FILE__, __LINE__, k))
#define ARM_COMPUTE_RETURN_ERROR_ON_UNCONFIGURED_KERNEL(k) \
    ARM_COMPUTE_RETURN_ON_ERROR(::arm_compute::error_on_unconfigured_kernel(__func__, __FILE__, __LINE__, k))

/** Return an error if if the coordinates and shape of the subtensor are within the parent tensor.
 *
 * @param[in] function     Function in which the error occurred.
 * @param[in] file         Name of the file where the error occurred.
 * @param[in] line         Line on which the error occurred.
 * @param[in] parent_shape Parent tensor shape
 * @param[in] coords       Coordinates inside the parent tensor where the first element of the subtensor is
 * @param[in] shape        Shape of the subtensor
 *
 * @return Status
 */
arm_compute::Status error_on_invalid_subtensor(const char *function, const char *file, const int line,
                                               const TensorShape &parent_shape, const Coordinates &coords, const TensorShape &shape);
#define ARM_COMPUTE_ERROR_ON_INVALID_SUBTENSOR(p, c, s) \
    ARM_COMPUTE_ERROR_THROW_ON(::arm_compute::error_on_invalid_subtensor(__func__, __FILE__, __LINE__, p, c, s))
#define ARM_COMPUTE_RETURN_ERROR_ON_INVALID_SUBTENSOR(p, c, s) \
    ARM_COMPUTE_RETURN_ON_ERROR(::arm_compute::error_on_invalid_subtensor(__func__, __FILE__, __LINE__, p, c, s))

/** Return an error if the valid region of a subtensor is not inside the valid region of the parent tensor.
 *
 * @param[in] function            Function in which the error occurred.
 * @param[in] file                Name of the file where the error occurred.
 * @param[in] line                Line on which the error occurred.
 * @param[in] parent_valid_region Parent valid region.
 * @param[in] valid_region        Valid region of subtensor.
 *
 * @return Status
 */
arm_compute::Status error_on_invalid_subtensor_valid_region(const char *function, const char *file, const int line,
                                                            const ValidRegion &parent_valid_region, const ValidRegion &valid_region);
#define ARM_COMPUTE_ERROR_ON_INVALID_SUBTENSOR_VALID_REGION(pv, sv) \
    ARM_COMPUTE_ERROR_THROW_ON(::arm_compute::error_on_invalid_subtensor_valid_region(__func__, __FILE__, __LINE__, pv, sv))
#define ARM_COMPUTE_RETURN_ERROR_ON_INVALID_SUBTENSOR_VALID_REGION(pv, sv) \
    ARM_COMPUTE_RETURN_ON_ERROR(::arm_compute::error_on_invalid_subtensor_valid_region(__func__, __FILE__, __LINE__, pv, sv))
}
#endif /* __ARM_COMPUTE_VALIDATE_H__*/
