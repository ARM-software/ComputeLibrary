/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_WINDOW_ITERATOR_H__
#define __ARM_COMPUTE_WINDOW_ITERATOR_H__
#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Window.h"

//FIXME: Delete the "PRINTF" before the release. In the meantime it's probably going to be useful to debug
//#define PRINTF printf
#define PRINTF(...)

namespace arm_compute
{
/** Convert an offset in window steps into absolute coordinates.
 *
 * @param[in] w      Window @p offset is related to.
 * @param[in] offset Offset inside the window expressed in number of window steps.
 *
 * @return Absolute coordinates.
 */
inline Coordinates convert_window_coord_to_position(const Window &w, const Coordinates &offset)
{
    Coordinates position;
    for(unsigned int i = 0; i < Coordinates::num_max_dimensions; ++i)
    {
        position.set(i, w[i].start() + offset[i] * w[i].step());
    }
    return position;
}

/** Tensor accessors to make it easier to interface with arm_gemm */
template <typename T>
class TensorAccessor
{
public:
    /** Constructor:
     *
     * @param[in] tensor Source tensor, must be allocated.
     */
    TensorAccessor(const ITensor &tensor)
        : _first(tensor.ptr_to_element(Coordinates())), _strides(tensor.info()->strides_in_bytes())
    {
    }
    /** Get the stride of the dimension dim expressed in number of Ts.
     *
     * @param[in] dim Dimension of the wanted stride.
     *
     * @return Stride in number of Ts.
     */
    inline size_t stride(size_t dim) const
    {
        ARM_COMPUTE_ERROR_ON(_strides[dim] % sizeof(T) != 0);
        return _strides[dim] / sizeof(T);
    }

    /** Manually set the stride of a dimension
     *
     * @param[in] dim  Dimension of the stride to set.
     * @param[in] size Value to set the stride to (in bytes).
     */
    void set_stride(size_t dim, size_t size)
    {
        _strides[dim] = size;
    }

    /** Manually set the strides
     *
     * @param[in] strides Strides to set
     */
    void set_strides(const Strides &strides)
    {
        _strides = strides;
    }

    /** Returns a pointer to the element at coordinates (x,y,z,w)
     *
     * @param[in] x X coordinates
     * @param[in] y (optional) Y coordinates
     * @param[in] z (optional) Z coordinates
     * @param[in] w (optional) W coordinates
     */
    inline T *get_ptr(unsigned int x, unsigned int y = 0, unsigned int z = 0, unsigned int w = 0)
    {
        return reinterpret_cast<T *>(_first + x * _strides[0] + y * _strides[1] + z * _strides[2] + w * _strides[3]);
    }

    /** Returns a pointer to the element at coordinates (x,y,z,w)
     *
     * @param[in] x X coordinates
     * @param[in] y (optional) Y coordinates
     * @param[in] z (optional) Z coordinates
     * @param[in] w (optional) W coordinates
     */
    inline T *operator()(unsigned int x, unsigned int y = 0, unsigned int z = 0, unsigned int w = 0)
    {
        return get_ptr(x, y, z, w);
    }

    /** Returns a pointer to the first element of the tensor
     *
     * @return Pointer to the first element.
     */
    inline T *first_element()
    {
        return reinterpret_cast<T *>(_first);
    }

    /** Returns a pointer to the first element of the tensor
     *
     * @return Pointer to the first element.
     */
    inline T *operator()()
    {
        return first_element();
    }

private:
    uint8_t *_first;   /**< Pointer to the first element of the tensor.*/
    Strides  _strides; /**< Strides in bytes of the tensor */
};

/** Iterate over a portion of a Window */
template <typename L>
class WindowIterator
{
public:
    /** Construct a WindowIterator object
     *
     * @param[in] w               Window to use for the iteration
     * @param[in] start           Where to start iterating from (In Window coordinates)
     * @param[in] end             Where to stop iterating (In Window coordinates).
     * @param[in] lambda_function Lambda function to call for every iteration between start and end. (It will be called last for end - 1)
     */
    WindowIterator(const Window &w, const Coordinates &start, const Coordinates &end, L &&lambda_function)
        : _lambda_function(std::move(lambda_function)),
          _position(convert_window_coord_to_position(w, start)),
          _end(convert_window_coord_to_position(w, end)),
          _w(w)
    {
    }
    /** Iterate over the lowest 3 dimensions of the window.
     *
     * @param[in] on_new_row_size Callback to be called before lambda_function every time the width of the row processed changes.
     */
    template <typename M>
    void iterate_3D(M &&on_new_row_size)
    {
        while(_end.z() != _position.z())
        {
            PRINTF("New slice %d\n", _position.z());
            iterate_2D_internal(on_new_row_size, _w.x().end() - _w.x().step(), _w.y().end() - _w.y().step());
            _position[2] += _w.z().step();
            _position[1] = _w.y().start();
            _position[0] = _w.x().start();
        }
        // Left over:
        PRINTF("Left over slice\n");
        iterate_2D(on_new_row_size);
    }

    /** Iterate over the lowest 2 dimensions of the window.
     *
     * @param[in] on_new_row_size Callback to be called before lambda_function every time the width of the row processed changes.
     */
    template <typename M>
    void iterate_2D(M &&on_new_row_size)
    {
        iterate_2D_internal(on_new_row_size, _end.x(), _end.y());
    }

    /** Change the step used for the iteration.
     *
     * @note Does not affect the start and end points.
     *
     * @param[in] dim  Dimension to change
     * @param[in] step New step to use for the given dimension.
     */
    inline void set_step(size_t dim, int step)
    {
        _w.set_dimension_step(dim, step);
    }

    /** Returns the coordinates in absolute coordinates of the end position
         *
         * @return End position coordinates.
         */
    const Coordinates &end_position() const
    {
        return _end;
    }

private:
    template <typename M>
    void iterate_2D_internal(M &&on_new_row_size, int end_x, int end_y)
    {
        //Is there more than one row to process ?
        if(end_y == _position.y())
        {
            // Single row:
            PRINTF("Partial row only\n");
            // Both start and end belong to the same row:
            iterate_over_dim0(end_x + _w.x().step(), on_new_row_size);
        }
        else
        {
            // Do we start from the beginning of the row ?
            if(_w.x().start() != _position.x())
            {
                //Start in the middle of a row: process left-over X
                PRINTF("Partial row first\n");
                iterate_over_dim0(_w.x().end(), on_new_row_size);
                _position[1] += _w.y().step();
            }

            //Middle rows
            bool no_leftover = end_x + _w.x().step() == _w.x().end();
            if(no_leftover)
            {
                PRINTF("no left over\n");
                //Switch to full row size:
                on_new_row_size(_w[0].start(), _w.x().end());
                // Shouldn't be possible to reach that point and not have at least one entire row to process
                ARM_COMPUTE_ERROR_ON(_w.y().end() == _position.y());
                // No leftover: all the rows lefts to process are full width:
                iterate_over_dim1(end_y + _w.y().step());
            }
            else
            {
                PRINTF("with left over\n");
                // Are there full rows to process ?
                if(_position[1] != end_y)
                {
                    PRINTF("full rows\n");
                    //Switch to full row size:
                    on_new_row_size(_w[0].start(), _w.x().end());
                    iterate_over_dim1(end_y);
                }

                PRINTF("Final leftover\n");
                //Leftover end x
                _position[0] = _w.x().start();
                iterate_over_dim0(end_x + _w.x().step(), on_new_row_size);
            }
        }
    }

    /** Process full rows below 'end'
     *
     * @param[in] end Y position to stop at.
     */
    void iterate_over_dim1(int end)
    {
        for(; _position[1] != end; _position[1] += _w[1].step())
        {
            _position[0] = _w[0].start();
            iterate_over_dim0(_w[0].end());
        }
    }

    /** Process elements of a given row up to 'end'
     *
     * @param[in] end             X position to stop at.
     * @param[in] on_new_row_size Callback to call before starting iterating
     */
    template <typename M>
    void iterate_over_dim0(int end, M &&on_new_row_size)
    {
        on_new_row_size(_position.x(), end);
        iterate_over_dim0(end);
    }

    /** Process elements of a given row up to 'end'
     *
     * @param[in] end X position to stop at.
     */
    void iterate_over_dim0(int end)
    {
        PRINTF("X [%d, %d, %d]\n", _position.x(), end, _w[0].step());
        // Both start and end belong to the same row:
        ARM_COMPUTE_ERROR_ON(_position[0] > end);
        for(; _position.x() < end; _position[0] += _w[0].step())
        {
            _lambda_function(_position);
        }
    }

    L           _lambda_function; /**< Function to call for each iteration */
    Coordinates _position;        /**< Absolute coordinates of the current position */
    Coordinates _end;             /**< Absolute coordinates of the point after the last iteration */
    Window      _w;               /**< Window to iterate over */
};

/** Create a WindowIterator object
 *
 * @param[in] w               Window to use for the iteration
 * @param[in] start           Where to start iterating from (In Window coordinates)
 * @param[in] end             Where to stop iterating (In Window coordinates).
 * @param[in] lambda_function Lambda function to call for every iteration between start and end. (It will be called last for end - 1)
 *
 * @return A WindowIterator object.
 */
template <typename L>
WindowIterator<L> create_window_iterator(const Window &w, const Coordinates &start, const Coordinates &end, L &&lambda_function)
{
    return WindowIterator<L>(w, start, end, std::move(lambda_function));
}
}
#endif /*__ARM_COMPUTE_WINDOW_ITERATOR_H__*/
