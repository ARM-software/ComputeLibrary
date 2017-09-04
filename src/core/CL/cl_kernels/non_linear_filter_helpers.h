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

/** Sorts element-wise two vectors.
 *
 * @param[in, out] a First vector
 * @param[in, out] b Second vector
 */
#define SORT(a, b)                  \
    {                               \
        uchar8 min_val = min(a, b); \
        uchar8 max_val = max(a, b); \
        a              = min_val;   \
        b              = max_val;   \
    }

// Sorting networks below were generated using http://pages.ripco.net/~jgamble/nw.html

/** Sorting network to sort 5 vectors of 8 elements and return their median.
 *
 * @param[in] p0 First element vector
 * @param[in] p1 Second element vector
 * @param[in] p2 Third element vector
 * @param[in] p3 Fourth element vector
 * @param[in] p4 Fifth element vector
 *
 * @return Median values for 8 elements.
 */
inline uchar8 sort5(uchar8 p0, uchar8 p1, uchar8 p2, uchar8 p3, uchar8 p4)
{
    SORT(p0, p1);
    SORT(p2, p3);
    SORT(p0, p2);
    SORT(p1, p3);
    SORT(p1, p2);
    SORT(p0, p4);
    SORT(p1, p4);
    SORT(p2, p4);

    return p2;
}

/** Sorting network to sort 9 vectors of 8 elements and return their median.
 *
 * @param[in] p0 First element vector
 * @param[in] p1 Second element vector
 * @param[in] p2 Third element vector
 * @param[in] p3 Fourth element vector
 * @param[in] p4 Fifth element vector
 * @param[in] p5 Sixth element vector
 * @param[in] p6 Seventh element vector
 * @param[in] p7 Eigth element vector
 * @param[in] p8 Ninth element vector
 *
 * @return Median values for 8 elements.
 */
inline uchar8 sort9(uchar8 p0, uchar8 p1, uchar8 p2, uchar8 p3, uchar8 p4, uchar8 p5, uchar8 p6, uchar8 p7, uchar8 p8)
{
    SORT(p1, p2);
    SORT(p4, p5);
    SORT(p7, p8);
    SORT(p0, p1);
    SORT(p3, p4);
    SORT(p6, p7);
    SORT(p1, p2);
    SORT(p4, p5);
    SORT(p7, p8);
    SORT(p0, p3);
    SORT(p5, p8);
    SORT(p4, p7);
    SORT(p3, p6);
    SORT(p1, p4);
    SORT(p2, p5);
    SORT(p4, p7);
    SORT(p4, p2);
    SORT(p6, p4);
    SORT(p4, p2);

    return p4;
}

/** Calculate the minimum of a sliding window of size 3.
 *
 * @param val Values to calculate the minimum values
 *
 * @return Minimum values of 8 elements on a sliding window of size 3.
 */
inline uchar8 row_reduce_min_3(uchar16 val)
{
    return min(val.s01234567, min(val.s12345678, val.s23456789));
}

/** Calculate the maximum of a sliding window of size 3.
 *
 * @param val Values to calculate the maximum values
 *
 * @return Maximum values of 8 elements on a sliding window of size 3.
 */
inline uchar8 row_reduce_max_3(uchar16 val)
{
    return max(val.s01234567, max(val.s12345678, val.s23456789));
}

/** Calculate the minimum of a sliding window of size 5.
 *
 * @param val Values to calculate the minimum values
 *
 * @return Minimum values of 8 elements on a sliding window of size 5.
 */
inline uchar8 row_reduce_min_5(uchar16 val)
{
    return min(val.s01234567, min(min(val.s12345678, val.s23456789), min(val.s3456789A, val.s456789AB)));
}

/** Calculate the maximum of a sliding window of size 5.
 *
 * @param val Values to calculate the maximum values
 *
 * @return Maximum values of 8 elements on a sliding window of size 5.
 */
inline uchar8 row_reduce_max_5(uchar16 val)
{
    return max(val.s01234567, max(max(val.s12345678, val.s23456789), max(val.s3456789A, val.s456789AB)));
}
