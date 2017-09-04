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
#ifndef ARM_COMPUTE_TYPES_H
#define ARM_COMPUTE_TYPES_H

/** 2D Coordinates structure */
typedef struct Coordinates2D
{
    int x; /**< The x coordinate. */
    int y; /**< The y coordinate. */
} Coordinates2D;

/* Keypoint struct */
typedef struct Keypoint
{
    int   x;               /**< The x coordinate. */
    int   y;               /**< The y coordinate. */
    float strength;        /**< The strength of the keypoint. Its definition is specific to the corner detector. */
    float scale;           /**< Initialized to 0 by corner detectors. */
    float orientation;     /**< Initialized to 0 by corner detectors. */
    int   tracking_status; /**< A zero indicates a lost point. Initialized to 1 by corner detectors. */
    float error;           /**< A tracking method specific error. Initialized to 0 by corner detectors. */
} Keypoint;

/** Detection window struct */
typedef struct DetectionWindow
{
    ushort x;         /**< Top-left x coordinate */
    ushort y;         /**< Top-left y coordinate */
    ushort width;     /**< Width of the detection window */
    ushort height;    /**< Height of the detection window */
    ushort idx_class; /**< Index of the class */
    float  score;     /**< Confidence value for the detection window */
} DetectionWindow;
#endif // ARM_COMPUTE_TYPES_H
