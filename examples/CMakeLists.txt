# Copyright (c) 2023 Arm Limited.
#
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

set(EXAMPLE_GRAPH_NAMES
    graph_alexnet
    graph_deepspeech_v0_4_1
    graph_edsr
    graph_googlenet
    graph_inception_resnet_v1
    graph_inception_resnet_v2
    graph_inception_v3
    graph_inception_v4
    graph_lenet
    graph_mobilenet_v2
    graph_mobilenet
    graph_resnet_v2_50
    graph_resnet12
    graph_resnet50
    graph_resnext50
    graph_shufflenet
    graph_squeezenet_v1_1
    graph_squeezenet
    graph_srcnn955
    graph_ssd_mobilenet
    graph_vgg_vdsr
    graph_vgg16
    graph_vgg19
    graph_yolov3
    PARENT_SCOPE)

set(EXAMPLE_NEON_NAMES
    neon_cnn neon_copy_objects neon_gemm_qasymm8 neon_permute neon_scale
    neon_sgemm
    PARENT_SCOPE)
