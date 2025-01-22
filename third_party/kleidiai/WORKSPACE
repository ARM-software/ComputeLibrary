#
# SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

workspace(name = "com_arm_kleidiai")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("kai_defs.bzl", "kai_local_archive")

http_archive(
    name = "bazel_skylib",
    sha256 = "08c0386f45821ce246bbbf77503c973246ed6ee5c3463e41efc197fa9bc3a7f4",
    strip_prefix = "bazel-skylib-288731ef9f7f688932bd50e704a91a45ec185f9b",
    urls = ["https://github.com/bazelbuild/bazel-skylib/archive/288731ef9f7f688932bd50e704a91a45ec185f9b.zip"],
)

kai_local_archive(
    name = "com_google_googletest",
    archive = "//:third_party/googletest-v1.14.0.zip",
    strip_prefix = "googletest-1.14.0",
)

kai_local_archive(
    name = "com_google_benchmark",
    archive = "//:third_party/benchmark-v1.8.4.zip",
    strip_prefix = "benchmark-v1.8.4",
)
