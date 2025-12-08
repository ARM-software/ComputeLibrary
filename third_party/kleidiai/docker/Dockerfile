#
# SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

# Ubuntu base version
ARG UBUNTU_VERSION=24.04

FROM ubuntu:${UBUNTU_VERSION}

# These arguments are required to support multi-arch images
ARG BUILDARCH
ARG BUILDPLATFORM
ARG TARGETARCH
ARG TARGETOS
ARG TARGETPLATFORM

# Component versions
ARG FVP_UBUNTU_VERSION=24.04.3
ARG BAZEL_BUILD_TOOLS_VER=v8.2.1
ARG BAZELISK_VERSION=v1.26.0
ARG NDK_VERSION=r27d
ARG FVP_VERSION=11.29_42

ENV XDG_CACHE_HOME=/cache ANDROID_NDK=/opt/devtools/android-ndk BUILD_CACHE=/build_cache

RUN mkdir -p /opt/devtools ${XDG_CACHE_HOME}

# Software from base image required during image usage
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get -y --no-install-recommends install \
    build-essential \
    ca-certificates \
    clang \
    clang-tidy \
    cmake \
    curl \
    e2tools \
    gcovr \
    git \
    llvm \
    libclang-rt-dev \
    ninja-build \
    pre-commit \
    python3 \
    python3-pip \
    reuse \
    unzip \
    wget

# Software from base image required during image build
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y --no-install-recommends install \
    autoconf \
    automake \
    device-tree-compiler \
    flex \
    bison \
    bc \
    libssl-dev \
    genext2fs \
    ccache

WORKDIR /opt/devtools

# =============================================================================
# Linux virtual machine on FVP.
# =============================================================================

# Downloads the latest Fixed Virtual Platform.
# The links could be found at https://developer.arm.com/Tools%20and%20Software/Fixed%20Virtual%20Platforms/Arm%20Architecture%20FVPs
RUN --mount=type=cache,target=${BUILD_CACHE} <<EOF
    FVP_NAME=FVP_Base_RevC-2xAEMvA_${FVP_VERSION}_Linux64$(test "${TARGETARCH}" = "arm64" && echo -n "_armv8l")
    set -e
    # Public URL changes quite frequent and using a different scheme. Try each variant sequentially for a bit of backward compatibility
    for FVP_BASE_VERSION in $(echo -n ${FVP_VERSION} | cut -d '_' -f 1) $(echo -n ${FVP_VERSION} | tr '_' '-') ; do
        wget -cO ${BUILD_CACHE}/${FVP_NAME}.tgz \
            "https://developer.arm.com/-/cdn-downloads/permalink/FVPs-Architecture/FM-${FVP_BASE_VERSION}/${FVP_NAME}.tgz" && break
    done
    set +e
    mkdir -vp ${FVP_NAME}
    tar xzf ${BUILD_CACHE}/${FVP_NAME}.tgz -C ${FVP_NAME}
    ln -s /opt/devtools/${FVP_NAME}/Base_RevC_AEMvA_pkg /opt/devtools/fvp_base_aemva
EOF

# Creates the root filesystem.
RUN --mount=type=cache,target=${BUILD_CACHE},sharing=locked \
    export UBUNTU_DISTRO_VERSION=$(echo -n "$FVP_UBUNTU_VERSION" | cut -d '.' -f 1,2) && \
    wget -cO ${BUILD_CACHE}/ubuntu-base-${FVP_UBUNTU_VERSION}-base-arm64.tar.gz "https://cdimage.ubuntu.com/ubuntu-base/releases/${UBUNTU_DISTRO_VERSION}/release/ubuntu-base-${FVP_UBUNTU_VERSION}-base-arm64.tar.gz" && \
    genext2fs -a ${BUILD_CACHE}/ubuntu-base-${FVP_UBUNTU_VERSION}-base-arm64.tar.gz -B 4096 -b 51200 linux-rootfs.img

# Compiles the latest Linux kernel, bootloader and root file system for FVP usage
RUN --mount=type=cache,target=${BUILD_CACHE},sharing=locked \
    --mount=type=bind,source=build_linux_bootloader.sh,target=build_linux_bootloader.sh \
    ./build_linux_bootloader.sh

WORKDIR /

# Download and install bazelisk
RUN wget "https://github.com/bazelbuild/bazelisk/releases/download/${BAZELISK_VERSION}/bazelisk-${TARGETOS}-${TARGETARCH}" -O /usr/bin/bazelisk && \
    chmod a+x /usr/bin/bazelisk

RUN wget "https://github.com/bazelbuild/buildtools/releases/download/${BAZEL_BUILD_TOOLS_VER}/buildifier-${TARGETOS}-${TARGETARCH}" -O /usr/bin/buildifier && \
    wget "https://github.com/bazelbuild/buildtools/releases/download/${BAZEL_BUILD_TOOLS_VER}/buildozer-${TARGETOS}-${TARGETARCH}" -O /usr/bin/buildozer && \
    wget "https://github.com/bazelbuild/buildtools/releases/download/${BAZEL_BUILD_TOOLS_VER}/unused_deps-${TARGETOS}-${TARGETARCH}" -O /usr/bin/unused_deps && \
    chmod a+x /usr/bin/buildifier /usr/bin/buildozer /usr/bin/unused_deps

# Android NDK for amd64 only
RUN --mount=type=cache,target=${BUILD_CACHE} \
    if [ "${TARGETARCH}" = "amd64" ] ; then \
        wget -cO ${BUILD_CACHE}/android-ndk-${NDK_VERSION}-${TARGETOS}.zip "https://dl.google.com/android/repository/android-ndk-${NDK_VERSION}-${TARGETOS}.zip" && \
        unzip ${BUILD_CACHE}/android-ndk-${NDK_VERSION}-${TARGETOS}.zip -d /opt/devtools/ && \
        ln -s /opt/devtools/android-ndk-${NDK_VERSION} ${ANDROID_NDK}; \
    fi
