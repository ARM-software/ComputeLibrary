#
# SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

FROM ubuntu:24.04

# These arguments are required to support multi-arch images via same Dockerfile
ARG TARGETARCH
ARG BUILDARCH
ARG FVP_UBUNTU_VERSION=24.04.1

ENV XDG_CACHE_HOME=/cache ANDROID_NDK=/opt/devtools/android-ndk BUILD_CACHE=/build_cache

RUN mkdir -p /opt/devtools ${XDG_CACHE_HOME}
# Bazel 6.5 cache config does not use XDG_CACHE_HOME and always ${HOME}/.cache/bazel
# but HOME is not set in the container environment
RUN ln -s ${XDG_CACHE_HOME} /.cache

RUN --mount=type=cache,target=/var/cache/apt --mount=type=cache,target=/var/lib/apt \
    apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get -y --no-install-recommends install \
    build-essential \
    ca-certificates \
    clang \
    clang-tidy \
    llvm \
    libclang-rt-dev \
    cmake \
    git \
    ninja-build \
    python3 \
    python3-pip \
    reuse \
    pre-commit \
    gcovr \
    e2tools \
    wget \
    autoconf \
    automake \
    device-tree-compiler \
    flex \
    bison \
    bc \
    libssl-dev \
    genext2fs \
    curl \
    unzip \
    ccache

WORKDIR /opt/devtools

# =============================================================================
# Linux virtual machine on FVP.
# =============================================================================

# Downloads the latest Fixed Virtual Platform.
RUN --mount=type=cache,target=${BUILD_CACHE} \
    export ARCH_SUFFIX=$(test "${TARGETARCH}" = "arm64" && echo -n "_armv8l") && \
    mkdir FVP_Base_RevC-2xAEMvA_11.25_15_Linux64${ARCH_SUFFIX} && \
    ln -s /opt/devtools/FVP_Base_RevC-2xAEMvA_11.25_15_Linux64${ARCH_SUFFIX}/Base_RevC_AEMvA_pkg /opt/devtools/fvp_base_aemva && \
    wget -cO ${BUILD_CACHE}/FVP_Base_RevC-2xAEMvA_11.25_15_Linux64${ARCH_SUFFIX}.tgz "https://developer.arm.com/-/media/Files/downloads/ecosystem-models/FM_11_25/FVP_Base_RevC-2xAEMvA_11.25_15_Linux64${ARCH_SUFFIX}.tgz" && \
    tar xzC FVP_Base_RevC-2xAEMvA_11.25_15_Linux64${ARCH_SUFFIX} -f ${BUILD_CACHE}/FVP_Base_RevC-2xAEMvA_11.25_15_Linux64${ARCH_SUFFIX}.tgz

# Creates the root filesystem.
RUN --mount=type=cache,target=${BUILD_CACHE} \
    export UBUNTU_DISTRO_VERSION=$(echo -n "$FVP_UBUNTU_VERSION" | cut -d '.' -f 1,2) && \
    wget -cO ${BUILD_CACHE}/ubuntu-base-${FVP_UBUNTU_VERSION}-base-arm64.tar.gz "https://cdimage.ubuntu.com/ubuntu-base/releases/${UBUNTU_DISTRO_VERSION}/release/ubuntu-base-${FVP_UBUNTU_VERSION}-base-arm64.tar.gz" && \
    genext2fs -a ${BUILD_CACHE}/ubuntu-base-${FVP_UBUNTU_VERSION}-base-arm64.tar.gz -B 4096 -b 51200 linux-rootfs.img && \
    tar cJ --remove-files -f linux-rootfs.img.xz linux-rootfs.img

# Compiles the latest Linux kernel, bootloader and root file system for FVP usage
RUN --mount=type=cache,target=${BUILD_CACHE} \
    --mount=type=bind,source=build_linux_bootloader.sh,target=build_linux_bootloader.sh \
    ./build_linux_bootloader.sh

WORKDIR /

RUN wget "https://github.com/bazelbuild/bazelisk/releases/download/v1.20.0/bazelisk-linux-${TARGETARCH}" -O /usr/bin/bazelisk && \
    chmod a+x /usr/bin/bazelisk && \
    bazelisk && \
    chmod -R a+rw ${XDG_CACHE_HOME}

ARG BAZEL_BUILD_TOOLS_VER=v7.1.2

RUN wget "https://github.com/bazelbuild/buildtools/releases/download/${BAZEL_BUILD_TOOLS_VER}/buildifier-linux-${TARGETARCH}" -O /usr/bin/buildifier && \
    wget "https://github.com/bazelbuild/buildtools/releases/download/${BAZEL_BUILD_TOOLS_VER}/buildozer-linux-${TARGETARCH}" -O /usr/bin/buildozer && \
    wget "https://github.com/bazelbuild/buildtools/releases/download/${BAZEL_BUILD_TOOLS_VER}/unused_deps-linux-${TARGETARCH}" -O /usr/bin/unused_deps && \
    chmod a+x /usr/bin/buildifier /usr/bin/buildozer /usr/bin/unused_deps

# Android NDK for amd64 only
ARG NDK_VERSION=r25c
RUN --mount=type=cache,target=${BUILD_CACHE} \
    if [ "${TARGETARCH}" = "amd64" ] ; then \
        wget -cO ${BUILD_CACHE}/android-ndk-${NDK_VERSION}-linux.zip "https://dl.google.com/android/repository/android-ndk-${NDK_VERSION}-linux.zip" && \
        unzip ${BUILD_CACHE}/android-ndk-${NDK_VERSION}-linux.zip -d /opt/devtools/ && \
        ln -s /opt/devtools/android-ndk-${NDK_VERSION} ${ANDROID_NDK}; \
    fi
