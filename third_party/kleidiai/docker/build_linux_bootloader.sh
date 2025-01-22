#!/bin/bash -eux

#
# SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

BUILD_CACHE=${BUILD_CACHE:-${HOME}/.cache/kleidiai}
download_and_extract()
{
    URL="$1"
    FOLDER="$2"
    ARCHIVE="${3:-$(basename $1)}"

    wget -cO ${BUILD_CACHE}/${ARCHIVE} "$URL"
    mkdir -p ${FOLDER}
    tar -xa -f ${BUILD_CACHE}/${ARCHIVE} --strip-components=1 -C ${FOLDER}
}

TARGETARCH=${TARGETARCH:-amd64}

# This script is used by Dockerfile to create a Linux bootloader with the latest Linux kernel.
if [ "${TARGETARCH}" = "amd64" ] ; then
    HOST_ARCH=x86_64
elif [ "${TARGETARCH}" = "arm64" ] ; then
    HOST_ARCH=aarch64
else
    echo "Unknown $TARGETARCH" && exit 1
fi

TOOLCHAIN_VER=13.3.rel1
TOOLCHAIN_TYPE=aarch64-none-elf
TOOLCHAIN_DIR=$(pwd)/toolchain-${TOOLCHAIN_TYPE}/
CROSS_COMPILE=${TOOLCHAIN_DIR}/bin/${TOOLCHAIN_TYPE}-
KERNEL_VERSION=6.9.12

mkdir -p ${BUILD_CACHE}

# Downloads tools and source code.
# Download Arm toolchain
download_and_extract \
    "https://developer.arm.com/-/media/Files/downloads/gnu/${TOOLCHAIN_VER}/binrel/arm-gnu-toolchain-${TOOLCHAIN_VER}-${HOST_ARCH}-${TOOLCHAIN_TYPE}.tar.xz" \
    "${TOOLCHAIN_DIR}"

# Download Linux Kernel
download_and_extract \
    "https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-${KERNEL_VERSION}.tar.xz" \
    "linux-${KERNEL_VERSION}"

# Download booloader
# Revision 1fea854771f9aee405c4ae204c0e0f912318da6f supports bare metal gcc, otherwise hosted toolchain should be used
download_and_extract \
    "https://git.kernel.org/pub/scm/linux/kernel/git/mark/boot-wrapper-aarch64.git/snapshot/boot-wrapper-aarch64-1fea854771f9aee405c4ae204c0e0f912318da6f.tar.gz" \
    boot-wrapper-aarch64

# Download DTS tooling
download_and_extract \
    "https://git.kernel.org/pub/scm/linux/kernel/git/devicetree/devicetree-rebasing.git/snapshot/devicetree-rebasing-$(echo $KERNEL_VERSION | cut -d '.' -f 1,2)-dts.tar.gz" \
    devicetree-rebasing

# Builds the Linux kernel.
cd linux-${KERNEL_VERSION}
CCACHE_DIR=${BUILD_CACHE}/ccache make ARCH=arm64 CROSS_COMPILE="ccache ${CROSS_COMPILE}" defconfig
CCACHE_DIR=${BUILD_CACHE}/ccache make ARCH=arm64 CROSS_COMPILE="ccache ${CROSS_COMPILE}" "-j$(nproc)" Image
cd ..

# Builds the device tree.
cd devicetree-rebasing
make CPP=${CROSS_COMPILE}cpp src/arm64/arm/fvp-base-revc.dtb
cd ..

# Builds the bootloader.
cd boot-wrapper-aarch64
PATH=$(dirname ${CROSS_COMPILE}gcc):$PATH
export PATH
autoreconf -i
./configure --host=${TOOLCHAIN_TYPE} \
    --enable-psci \
    --enable-gicv3 \
    --with-kernel-dir=../linux-${KERNEL_VERSION} \
    --with-dtb=../devicetree-rebasing/src/arm64/arm/fvp-base-revc.dtb \
    --with-cmdline="console=ttyAMA0 earlycon=pl011,0x1c090000 panic=1 root=/dev/vda rw init=/bin/bash -- /root/startup"
make "-j$(nproc)"
cd ..

mv boot-wrapper-aarch64/linux-system.axf .

# Cleans up.
rm -rf \
    ${TOOLCHAIN_DIR} \
    linux-${KERNEL_VERSION} \
    devicetree-rebasing \
    boot-wrapper-aarch64
