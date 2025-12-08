#!/bin/bash -eux

#
# SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

TARGETARCH=${TARGETARCH:-$(if [ "`uname -m`" == "aarch64" ]; then echo "arm64"; else echo "amd64"; fi)}

# This script is used by Dockerfile to create a Linux bootloader with the latest Linux kernel.
if [ "`uname -s`" = "Darwin" ]; then
    HOST_ARCH=darwin-arm64
    TARGETARCH=arm64
elif [ "${TARGETARCH}" = "amd64" ] ; then
    HOST_ARCH=x86_64
elif [ "${TARGETARCH}" = "arm64" ] ; then
    HOST_ARCH=aarch64
else
    echo "Unknown $TARGETARCH" && exit 1
fi

TOOLCHAIN_VERSION=14.3.rel1
TOOLCHAIN_TYPE=aarch64-none-elf
TOOLCHAIN_DIR=$(pwd)/toolchain-${TOOLCHAIN_TYPE}
CROSS_COMPILE=${TOOLCHAIN_DIR}/bin/${TOOLCHAIN_TYPE}-
KERNEL_VERSION=6.16
# Derive DTS version from kernel version
DTS_VERSION=$(echo $KERNEL_VERSION | cut -d '.' -f 1,2)
BOOTLOADER_VERSION=785302c1f7b9eceab3b72a8cb3d79eaf526fd2e3

mkdir -p ${BUILD_CACHE}

# Downloads tools and source code.
# Download Arm toolchain
download_and_extract \
    "https://developer.arm.com/-/media/Files/downloads/gnu/${TOOLCHAIN_VERSION}/binrel/arm-gnu-toolchain-${TOOLCHAIN_VERSION}-${HOST_ARCH}-${TOOLCHAIN_TYPE}.tar.xz" \
    "${TOOLCHAIN_DIR}"

# Download Linux Kernel
if [[ "${KERNEL_VERSION}" =~ "-rc" ]]; then
download_and_extract \
    "https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/snapshot/linux-${KERNEL_VERSION}.tar.gz" \
    "linux-${KERNEL_VERSION}"
else
download_and_extract \
    "https://cdn.kernel.org/pub/linux/kernel/v$(echo $KERNEL_VERSION | cut -d '.' -f 1).x/linux-${KERNEL_VERSION}.tar.xz" \
    "linux-${KERNEL_VERSION}"
fi

# Download booloader
download_and_extract \
    "https://git.kernel.org/pub/scm/linux/kernel/git/mark/boot-wrapper-aarch64.git/snapshot/boot-wrapper-aarch64-${BOOTLOADER_VERSION}.tar.gz" \
    boot-wrapper-aarch64

# Download DTS tooling
download_and_extract \
    "https://git.kernel.org/pub/scm/linux/kernel/git/devicetree/devicetree-rebasing.git/snapshot/devicetree-rebasing-${DTS_VERSION}-dts.tar.gz" \
    devicetree-rebasing

# Builds the Linux kernel.
cd linux-${KERNEL_VERSION}
CCACHE_DIR=${BUILD_CACHE}/ccache make ARCH=arm64 CROSS_COMPILE="ccache ${CROSS_COMPILE}" defconfig
CCACHE_DIR=${BUILD_CACHE}/ccache make ARCH=arm64 CROSS_COMPILE="ccache ${CROSS_COMPILE}" "-j$(nproc)" Image
cd ..

# Builds the device tree.
cd devicetree-rebasing
# Reduce number of CPUs to 2 and disable second cluster to keep in sync with FVP parameters. In FVP no more
# than 2 CPUs are needed. cpu0 to run Linux kernel and system services. cpu1 used to run a test program in isolation.
cat <<EOF > src/arm64/arm/kleidiai_fvp.dts
/dts-v1/;

#include "fvp-base-revc.dts"

/delete-node/ &cpu2;
/delete-node/ &cpu3;
/delete-node/ &cpu4;
/delete-node/ &cpu5;
/delete-node/ &cpu6;
/delete-node/ &cpu7;
/delete-node/ &C1_L2;

/ {
    /delete-node/ ete-2;
    /delete-node/ ete-3;
    /delete-node/ ete-4;
    /delete-node/ ete-5;
    /delete-node/ ete-6;
    /delete-node/ ete-7;
};
EOF
make CPP=${CROSS_COMPILE}cpp src/arm64/arm/kleidiai_fvp.dtb
cd ..

# Builds the bootloader.
cd boot-wrapper-aarch64
PATH=$(dirname ${CROSS_COMPILE}gcc):$PATH
export PATH
autoreconf -i
# Extra CFLAGS are needed to avoid using standard libraries when using baremetal toolchain
# -O2 required as otherwise boot-wrapper would not boot
CFLAGS="-ffreestanding -nostdlib -O2" ./configure --host=${TOOLCHAIN_TYPE} \
    --enable-psci \
    --enable-gicv3 \
    --with-kernel-dir=../linux-${KERNEL_VERSION} \
    --with-dtb=../devicetree-rebasing/src/arm64/arm/kleidiai_fvp.dtb \
    --with-cmdline="console=ttyAMA0 earlycon=pl011,0x1c090000 panic=1 root=/dev/vda rw isolcpus=1 init=/bin/bash -- /root/startup"
make "-j$(nproc)"
cd ..

mv boot-wrapper-aarch64/linux-system.axf .

# Cleans up.
rm -rf \
    ${TOOLCHAIN_DIR} \
    linux-${KERNEL_VERSION} \
    devicetree-rebasing \
    boot-wrapper-aarch64
