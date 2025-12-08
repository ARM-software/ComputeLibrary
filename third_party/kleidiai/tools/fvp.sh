#! /bin/bash
#
# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
# Brief: fvp.sh — helper to run an Arm FVP (Fixed Virtual Platform) with a Linux®
#        root filesystem image and a temporary startup script. It copies a
#        rootfs image to a temp file, injects a startup script that mounts a
#        host directory via 9p, and runs the provided test executable inside
#        the emulated environment.
#
# Main options (CLI):
#  --host-path     Path on the host to expose to the emulated target (default: $PWD)
#  --target-path   Mount point inside the emulated target (default: same as host-path)
#  --model-extra   Extra FVP model parameters to append to the invocation
#  <executable>    Test executable (and args) to run inside the guest (required)
#
# Environment variables might be used as alternative to command line options:
#  FVP_HOST_PATH    set from --host-path or defaults to $PWD
#  FVP_TARGET_PATH  set from --target-path or same as $FVP_HOST_PATH
#  FVP_MODEL_EXTRA  optional additional model parameters
#
# in addition to script these are passed automatically to FVP target
#  GTEST_*          GoogleTest variables
#

set -exu -o pipefail

cleanup()
{
    [ -n "${ROOTFS_IMAGE:-}" ] && rm -fv ${ROOTFS_IMAGE}
    [ -n "${TMP_FILE:-}" ] && rm -fv ${TMP_FILE}
}

trap cleanup EXIT INT

# Default values
HOST_PATH="${FVP_HOST_PATH:-}"
TARGET_PATH="${FVP_TARGET_PATH:-}"
MODEL_EXTRA="${FVP_MODEL_EXTRA:-}"

# Use getopt for argument parsing
OPTS=$(getopt -o 'h:t:l:m:' --long host-path:,target-path:,model-extra:, -n "$0" -- "$@")

if [ $? != 0 ]; then
    echo "Failed to parse arguments." >&2
    exit 1
fi

eval set -- "$OPTS"

while true; do
    case "$1" in
        --host-path)
            HOST_PATH="$2"
            shift 2
            ;;
        --target-path)
            TARGET_PATH="$2"
            shift 2
            ;;
        --model-extra)
            MODEL_EXTRA="$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        --*)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# FVP executable and parameters take rest of arguments
FVP_TEST_EXECUTABLE="$*"
# Check executable is set
[ -z "$FVP_TEST_EXECUTABLE" ] && exit 1

# Export variables for later use
FVP_HOST_PATH="${HOST_PATH:-$PWD}"

# Set target path, use host as default
FVP_TARGET_PATH="${TARGET_PATH:-$FVP_HOST_PATH}"

# Set extra model parameters
FVP_MODEL_EXTRA="$MODEL_EXTRA"

export FVP_HOST_PATH
export FVP_TARGET_PATH
export FVP_MODEL_EXTRA

# Temporary file for startup script
TMP_FILE=$(mktemp -t kaiXXXXXX)

cat > $TMP_FILE <<EOF
#!/bin/sh

set -e
echo '=================================================='
echo '== START                                        =='
echo '=================================================='
echo '== CPU INFO                                     =='
if [ ! -f /proc/cpuinfo ]; then mount -vt proc -o rw,nosuid,nodev,noexec proc /proc; fi
cat /proc/cpuinfo
echo '=================================================='

set -x

mkdir -vp '$FVP_TARGET_PATH'
mount -vt 9p -o trans=virtio,version=9p2000.L FM '$FVP_TARGET_PATH'

# Import GTest environment variables from host env
$(printenv | awk -F= '/^GTEST_/ {print "export " $1 "=\"" $2 "\"" }')

cd '$FVP_TARGET_PATH'
# Disable immediate exit on error
set +e
${FVP_TEST_EXECUTABLE} && echo 'FINISHED WITHOUT ERROR'
sync

echo '=================================================='
echo '== END                                          =='
echo '=================================================='
EOF

# Copy root fs to a temporary file to allow parallel execution in a same instance
ROOTFS_IMAGE=$(mktemp -t kaiXXXXXX)
cp /opt/devtools/linux-rootfs.img ${ROOTFS_IMAGE}

e2cp -O 0 -G 0 -P 755 $TMP_FILE ${ROOTFS_IMAGE}:/root/startup

FVP_TOOL_BINARY=(/opt/devtools/fvp_base_aemva/models/Linux64*GCC-9.3/FVP_Base_RevC-2xAEMvA)
if [ "${#FVP_TOOL_BINARY[@]}" -ne 1 ]; then
    echo "Error: Several FVP executables found. Must have exactly one element, found ${#FVP_TOOL_BINARY[@]}"
    exit 1
fi

${FVP_TOOL_BINARY[0]} \
    -C cache_state_modelled=0 \
    -C bp.refcounter.non_arch_start_at_default=1 \
    -C bp.secure_memory=0 \
    -C bp.pl011_uart0.out_file=- \
    -C bp.pl011_uart0.shutdown_tag="System halted" \
    -C bp.pl011_uart0.unbuffered_output=1 \
    -C bp.terminal_0.mode=telnet \
    -C bp.terminal_0.start_telnet=0 \
    -C bp.terminal_1.mode=raw \
    -C bp.terminal_1.start_telnet=0 \
    -C bp.terminal_2.mode=raw \
    -C bp.terminal_2.start_telnet=0 \
    -C bp.terminal_3.mode=raw \
    -C bp.terminal_3.start_telnet=0 \
    -C pctl.startup=*.*.*.* \
    -C cluster1.NUM_CORES=0 \
    -C cluster0.NUM_CORES=2 \
    -C cluster0.has_arm_v8-1=1 \
    -C cluster0.has_arm_v8-2=1 \
    -C cluster0.has_arm_v8-3=1 \
    -C cluster0.has_arm_v8-4=1 \
    -C cluster0.has_arm_v8-5=1 \
    -C cluster0.has_arm_v8-6=1 \
    -C cluster0.has_arm_v8-7=1 \
    -C cluster0.has_arm_v8-8=1 \
    -C cluster0.has_arm_v8-9=1 \
    -C cluster0.has_arm_v9-0=1 \
    -C cluster0.has_arm_v9-1=1 \
    -C cluster0.has_arm_v9-2=1 \
    -C cluster0.has_arm_v9-3=1 \
    -C cluster0.has_arm_v9-4=1 \
    -C cluster0.has_arm_v9-5=1 \
    -C cluster0.has_arm_v9-6=1 \
    -C cluster0.has_sve=1 \
    -C cluster0.sve.has_b16b16=1 \
    -C cluster0.sve.has_sve2=1 \
    -C cluster0.sve.has_sme=1 \
    -C cluster0.sve.has_sme2=1 \
    -C cluster0.sve.has_sme_f16f16=1 \
    -C cluster0.sve.has_sme_fa64=1 \
    -C cluster0.sve.has_sme_lutv2=1 \
    -C cluster0.sve.sme2_version=1 \
    -C cluster0.sve.veclen=2 \
    -C cluster0.sve.sme_veclens_implemented=4 \
    -C bp.virtio_rng.enabled=1 \
    -C bp.virtioblockdevice.image_path=$ROOTFS_IMAGE \
    -C bp.vis.disable_visualisation=1 \
    -C bp.virtiop9device.root_path=$FVP_HOST_PATH \
    -a cluster*.cpu*=/opt/devtools/linux-system.axf \
    --stat \
    ${FVP_MODEL_EXTRA:-} |& tee $TMP_FILE

grep -q "FINISHED WITHOUT ERROR" $TMP_FILE
