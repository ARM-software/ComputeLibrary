//
// Copyright © 2020-2023 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

package arm_compute_library_nn_driver

import (
    "android/soong/android"
    "android/soong/cc"
    "strings"
)

func isVersionAtLeast(version_name string, target_version int) bool {
    name_map := map[string]int {
    "L": 5, "5": 5,
    "M": 6, "6": 6,
    "N": 7, "7": 7,
    "O": 8, "8": 8,
    "P": 9, "9": 9,
    "Q": 10, "10": 10,
    "R": 11, "11": 11,
    "S": 12, "12": 12,
    "T": 13, "13": 13,
    "U": 14, "14": 14,
    }
    if _, ok := name_map[version_name]; ok {
        return name_map[version_name] >= target_version
    } else {
        return false
    }
}

func globalFlags(ctx android.BaseContext) []string {
    var cppflags []string

   if ctx.AConfig().PlatformVersionName() == "Q" || ctx.AConfig().PlatformVersionName() == "10" ||
      ctx.AConfig().PlatformVersionName() == "R" || ctx.AConfig().PlatformVersionName() == "11" ||
      ctx.AConfig().PlatformVersionName() == "S" || ctx.AConfig().PlatformVersionName() == "12" {
        cppflags = append(cppflags, "-fno-addrsig")
    }

    if ctx.AConfig().PlatformVersionName() == "R" || ctx.AConfig().PlatformVersionName() == "11" {
      for _, a := range ctx.DeviceConfig().Arches() {
        theArch := a.ArchType.String()
        if theArch == "armv8-2a" {
          cppflags = append(cppflags, "-march=armv8.2-a+fp16")
          cppflags = append(cppflags, "-DARM_COMPUTE_ENABLE_FP16")
        }
      }
    }

    // Since Android T, the underlying NDK stops supporting system assembler like GAS, in favor of integrated assembler
    // However for Android < Android T we still want to suppress integrated assembler for backward compatibility
    if ! isVersionAtLeast(ctx.AConfig().PlatformVersionName(), 13) {
        cppflags = append(cppflags, "-no-integrated-as")
    }

    data_types := strings.Split(ctx.AConfig().GetenvWithDefault("COMPUTE_LIB_DATA_TYPE", "ALL"), ",")

    for _, x := range data_types {
        if strings.ToUpper(x) == "ALL" || strings.ToUpper(x) == "INTEGER" {
            cppflags = append(cppflags, "-DENABLE_INTEGER_KERNELS")
        }
        if strings.ToUpper(x) == "ALL" || strings.ToUpper(x) == "QASYMM8" {
            cppflags = append(cppflags, "-DENABLE_QASYMM8_KERNELS")
        }
        if strings.ToUpper(x) == "ALL" || strings.ToUpper(x) == "QASYMM8_SIGNED" {
            cppflags = append(cppflags, "-DENABLE_QASYMM8_SIGNED_KERNELS")
        }
        if strings.ToUpper(x) == "ALL" || strings.ToUpper(x) == "QASYMM16" {
            cppflags = append(cppflags, "-DENABLE_QASYMM16_KERNELS")
        }
        if strings.ToUpper(x) == "ALL" || strings.ToUpper(x) == "QSYMM16" {
            cppflags = append(cppflags, "-DENABLE_QSYMM16_KERNELS")
        }
        if strings.ToUpper(x) == "ALL" || strings.ToUpper(x) == "FP16" {
            cppflags = append(cppflags, "-DENABLE_FP16_KERNELS")
        }
        if strings.ToUpper(x) == "ALL" || strings.ToUpper(x) == "FP32" {
            cppflags = append(cppflags, "-DENABLE_FP32_KERNELS")
        }
    }

    data_layouts := strings.Split(ctx.AConfig().GetenvWithDefault("COMPUTE_LIB_DATA_LAYOUT", "ALL"), ",")

    for _, x := range data_layouts {
        if strings.ToUpper(x) == "ALL" || strings.ToUpper(x) == "NHWC" {
            cppflags = append(cppflags, "-DENABLE_NHWC_KERNELS")
        }
        if strings.ToUpper(x) == "ALL" || strings.ToUpper(x) == "NCHW" {
            cppflags = append(cppflags, "-DENABLE_NCHW_KERNELS")
        }
    }

    cppflags = append(cppflags, "-DARM_COMPUTE_CPU_ENABLED")
    cppflags = append(cppflags, "-DARM_COMPUTE_OPENCL_ENABLED")

    return cppflags
}

func clframeworkNNDriverDefaults(ctx android.LoadHookContext) {
        type props struct {
                Cppflags []string
        }

        p := &props{}
        p.Cppflags = globalFlags(ctx)

        ctx.AppendProperties(p)
}

func init() {

  android.RegisterModuleType("arm_compute_library_defaults", clframeworkNNDriverDefaultsFactory)
}

func clframeworkNNDriverDefaultsFactory() android.Module {

   module := cc.DefaultsFactory()
   android.AddLoadHook(module, clframeworkNNDriverDefaults)
   return module
}
