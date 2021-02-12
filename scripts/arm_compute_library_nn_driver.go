//
// Copyright © 2020 ARM Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

package arm_compute_library_nn_driver

import (
    "android/soong/android"
    "android/soong/cc"
    "strings"
)

func globalFlags(ctx android.BaseContext) []string {
    var cppflags []string

   if ctx.AConfig().PlatformVersionName() == "Q" || ctx.AConfig().PlatformVersionName() == "10" || 
      ctx.AConfig().PlatformVersionName() == "R" || ctx.AConfig().PlatformVersionName() == "11" ||
      ctx.AConfig().PlatformVersionName() == "S" || ctx.AConfig().PlatformVersionName() == "12" {
        cppflags = append(cppflags, "-fno-addrsig")
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
