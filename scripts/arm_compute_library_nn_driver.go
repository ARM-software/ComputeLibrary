//
// Copyright © 2020 ARM Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

package arm_compute_library_nn_driver

import (
    "android/soong/android"
    "android/soong/cc"
)

func globalFlags(ctx android.BaseContext) []string {
    var cppflags []string

    if ctx.AConfig().PlatformVersionName() == "Q" || ctx.AConfig().PlatformVersionName() == "10" {
        cppflags = append(cppflags, "-fno-addrsig")
    }

    if ctx.AConfig().PlatformVersionName() == "R" || ctx.AConfig().PlatformVersionName() == "11" {
        cppflags = append(cppflags, "-fno-addrsig")
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
