// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use std::env;
use std::path::PathBuf;

fn main() {
    let curve = "FEATURE_BLS12_377";

    // account for cross-compilation [by examining environment variable]
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();

    // Set CC environment variable to choose alternative C compiler.
    // Optimization level depends on whether or not --release is passed
    // or implied.
    let mut cc = cc::Build::new();

    let c_src_dir = PathBuf::from("src");
    let files = vec![c_src_dir.join("lib.c")];
    let mut cc_opt = None;

    match (cfg!(feature = "portable"), cfg!(feature = "force-adx")) {
        (true, false) => {
            println!("Compiling in portable mode without ISA extensions");
            cc_opt = Some("__BLST_PORTABLE__");
        }
        (false, true) => {
            if target_arch.eq("x86_64") {
                println!("Enabling ADX support via `force-adx` feature");
                cc_opt = Some("__ADX__");
            } else {
                println!("`force-adx` is ignored for non-x86_64 targets");
            }
        }
        (false, false) => {
            #[cfg(target_arch = "x86_64")]
            if target_arch.eq("x86_64") && std::is_x86_feature_detected!("adx")
            {
                println!("Enabling ADX because it was detected on the host");
                cc_opt = Some("__ADX__");
            }
        }
        (true, true) => panic!(
            "Cannot compile with both `portable` and `force-adx` features"
        ),
    }

    cc.flag_if_supported("-mno-avx") // avoid costly transitions
        .flag_if_supported("-fno-builtin")
        .flag_if_supported("-Wno-unused-command-line-argument");
    if !cfg!(debug_assertions) {
        cc.opt_level(2);
    }
    if let Some(def) = cc_opt {
        cc.define(def, None);
    }
    if let Some(include) = env::var_os("DEP_BLST_C_SRC") {
        cc.include(include);
    }
    cc.files(&files).compile("blst_msm");

    if cfg!(target_os = "windows") && !cfg!(target_env = "msvc") {
        return;
    }
    // Detect if there is CUDA compiler and engage "cuda" feature accordingly
    let nvcc = match env::var("NVCC") {
        Ok(var) => which::which(var),
        Err(_) => which::which("nvcc"),
    };

    if nvcc.is_ok() {
        let mut nvcc = cc::Build::new();
        nvcc.cuda(true);
        nvcc.flag("-arch=sm_80");
        if let Some(def) = cc_opt {
            nvcc.define(def, None);
        }
        nvcc.file("yrrid-msm/MSM.cu").compile("msm");

        println!("cargo:rustc-cfg=feature=\"cuda\"");
        println!("cargo:rerun-if-changed=cuda");
        println!("cargo:rerun-if-env-changed=CXXFLAGS");
    } else {
        panic!("nvcc must be in the path. Consider adding /usr/local/cuda/bin.")
    }
}
