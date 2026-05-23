/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Disk-cache (`JitStore`) integration test.
//!
//! This is its own test binary on purpose: `set_jit_store` is a process-global
//! one-shot, so a dedicated process keeps the store deterministic and lets the
//! test assert exact `jit_compile_count` / `jit_disk_hit_count` deltas instead
//! of relying on wall-clock timing.
//!
//! Requires a GPU (it JIT-compiles and loads a real kernel).

use cutile::api;
use cutile::jit_store::FileSystemJitStore;
use cutile::prelude::{DeviceOp, PartitionOp};
use cutile::tile_kernel::{
    evict_kernel, get_default_device, jit_compile_count, jit_disk_hit_count, FunctionKey,
    TileFunctionKey, TileKernel,
};
use cutile_compiler::cuda_tile_runtime_utils::{
    get_compiler_version, get_cuda_toolkit_version, get_gpu_name,
};
use cutile_compiler::specialization::SpecializationBits;

mod common;

#[cutile::module]
mod disk_cache_test_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn vector_add<T: ElementType, const N: i32>(
        z: &mut Tensor<T, { [N] }>,
        x: &Tensor<T, { [-1] }>,
        y: &Tensor<T, { [-1] }>,
    ) {
        let tile_x = load_tile_like(x, z);
        let tile_y = load_tile_like(y, z);
        z.store(tile_x + tile_y);
    }
}

fn stride_args() -> Vec<(String, Vec<i32>)> {
    vec![
        ("z".to_string(), vec![1]),
        ("x".to_string(), vec![1]),
        ("y".to_string(), vec![1]),
    ]
}

fn spec_args() -> Vec<(String, SpecializationBits)> {
    let x = api::ones::<f32>(&[256]).sync().unwrap();
    let y = api::ones::<f32>(&[256]).sync().unwrap();
    let z = api::zeros::<f32>(&[256]).partition([256]).sync().unwrap();
    vec![
        ("z".to_string(), z.unpartition().spec().clone()),
        ("x".to_string(), x.spec().clone()),
        ("y".to_string(), y.spec().clone()),
    ]
}

/// Reconstruct the exact in-memory cache key for our `vector_add` launch so we
/// can evict *only* this kernel between phases without recompiling unrelated
/// helper kernels (e.g. the fill kernel behind `api::ones`/`api::zeros`).
fn vector_add_key() -> TileFunctionKey {
    let device_id = get_default_device();
    TileFunctionKey::builder("disk_cache_test_module", "vector_add")
        .generics(vec!["f32".into(), "256".into()])
        .stride_args(stride_args())
        .spec_args(spec_args())
        .source_hash(disk_cache_test_module::_SOURCE_HASH)
        .gpu_name(get_gpu_name(device_id))
        .compiler_version(get_compiler_version())
        .cuda_toolkit_version(get_cuda_toolkit_version())
        .build()
}

fn launch_vector_add() {
    let x = api::ones::<f32>(&[256]).sync().unwrap();
    let y = api::ones::<f32>(&[256]).sync().unwrap();
    let z = api::zeros::<f32>(&[256]).partition([256]).sync().unwrap();
    disk_cache_test_module::vector_add(z, &x, &y)
        .generics(vec!["f32".into(), "256".into()])
        .sync()
        .expect("vector_add launch failed");
}

fn cubin_files(dir: &std::path::Path) -> usize {
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter(|e| {
                e.as_ref()
                    .unwrap()
                    .path()
                    .extension()
                    .is_some_and(|ext| ext == "cubin")
            })
            .count()
        })
        .unwrap_or(0)
}

#[test]
fn disk_cache_miss_hit_and_bad_cubin_recovery() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();

        let dir = std::env::temp_dir()
            .join(format!("cutile_jit_disk_cache_it_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let store =
            FileSystemJitStore::new(dir.clone()).expect("failed to create FileSystemJitStore");
        // Dedicated binary: the store must be unset and ours.
        assert!(
            cuda_async::jit_store::set_jit_store_if_unset(Some(Box::new(store))),
            "JIT store was unexpectedly already configured in this test binary"
        );

        // Prime fill kernel so tensor construction won't move the JIT counter later.
        let _ = api::ones::<f32>(&[256]).sync().unwrap();
        let _ = api::zeros::<f32>(&[256]).partition([256]).sync().unwrap();

        // Grab fill kernel's cubin (valid ELF, but lacks vector_add's symbol)
        // before vector_add is ever compiled — Phase 4 uses it as a foreign entry.
        let foreign_cubin = {
            let path = std::fs::read_dir(&dir)
                .unwrap()
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .find(|p| p.extension().is_some_and(|ext| ext == "cubin"))
                .expect("fill kernel cubin should exist after priming");
            std::fs::read(&path).unwrap()
        };

        let key_str = vector_add_key().get_hash_string();

        // ── Phase 1: cold — in-memory miss + disk miss → full compile ──
        evict_kernel(&key_str);
        let c0 = jit_compile_count();
        let h0 = jit_disk_hit_count();
        launch_vector_add();
        assert_eq!(
            jit_compile_count(),
            c0 + 1,
            "first launch must JIT-compile exactly once"
        );
        assert_eq!(
            jit_disk_hit_count(),
            h0,
            "first launch is not a disk hit"
        );
        assert!(
            cubin_files(&dir) > 0,
            "the compiled cubin must be persisted to disk"
        );

        // ── Phase 2: warm disk — evict in-memory only → disk HIT ──
        evict_kernel(&key_str);
        let c1 = jit_compile_count();
        let h1 = jit_disk_hit_count();
        launch_vector_add();
        assert_eq!(
            jit_compile_count(),
            c1,
            "disk hit must NOT recompile (jit_compile_count must not move)"
        );
        assert_eq!(
            jit_disk_hit_count(),
            h1 + 1,
            "second launch must be served from the disk cache exactly once"
        );

        // ── Phase 3: corrupted cubin → delete + recompile ──
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|e| e == "cubin") {
                std::fs::write(&path, b"not a valid cubin").unwrap();
            }
        }
        evict_kernel(&key_str);
        let c2 = jit_compile_count();
        let h2 = jit_disk_hit_count();
        launch_vector_add();
        assert_eq!(
            jit_compile_count(),
            c2 + 1,
            "a corrupted cubin must fall through to a full recompile"
        );
        assert_eq!(
            jit_disk_hit_count(),
            h2,
            "a corrupted cubin is not a successful disk hit"
        );
        assert!(
            cubin_files(&dir) > 0,
            "the recompiled cubin must be re-persisted to disk"
        );

        // ── Phase 4: valid cubin, wrong symbol → delete + recompile ──
        let disk_path = dir.join(format!("{}.cubin", vector_add_key().get_disk_hash_string()));
        std::fs::write(&disk_path, &foreign_cubin).unwrap();
        evict_kernel(&key_str);
        let c3 = jit_compile_count();
        let h3 = jit_disk_hit_count();
        launch_vector_add();
        assert_eq!(
            jit_compile_count(),
            c3 + 1,
            "a cubin missing the entry symbol must fall through to a full recompile"
        );
        assert_eq!(
            jit_disk_hit_count(),
            h3,
            "a cubin missing the entry symbol is not a successful disk hit"
        );

        let _ = std::fs::remove_dir_all(&dir);
    });
}
