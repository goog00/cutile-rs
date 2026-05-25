/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! GPU integration tests for compile_warmup and execute_warmup.

use crate::common;
use cutile::api;
use cutile::prelude::{DeviceOp, PartitionOp};
use cutile::tile_kernel::{
    clear_kernel_cache, contains_cuda_function, get_default_device, jit_compile_count,
    LaunchHook, TileFunctionKey, TileKernel, WarmupCtx, WarmupSpec,
};
use cutile_compiler::cuda_tile_runtime_utils::{
    get_compiler_version, get_cuda_toolkit_version, get_gpu_name,
};
use cutile_compiler::specialization::SpecializationBits;
use std::sync::Arc;

#[cutile::module]
mod warmup_test_module {
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

fn vector_add_stride_args() -> Vec<(String, Vec<i32>)> {
    vec![
        ("z".to_string(), vec![1]),
        ("x".to_string(), vec![1]),
        ("y".to_string(), vec![1]),
    ]
}

fn vector_add_spec_args(len: usize, tile: usize) -> Vec<(String, SpecializationBits)> {
    let x = api::ones::<f32>(&[len]).sync().unwrap();
    let y = api::ones::<f32>(&[len]).sync().unwrap();
    let z = api::zeros::<f32>(&[len]).partition([tile]).sync().unwrap();
    let z_spec = z.unpartition().spec().clone();
    vec![
        ("z".to_string(), z_spec),
        ("x".to_string(), x.spec().clone()),
        ("y".to_string(), y.spec().clone()),
    ]
}

#[test]
fn compile_warmup_populates_cache() {
    common::with_test_stack(move || {
        let _guard = common::cache_test_lock();

        warmup_test_module::_compile_warmup(&[WarmupSpec::new(
            "vector_add",
            vec!["f32".into(), "64".into()],
        )
        .with_strides(vector_add_stride_args())
        .with_spec_args(vector_add_spec_args(256, 64))])
        .expect("_compile_warmup failed");

        let device_id = get_default_device();
        let key = TileFunctionKey::builder("warmup_test_module", "vector_add")
            .generics(vec!["f32".into(), "64".into()])
            .stride_args(vector_add_stride_args())
            .spec_args(vector_add_spec_args(256, 64))
            .source_hash(warmup_test_module::_SOURCE_HASH)
            .gpu_name(get_gpu_name(device_id))
            .compiler_version(get_compiler_version())
            .cuda_toolkit_version(get_cuda_toolkit_version())
            .build();
        assert!(
            contains_cuda_function(device_id, &key),
            "kernel should be in cache after compile_warmup"
        );
    });
}

#[test]
fn compile_warmup_skips_duplicate() {
    common::with_test_stack(move || {
        let _guard = common::cache_test_lock();
        let specs = &[WarmupSpec::new("vector_add", vec!["f32".into(), "128".into()])
            .with_strides(vector_add_stride_args())
            .with_spec_args(vector_add_spec_args(256, 128))];
        warmup_test_module::_compile_warmup(specs)
            .expect("first compile_warmup failed");
        warmup_test_module::_compile_warmup(specs)
            .expect("second compile_warmup failed");
    });
}

#[test]
fn failed_warmup_does_not_poison_cache() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();
        let result = warmup_test_module::_compile_warmup(
            &[WarmupSpec::new("nonexistent_fn", vec!["f32".into()])],
        );
        assert!(result.is_err(), "should error for unknown function");

        // Must succeed despite the prior failure.
        warmup_test_module::_compile_warmup(&[WarmupSpec::new("vector_add", vec!["f32".into(), "4".into()])
            .with_strides(vector_add_stride_args())
            .with_spec_args(vector_add_spec_args(256, 4))])
        .expect("valid warmup should succeed after failed one");

        let device_id = get_default_device();
        let key = TileFunctionKey::builder("warmup_test_module", "vector_add")
            .generics(vec!["f32".into(), "4".into()])
            .stride_args(vector_add_stride_args())
            .spec_args(vector_add_spec_args(256, 4))
            .source_hash(warmup_test_module::_SOURCE_HASH)
            .gpu_name(get_gpu_name(device_id))
            .compiler_version(get_compiler_version())
            .cuda_toolkit_version(get_cuda_toolkit_version())
            .build();
        assert!(
            contains_cuda_function(device_id, &key),
            "kernel should be in cache after recovery from prior failure"
        );
    });
}

#[test]
fn multi_spec_warmup_compiles_all() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();
        warmup_test_module::_compile_warmup(&[WarmupSpec::new("vector_add", vec!["f32".into(), "32".into()])
            .with_strides(vector_add_stride_args())
            .with_spec_args(vector_add_spec_args(256, 32))])
        .expect("pre-compile spec A failed");

        // A is cached (skip); B must still compile.
        warmup_test_module::_compile_warmup(&[
            WarmupSpec::new("vector_add", vec!["f32".into(), "32".into()])
                .with_strides(vector_add_stride_args())
                .with_spec_args(vector_add_spec_args(256, 32)),
            WarmupSpec::new("vector_add", vec!["f32".into(), "16".into()])
                .with_strides(vector_add_stride_args())
                .with_spec_args(vector_add_spec_args(256, 16)),
        ])
        .expect("multi-spec warmup failed");

        let device_id = get_default_device();
        let gpu_name = get_gpu_name(device_id);
        let compiler_version = get_compiler_version();
        let cuda_toolkit_version = get_cuda_toolkit_version();

        let key_a = TileFunctionKey::builder("warmup_test_module", "vector_add")
            .generics(vec!["f32".into(), "32".into()])
            .stride_args(vector_add_stride_args())
            .spec_args(vector_add_spec_args(256, 32))
            .source_hash(warmup_test_module::_SOURCE_HASH)
            .gpu_name(gpu_name.clone())
            .compiler_version(compiler_version.clone())
            .cuda_toolkit_version(cuda_toolkit_version.clone())
            .build();
        let key_b = TileFunctionKey::builder("warmup_test_module", "vector_add")
            .generics(vec!["f32".into(), "16".into()])
            .stride_args(vector_add_stride_args())
            .spec_args(vector_add_spec_args(256, 16))
            .source_hash(warmup_test_module::_SOURCE_HASH)
            .gpu_name(gpu_name.clone())
            .compiler_version(compiler_version.clone())
            .cuda_toolkit_version(cuda_toolkit_version.clone())
            .build();

        assert!(
            contains_cuda_function(device_id, &key_a),
            "spec A should be in cache"
        );
        assert!(
            contains_cuda_function(device_id, &key_b),
            "spec B must be in cache — multi-spec warmup should not skip subsequent specs"
        );
    });
}

#[test]
fn compile_warmup_empty_specs() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();
        let result = warmup_test_module::_compile_warmup(&[]);
        assert!(
            result.is_ok(),
            "compile_warmup with empty specs should return Ok(())"
        );
    });
}

#[test]
fn different_keys_parallel_compile() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();
        let barrier = Arc::new(std::sync::Barrier::new(2));

        let b1 = Arc::clone(&barrier);
        let h1 = std::thread::Builder::new()
            .stack_size(common::TEST_STACK_SIZE)
            .spawn(move || {
                b1.wait();
                let x = api::ones::<f32>(&[256]).sync().unwrap();
                let y = api::ones::<f32>(&[256]).sync().unwrap();
                let z = api::zeros::<f32>(&[256]).partition([128]).sync().unwrap();
                warmup_test_module::vector_add(z, &x, &y)
                    .generics(vec!["f32".into(), "128".into()])
                    .sync()
                    .unwrap();
            })
            .unwrap();

        let b2 = Arc::clone(&barrier);
        let h2 = std::thread::Builder::new()
            .stack_size(common::TEST_STACK_SIZE)
            .spawn(move || {
                b2.wait();
                let x = api::ones::<f32>(&[512]).sync().unwrap();
                let y = api::ones::<f32>(&[512]).sync().unwrap();
                let z = api::zeros::<f32>(&[512]).partition([256]).sync().unwrap();
                warmup_test_module::vector_add(z, &x, &y)
                    .generics(vec!["f32".into(), "256".into()])
                    .sync()
                    .unwrap();
            })
            .unwrap();

        h1.join().expect("thread 1 panicked");
        h2.join().expect("thread 2 panicked");

        let x_probe_8 = api::ones::<f32>(&[256]).sync().unwrap();
        let y_probe_8 = api::ones::<f32>(&[256]).sync().unwrap();
        let z_probe_8 = api::zeros::<f32>(&[256]).partition([128]).sync().unwrap();
        let z_spec_8 = z_probe_8.unpartition().spec().clone();

        let x_probe_32 = api::ones::<f32>(&[512]).sync().unwrap();
        let y_probe_32 = api::ones::<f32>(&[512]).sync().unwrap();
        let z_probe_32 = api::zeros::<f32>(&[512]).partition([256]).sync().unwrap();
        let z_spec_32 = z_probe_32.unpartition().spec().clone();

        let device_id = get_default_device();
        let gpu_name = get_gpu_name(device_id);
        let cv = get_compiler_version();
        let tv = get_cuda_toolkit_version();
        let key_8 = TileFunctionKey::builder("warmup_test_module", "vector_add")
            .generics(vec!["f32".into(), "128".into()])
            .stride_args(vector_add_stride_args())
            .spec_args(vec![
                ("z".to_string(), z_spec_8),
                ("x".to_string(), x_probe_8.spec().clone()),
                ("y".to_string(), y_probe_8.spec().clone()),
            ])
            .source_hash(warmup_test_module::_SOURCE_HASH)
            .gpu_name(gpu_name.clone())
            .compiler_version(cv.clone())
            .cuda_toolkit_version(tv.clone())
            .build();
        let key_32 = TileFunctionKey::builder("warmup_test_module", "vector_add")
            .generics(vec!["f32".into(), "256".into()])
            .stride_args(vector_add_stride_args())
            .spec_args(vec![
                ("z".to_string(), z_spec_32),
                ("x".to_string(), x_probe_32.spec().clone()),
                ("y".to_string(), y_probe_32.spec().clone()),
            ])
            .source_hash(warmup_test_module::_SOURCE_HASH)
            .gpu_name(gpu_name)
            .compiler_version(cv)
            .cuda_toolkit_version(tv)
            .build();
        assert!(contains_cuda_function(device_id, &key_8));
        assert!(contains_cuda_function(device_id, &key_32));
    });
}

// 4 threads race the same fresh kernel; single-flight dedup must fire exactly once.
// Proven by jit_compile_count() — broken dedup would compile up to 4 times.
#[test]
fn multi_thread_dedup_timing_evidence() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();
        // Prime fill kernel so only vector_add can move the counter below.
        let t_single = std::time::Instant::now();
        warmup_test_module::_compile_warmup(&[WarmupSpec::new(
            "vector_add",
            vec!["f32".into(), "128".into()],
        )
        .with_strides(vector_add_stride_args())
        .with_spec_args(vector_add_spec_args(256, 128))])
        .unwrap();
        let single_duration = t_single.elapsed();

        // tile=8 is fresh; tile=4 is taken by failed_warmup_does_not_poison_cache.
        let c_before_race = jit_compile_count();
        let n_threads = 4;
        let barrier = Arc::new(std::sync::Barrier::new(n_threads));
        let t_parallel = std::time::Instant::now();
        let handles: Vec<_> = (0..n_threads)
            .map(|_| {
                let barrier = Arc::clone(&barrier);
                std::thread::Builder::new()
                    .stack_size(common::TEST_STACK_SIZE)
                    .spawn(move || {
                        barrier.wait();
                        let x = api::ones::<f32>(&[256]).sync().unwrap();
                        let y = api::ones::<f32>(&[256]).sync().unwrap();
                        let z = api::zeros::<f32>(&[256]).partition([8]).sync().unwrap();
                        warmup_test_module::vector_add(z, &x, &y)
                            .generics(vec!["f32".into(), "8".into()])
                            .sync()
                            .unwrap();
                    })
                    .unwrap()
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        let parallel_duration = t_parallel.elapsed();
        let compiles_during_race = jit_compile_count() - c_before_race;

        eprintln!(
            "[dedup] single={:.1?}  parallel(4)={:.1?}  compiles_during_race={}",
            single_duration, parallel_duration, compiles_during_race
        );

        assert_eq!(
            compiles_during_race, 1,
            "single-flight dedup: 4 concurrent threads on the same fresh kernel \
             must trigger exactly ONE JIT compile, got {compiles_during_race}"
        );
    });
}

// ============================================================================
// execute_warmup tests
// ============================================================================

// Convenience: build a launch hook for vector_add with the given tile size.
fn vector_add_hook(tile: i32) -> LaunchHook {
    LaunchHook::new("vector_add", move |_| {
        let x = api::ones::<f32>(&[256]).sync()?;
        let y = api::ones::<f32>(&[256]).sync()?;
        let z = api::zeros::<f32>(&[256])
            .partition([tile as usize])
            .sync()?;
        let _ = warmup_test_module::vector_add(z, &x, &y)
            .generics(vec!["f32".into(), tile.to_string()])
            .sync()?;
        Ok(())
    })
}

#[test]
fn execute_warmup_runs_all_hooks() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();
        let report = warmup_test_module::_execute_warmup(vec![
            vector_add_hook(64),
            vector_add_hook(128),
        ])
        .expect("device init must succeed");

        assert_eq!(report.outcomes.len(), 2);
        assert!(report.all_ok(), "all hooks must succeed: {:?}", report);
        assert_eq!(report.ok_count(), 2);
        assert_eq!(report.err_count(), 0);
    });
}

#[test]
fn execute_warmup_continues_after_failure() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();
        let err_hook = LaunchHook::new(
            "vector_add",
            |_: &WarmupCtx| -> Result<(), cutile::error::Error> {
                Err(cutile::error::Error::KernelLaunch(
                    cutile::error::KernelLaunchError("simulated hook failure".into()),
                ))
            },
        );
        let report = warmup_test_module::_execute_warmup(vec![
            vector_add_hook(64),
            err_hook,
            vector_add_hook(128),
        ])
        .expect("device init must succeed");

        assert_eq!(report.outcomes.len(), 3);
        assert!(report.outcomes[0].result.is_ok());
        assert!(report.outcomes[1].result.is_err());
        assert!(
            report.outcomes[2].result.is_ok(),
            "third hook must still run after second returned Err (failure isolation)"
        );
    });
}

#[test]
fn execute_warmup_catches_panic() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();
        let panic_hook = LaunchHook::new(
            "vector_add",
            |_: &WarmupCtx| -> Result<(), cutile::error::Error> {
                panic!("simulated panic in warmup hook");
            },
        );
        let report = warmup_test_module::_execute_warmup(vec![panic_hook, vector_add_hook(64)])
            .expect("device init must succeed");

        assert_eq!(report.outcomes.len(), 2);
        assert!(
            report.outcomes[0].result.is_err(),
            "panicking hook must be reported as Err"
        );
        assert!(
            report.outcomes[1].result.is_ok(),
            "hook after panic must still run"
        );
    });
}

#[test]
fn execute_warmup_unknown_function_isolated() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();
        let bad_hook = LaunchHook::new(
            "nonexistent_kernel",
            |_: &WarmupCtx| -> Result<(), cutile::error::Error> {
                panic!("hook for unknown function must not be invoked");
            },
        );
        let report = warmup_test_module::_execute_warmup(vec![bad_hook, vector_add_hook(64)])
            .expect("device init must succeed");

        assert_eq!(report.outcomes.len(), 2);
        assert!(report.outcomes[0].result.is_err());
        assert_eq!(
            report.outcomes[0].jit_compiles, 0,
            "rejected hook must not have run JIT"
        );
        assert!(
            report.outcomes[1].result.is_ok(),
            "valid hook must still run after unknown-function hook was rejected"
        );
    });
}

#[test]
fn execute_warmup_after_compile_warmup_no_recompile() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();

        clear_kernel_cache();

        let spec_args = vector_add_spec_args(256, 64);

        warmup_test_module::_compile_warmup(&[WarmupSpec::new(
            "vector_add",
            vec!["f32".into(), "64".into()],
        )
        .with_strides(vector_add_stride_args())
        .with_spec_args(spec_args.clone())])
        .expect("compile_warmup failed");

        let c_before = jit_compile_count();
        let report = warmup_test_module::_execute_warmup(vec![vector_add_hook(64)])
            .expect("device init must succeed");
        let c_after = jit_compile_count();

        assert!(report.all_ok(), "hook must succeed: {:?}", report);
        assert_eq!(report.outcomes.len(), 1);
        assert_eq!(
            report.outcomes[0].jit_compiles, 0,
            "execute_warmup after compile_warmup with same key must NOT recompile \
             — counter delta means warmup and launch keys disagree"
        );
        assert_eq!(
            c_after, c_before,
            "global counter must not move either"
        );
    });
}

#[test]
fn execute_warmup_timing_recorded() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();
        let report = warmup_test_module::_execute_warmup(vec![vector_add_hook(64)])
            .expect("device init must succeed");
        assert_eq!(report.outcomes.len(), 1);
        assert!(
            report.outcomes[0].elapsed > std::time::Duration::ZERO,
            "elapsed must be recorded"
        );
    });
}
