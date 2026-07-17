/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! GPU tests for `cutile::warmup::execute_warmup` (runtime launch warmup).
//!
//! The per-hook counter deltas are the acceptance surface: they must identify
//! which cache path each hook took (in-memory hit / disk hit / cold compile),
//! per the interpretation table in `cutile::warmup`.
//!
//! All tests hold [`common::cache_test_lock`]: the counters are process-global
//! and each test needs a quiet window. Each test uses tile sizes unique to
//! this module so it controls whether its kernels are fresh.

use crate::common;
use cutile::api;
use cutile::error::kernel_launch_error;
use cutile::jit_cache::{self, FileSystemJitStore};
use cutile::prelude::{DeviceOp, PartitionOp};
use cutile::tile_kernel::TileKernel;
use cutile::warmup::{execute_warmup, WarmupHook};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[cutile::module]
mod execute_warmup_test_module {
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

/// A hook that launches `vector_add` with real tensors, production-style.
fn add_hook(label: &str, tile: usize) -> WarmupHook {
    let generics = vec!["f32".to_string(), tile.to_string()];
    WarmupHook::new(label, move |_cx| {
        let x = api::ones::<f32>(&[256]).sync()?;
        let y = api::ones::<f32>(&[256]).sync()?;
        let z = api::zeros::<f32>(&[256]).partition([tile]).sync()?;
        execute_warmup_test_module::vector_add(z, &x, &y)
            .generics(generics)
            .sync()?;
        Ok(())
    })
}

/// Pre-pays compilation for one specialization via the meta `.compile()`
/// terminal — the step `execute_warmup` reports about.
fn compile_terminal(tile: usize) {
    let z = api::meta::<f32>(&[256]).partition([tile]);
    let x = api::meta::<f32>(&[256]);
    let y = api::meta::<f32>(&[256]);
    execute_warmup_test_module::vector_add(z, x, y)
        .generics(vec!["f32".into(), tile.to_string()])
        .compile()
        .expect("meta .compile() warmup failed");
}

/// Primes the fill kernel behind `api::ones`/`zeros` so only `vector_add`
/// moves the counters inside the measured hooks.
fn prime_fill_kernel() {
    let _ = api::ones::<f32>(&[256]).sync().unwrap();
}

// After `.compile()` of the same specialization, the hook must be fully warm
// — (jit, disk, backend) == (0, 0, 0). This is also
// the key-compatibility regression between the meta compile path and a real
// production launch.
#[test]
fn after_compile_terminal_all_warm() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();
        prime_fill_kernel();
        compile_terminal(16);

        let report = execute_warmup(vec![add_hook("vector_add.f32.16", 16)]);

        assert!(report.all_ok(), "hook failed: {:?}", report.results);
        let r = &report.results[0];
        assert_eq!(
            (r.jit_compiles, r.disk_hits, r.backend_compiles),
            (0, 0, 0),
            "compile-warmed hook must be fully warm, got {r:?}"
        );
        assert!(r.fully_warm() && !r.recompiled());
        assert!(report.all_warm());
    });
}

// The hook's shape does not match what `.compile()` warmed, so the report
// must surface an expensive recompile — the "warmup missed" signal.
#[test]
fn key_mismatch_surfaces_recompile() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();
        jit_cache::disable();
        prime_fill_kernel();
        compile_terminal(32); // warm tile=32 …

        let report = execute_warmup(vec![add_hook("vector_add.f32.64", 64)]); // … launch a fresh tile=64

        assert!(report.all_ok(), "hook failed: {:?}", report.results);
        let r = &report.results[0];
        assert_eq!(
            (r.jit_compiles, r.disk_hits, r.backend_compiles),
            (1, 0, 1),
            "mismatched hook must cold-compile exactly once, got {r:?}"
        );
        assert!(r.recompiled() && !r.fully_warm());
        assert!(!report.all_warm());
    });
}

// R2: a failing hook and a panicking hook are both recorded and neither stops
// the hooks after them. Results keep the submission order.
#[test]
fn failures_are_isolated_and_ordered() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();
        prime_fill_kernel();

        let report = execute_warmup(vec![
            add_hook("ok.before", 8),
            WarmupHook::new("fails", |_cx| {
                Err(kernel_launch_error("intentional failure"))
            }),
            WarmupHook::new("panics", |_cx| panic!("intentional panic")),
            add_hook("ok.after", 8),
        ]);

        let labels: Vec<&str> = report.results.iter().map(|r| r.label.as_str()).collect();
        assert_eq!(labels, ["ok.before", "fails", "panics", "ok.after"]);
        assert_eq!(report.ok_count(), 2);
        assert_eq!(report.err_count(), 2);
        assert!(report.results[0].outcome.is_ok());
        assert!(
            report.results[3].outcome.is_ok(),
            "hooks after a failure must still run"
        );

        let panic_msg = format!("{:?}", report.results[2].outcome);
        assert!(
            panic_msg.contains("intentional panic"),
            "panic payload must be preserved: {panic_msg}"
        );
    });
}

// The documented meta guardrail: a hook that mistakenly launches with
// `api::meta` tensors panics when the kernel reads the device pointer; the
// panic must land in that hook's outcome and leave the rest of the batch alive.
#[test]
fn meta_tensor_misuse_is_isolated() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();
        prime_fill_kernel();

        let report = execute_warmup(vec![
            WarmupHook::new("meta.misuse", |_cx| {
                let z = api::meta::<f32>(&[256]).partition([16]);
                let x = api::meta::<f32>(&[256]);
                let y = api::meta::<f32>(&[256]);
                execute_warmup_test_module::vector_add(z, x, y)
                    .generics(vec!["f32".into(), "16".into()])
                    .sync()?;
                Ok(())
            }),
            add_hook("ok.after.misuse", 8),
        ]);

        assert!(report.results[0].outcome.is_err(), "meta launch must fail");
        assert!(
            report.results[1].outcome.is_ok(),
            "next hook must still run"
        );
    });
}

// With the disk layer explicitly off, a fresh specialization pays the
// backend and touches no disk counters.
#[test]
fn disk_disabled_counts_backend() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();
        jit_cache::disable();
        prime_fill_kernel();

        let report = execute_warmup(vec![add_hook("vector_add.f32.128", 128)]);

        assert!(report.all_ok(), "hook failed: {:?}", report.results);
        let r = &report.results[0];
        assert_eq!(
            (r.jit_compiles, r.disk_hits, r.backend_compiles),
            (1, 0, 1),
            "fresh kernel without a disk store must cold-compile, got {r:?}"
        );
    });
}

fn fresh_dir(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "cutile_execute_warmup_gpu_{label}_{}",
        std::process::id(),
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Env vars the orchestrator uses to drive its child processes. Read only by
/// this test — the library itself has no env switch.
const CHILD_DIR_ENV: &str = "CUTILE_TEST_EXECUTE_WARMUP_CHILD_DIR";
const CHILD_ROLE_ENV: &str = "CUTILE_TEST_EXECUTE_WARMUP_CHILD_ROLE";

// After a process restart with a warm disk cache, the hook reports
// (jit ≥ 1, disk ≥ 1, backend == 0) — the disk absorbed every `tileiras` run.
// Both roles run in fresh child processes so their in-memory cache is empty
// and every kernel the hook launches reaches the disk layer (same pattern as
// `jit_disk_cache::disk_cache_cross_process_hit`).
#[test]
fn disk_hit_after_restart() {
    if let (Some(dir), Some(role)) = (
        std::env::var_os(CHILD_DIR_ENV),
        std::env::var_os(CHILD_ROLE_ENV),
    ) {
        let is_reader = role.to_str() == Some("reader");
        common::with_test_stack(move || {
            jit_cache::enable(Arc::new(
                FileSystemJitStore::new(Path::new(&dir)).expect("open store"),
            ));
            let report = execute_warmup(vec![add_hook("vector_add.f32.64", 64)]);
            jit_cache::disable();

            assert!(report.all_ok(), "hook failed: {:?}", report.results);
            let r = &report.results[0];
            if is_reader {
                assert_eq!(
                    r.backend_compiles, 0,
                    "reader must not spawn tileiras — the writer stored every kernel: {r:?}"
                );
                assert!(r.disk_hits >= 1, "reader must be served from disk: {r:?}");
                assert!(
                    r.jit_compiles >= 1,
                    "reader's in-memory cache starts cold: {r:?}"
                );
                assert!(!r.recompiled());
            } else {
                assert_eq!(
                    r.disk_hits, 0,
                    "writer starts cold, nothing on disk yet: {r:?}"
                );
                assert!(
                    r.backend_compiles >= 1,
                    "writer must compile and store: {r:?}"
                );
                assert!(r.recompiled());
            }
        });
        return;
    }

    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();
        let dir = fresh_dir("xproc");
        let exe = std::env::current_exe().expect("current_exe");

        let run = |role: &str| {
            std::process::Command::new(&exe)
                .args([
                    "execute_warmup::disk_hit_after_restart",
                    "--exact",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(CHILD_DIR_ENV, &dir)
                .env(CHILD_ROLE_ENV, role)
                .output()
                .expect("spawn child test process")
        };

        for role in ["writer", "reader"] {
            let out = run(role);
            assert!(
                out.status.success(),
                "{role} process failed.\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr),
            );
        }

        let _ = std::fs::remove_dir_all(&dir);
    });
}
