/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Runtime launch warmup: run caller-provided, per-kernel launch hooks once
//! with realistic shapes and data, ahead of serving traffic.
//!
//! The `.compile()` terminal (with [`crate::api::meta`] inputs) pre-pays a
//! kernel's *compilation* cost — frontend, `tileiras`, module load — without
//! allocating device memory or launching. What it cannot pay are the costs
//! that only a **real launch** incurs: real device allocation (caching
//! allocator pool growth), the launch-submission path's driver lazy-init,
//! occupancy/shared-memory setup, argument marshaling, and grid inference.
//! `execute_warmup` closes that gap: each [`WarmupHook`] wraps an ordinary,
//! production-identical kernel call with real tensors, and the orchestrator
//! runs them one by one, isolating failures and reporting per-hook timing and
//! compile activity.
//!
//! Recommended startup sequence:
//!
//! ```rust,ignore
//! cutile::jit_cache::enable_default()?;         // optional L2 disk cache
//! /* batch of meta `.compile()` calls */        // pre-pay compilation
//! let report = cutile::warmup::execute_warmup(vec![/* hooks */]);
//! assert!(report.all_ok() && report.all_warm());
//! ```
//!
//! # Interpreting the per-hook counters
//!
//! Every [`HookResult`] carries deltas of the process-global compile counters,
//! taken across the hook's execution. For a hook that exercises a single
//! kernel, the triple `(jit_compiles, disk_hits, backend_compiles)` identifies
//! the cache path it took:
//!
//! | jit | disk | backend | meaning |
//! |-----|------|---------|---------|
//! | 0 | 0 | 0 | in-memory hit — fully warm (ideal after `.compile()`) |
//! | 1 | 1 | 0 | L1 miss, disk hit — normal after a process restart |
//! | 1 | 0 | 1 | cold compile — first run ever, or the hook's key does **not** match the one `.compile()` warmed |
//! | 1 | 1 | 1 | disk entry rejected by the driver and recompiled (self-healing) |
//!
//! The counters are process-global: run warmup before serving traffic, or
//! concurrent launches from other threads will pollute the deltas. Helper
//! kernels launched inside a hook (`api::ones` compiles a fill kernel on first
//! use) are counted too — they are being warmed as well.

use std::panic::AssertUnwindSafe;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use crate::error::{Error, KernelLaunchError};
use crate::tile_kernel::{get_default_device, jit_compile_count, with_global_device_context};
use cuda_core::Stream;
use cutile_compiler::cuda_tile_runtime_utils::env_flag_enabled;
use cutile_compiler::jit_cache;

fn warmup_log_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag_enabled("CUTILE_JIT_LOG"))
}

macro_rules! warmup_log {
    ($($arg:tt)*) => {
        if warmup_log_enabled() {
            eprintln!("[cutile::warmup] {}", format!($($arg)*));
        }
    };
}

/// One runtime-warmup entry: a real launch closure plus a label for the report.
///
/// The closure body is an ordinary kernel call with **real tensors** — the
/// exact code a production request would run, executed ahead of time:
///
/// ```rust,ignore
/// WarmupHook::new("vector_add.f32.128", |_cx| {
///     let z = api::zeros::<f32>(&[1024]).partition([128]).sync()?;
///     let x = api::ones::<f32>(&[1024]).sync()?;
///     let y = api::ones::<f32>(&[1024]).sync()?;
///     kernels::vector_add(z, &x, &y)
///         .generics(vec!["f32".into(), "128".into()])
///         .sync()?;
///     Ok(())
/// })
/// ```
///
/// Do not use [`crate::api::meta`] tensors here: meta tensors never allocate,
/// and a real launch through one panics when the kernel reads its device
/// pointer. The panic is caught and recorded as that hook's failure.
pub struct WarmupHook {
    label: String,
    run: Box<dyn FnOnce(&WarmupCtx) -> Result<(), Error> + Send>,
}

impl WarmupHook {
    pub fn new(
        label: impl Into<String>,
        run: impl FnOnce(&WarmupCtx) -> Result<(), Error> + Send + 'static,
    ) -> Self {
        Self {
            label: label.into(),
            run: Box::new(run),
        }
    }

    /// The label this hook will carry in the report.
    pub fn label(&self) -> &str {
        &self.label
    }
}

/// Context handed to each hook: where to allocate and launch.
#[non_exhaustive]
#[derive(Debug, Clone, Copy)]
pub struct WarmupCtx {
    pub device_id: usize,
}

/// Result of one hook, including the compile-counter deltas measured across it.
#[derive(Debug)]
pub struct HookResult {
    pub label: String,
    /// `Err` from the hook, or a normalized panic. Failures are isolated:
    /// they never stop the remaining hooks.
    pub outcome: Result<(), Error>,
    pub elapsed: Duration,
    /// Frontend compiles (in-memory cache misses) during this hook.
    /// See [`crate::tile_kernel::jit_compile_count`].
    pub jit_compiles: u64,
    /// Compiles served from the on-disk cubin cache during this hook.
    /// See [`jit_cache::jit_disk_hit_count`].
    pub disk_hits: u64,
    /// `tileiras` backend compiles (the expensive event) during this hook.
    /// See [`jit_cache::jit_backend_compile_count`].
    pub backend_compiles: u64,
    /// Soft disk-cache I/O failures during this hook. Nonzero with an enabled
    /// store means the disk layer is broken and every miss pays the backend.
    pub disk_io_errors: u64,
}

impl HookResult {
    /// The hook hit the in-memory cache for everything it launched: zero
    /// compilation cost, only the runtime first-launch costs were paid.
    pub fn fully_warm(&self) -> bool {
        self.jit_compiles == 0
    }

    /// The hook triggered at least one expensive `tileiras` recompile. After a
    /// `.compile()` pass this signals a warmup/production key mismatch.
    pub fn recompiled(&self) -> bool {
        self.backend_compiles > 0
    }
}

/// Aggregate report for one `execute_warmup` run. Always returned in full —
/// per-hook failures land in [`HookResult::outcome`], never abort the batch.
#[derive(Debug, Default)]
pub struct WarmupReport {
    pub results: Vec<HookResult>,
    pub total_elapsed: Duration,
}

impl WarmupReport {
    pub fn ok_count(&self) -> usize {
        self.results.iter().filter(|r| r.outcome.is_ok()).count()
    }

    pub fn err_count(&self) -> usize {
        self.results.len() - self.ok_count()
    }

    pub fn all_ok(&self) -> bool {
        self.err_count() == 0
    }

    /// Every hook ran with zero frontend compiles — the warmup list and the
    /// preceding `.compile()` list match exactly.
    pub fn all_warm(&self) -> bool {
        self.results.iter().all(HookResult::fully_warm)
    }
}

/// Runs the hooks one by one on the calling thread's default device.
///
/// Always returns a report; a failing (or panicking) hook is recorded and the
/// remaining hooks still run. Whether any failure is fatal is the caller's
/// policy — check [`WarmupReport::all_ok`] / [`WarmupReport::all_warm`].
///
/// Hooks run **serially**. That is what makes the per-hook counter deltas
/// attributable: the counters are process-global, so this must not run
/// concurrently with other kernel launches (call it before serving traffic).
pub fn execute_warmup(hooks: Vec<WarmupHook>) -> WarmupReport {
    run_hooks(get_default_device(), hooks)
}

/// Like [`execute_warmup`], but on the device that owns `stream` — for
/// borrowed interop streams and per-device warmup (mirrors `sync`/`sync_on`
/// and `compile`/`compile_on`). Hooks that want the stream itself capture it.
pub fn execute_warmup_on(stream: &Arc<Stream>, hooks: Vec<WarmupHook>) -> WarmupReport {
    run_hooks(stream.device().ordinal(), hooks)
}

fn run_hooks(device_id: usize, hooks: Vec<WarmupHook>) -> WarmupReport {
    // Touch the device context up front so driver/context init is not billed
    // to the first hook's timing. A failure here is not fatal to the report:
    // each hook's own launch surfaces the real error in its outcome.
    let _ = with_global_device_context(device_id, |_| {});

    let ctx = WarmupCtx { device_id };
    let total_start = Instant::now();
    let mut results = Vec::with_capacity(hooks.len());

    for hook in hooks {
        let WarmupHook { label, run } = hook;
        let jit0 = jit_compile_count();
        let disk0 = jit_cache::jit_disk_hit_count();
        let backend0 = jit_cache::jit_backend_compile_count();
        let io0 = jit_cache::stats().io_errors;
        let t0 = Instant::now();

        let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| run(&ctx)))
            .unwrap_or_else(|payload| Err(panic_to_error(&label, payload)));

        let result = HookResult {
            elapsed: t0.elapsed(),
            jit_compiles: jit_compile_count().saturating_sub(jit0),
            disk_hits: jit_cache::jit_disk_hit_count().saturating_sub(disk0),
            backend_compiles: jit_cache::jit_backend_compile_count().saturating_sub(backend0),
            disk_io_errors: jit_cache::stats().io_errors.saturating_sub(io0),
            outcome,
            label,
        };

        warmup_log!(
            "execute_warmup: {} → {} ({:.1?}, jit+{} disk+{} backend+{}{})",
            result.label,
            if result.outcome.is_ok() { "ok" } else { "err" },
            result.elapsed,
            result.jit_compiles,
            result.disk_hits,
            result.backend_compiles,
            if result.disk_io_errors > 0 {
                format!(" io_errors+{}", result.disk_io_errors)
            } else {
                String::new()
            },
        );

        results.push(result);
    }

    WarmupReport {
        results,
        total_elapsed: total_start.elapsed(),
    }
}

/// Normalizes a hook panic into an [`Error`], extracting the usual string
/// payloads so the report stays readable.
fn panic_to_error(label: &str, payload: Box<dyn std::any::Any + Send>) -> Error {
    let message = if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "non-string panic payload".to_string()
    };
    Error::KernelLaunch(KernelLaunchError(format!(
        "panic in warmup hook '{label}': {message}"
    )))
}
