/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CPU tests for the `execute_warmup` orchestrator: report structure, ordering,
//! and failure isolation. These hooks never touch the GPU, so the tests run
//! without a CUDA device (the orchestrator's best-effort device-context touch
//! is allowed to fail).

use cutile::error::kernel_launch_error;
use cutile::tile_kernel::get_default_device;
use cutile::warmup::{execute_warmup, HookResult, WarmupHook, WarmupReport};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

#[test]
fn empty_hooks_yield_empty_ok_report() {
    let report = execute_warmup(vec![]);
    assert!(report.results.is_empty());
    assert!(report.all_ok());
    assert!(report.all_warm());
    assert_eq!(report.ok_count(), 0);
    assert_eq!(report.err_count(), 0);
}

#[test]
fn report_preserves_labels_order_and_outcomes() {
    let report = execute_warmup(vec![
        WarmupHook::new("first.ok", |_cx| Ok(())),
        WarmupHook::new("second.err", |_cx| Err(kernel_launch_error("intentional"))),
        WarmupHook::new("third.ok", |_cx| Ok(())),
    ]);

    let labels: Vec<&str> = report.results.iter().map(|r| r.label.as_str()).collect();
    assert_eq!(labels, ["first.ok", "second.err", "third.ok"]);
    assert!(report.results[0].outcome.is_ok());
    assert!(report.results[1].outcome.is_err());
    assert!(report.results[2].outcome.is_ok());
    assert_eq!(report.ok_count(), 2);
    assert_eq!(report.err_count(), 1);
    assert!(!report.all_ok());
    // No GPU work happened, so no compile activity may be attributed.
    assert!(report.all_warm());
    assert!(report
        .results
        .iter()
        .all(|r| r.disk_hits == 0 && r.backend_compiles == 0 && r.disk_io_errors == 0));
}

#[test]
fn panic_is_normalized_and_isolated() {
    let ran_after_panic = Arc::new(AtomicUsize::new(0));
    let flag = Arc::clone(&ran_after_panic);

    let report = execute_warmup(vec![
        WarmupHook::new("panics.str", |_cx| panic!("boom")),
        WarmupHook::new("panics.string", |_cx| panic!("boom {}", 42)),
        WarmupHook::new("survivor", move |_cx| {
            flag.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }),
    ]);

    assert_eq!(report.err_count(), 2);
    assert_eq!(
        ran_after_panic.load(Ordering::Relaxed),
        1,
        "hooks after panics must run"
    );

    let first = format!("{:?}", report.results[0].outcome);
    assert!(
        first.contains("panics.str") && first.contains("boom"),
        "panic message must carry the hook label and payload: {first}"
    );
    let second = format!("{:?}", report.results[1].outcome);
    assert!(
        second.contains("boom 42"),
        "formatted panic payload must be preserved: {second}"
    );
}

#[test]
fn ctx_carries_default_device_and_hooks_run_once() {
    let seen = Arc::new(AtomicUsize::new(usize::MAX));
    let seen_in_hook = Arc::clone(&seen);

    let report = execute_warmup(vec![WarmupHook::new("ctx.probe", move |cx| {
        seen_in_hook.store(cx.device_id, Ordering::Relaxed);
        Ok(())
    })]);

    assert!(report.all_ok());
    assert_eq!(seen.load(Ordering::Relaxed), get_default_device());
}

#[test]
fn hook_label_accessor() {
    let hook = WarmupHook::new("my.label", |_cx| Ok(()));
    assert_eq!(hook.label(), "my.label");
}

#[test]
fn display_summary_and_aggregates() {
    let report = execute_warmup(vec![
        WarmupHook::new("alpha.ok", |_cx| Ok(())),
        WarmupHook::new("beta.err", |_cx| Err(kernel_launch_error("grid required"))),
    ]);

    let text = report.to_string();
    let mut lines = text.lines();

    // Header: counts and pluralized "hooks".
    let header = lines.next().unwrap();
    assert!(
        header.starts_with("execute_warmup: 2 hooks, 1 ok, 1 failed,"),
        "{header}"
    );
    // No GPU work happened, so the header carries no compile-activity suffix.
    assert!(!header.contains("jit+"), "{header}");

    // One line per hook, in order, each carrying status + label.
    let alpha = lines.next().unwrap();
    assert!(alpha.contains("ok") && alpha.contains("alpha.ok"), "{alpha}");
    let beta = lines.next().unwrap();
    assert!(beta.contains("FAILED") && beta.contains("beta.err"), "{beta}");
    // The failed line surfaces the error message.
    assert!(beta.contains("grid required"), "{beta}");
    assert!(lines.next().is_none(), "unexpected trailing output: {text}");

    // Aggregates: nothing compiled on the CPU path.
    assert_eq!(report.total_jit_compiles(), 0);
    assert_eq!(report.total_disk_hits(), 0);
    assert_eq!(report.total_backend_compiles(), 0);
    assert_eq!(report.total_disk_io_errors(), 0);
    assert!(!report.any_recompiled());
}

#[test]
fn display_empty_report_is_single_line() {
    let text = execute_warmup(vec![]).to_string();
    assert_eq!(text.lines().count(), 1);
    assert!(text.starts_with("execute_warmup: 0 hooks, 0 ok, 0 failed,"), "{text}");
}

// Constructed by hand (all report fields are public) so the failed-hook
// rendering can be asserted without a GPU: a hook that compiled and then failed
// must show its counter suffix *before* the error, while a clean failure shows
// only the error.
#[test]
fn display_failed_hook_shows_counters_before_error() {
    let report = WarmupReport {
        results: vec![
            HookResult {
                label: "compiled_then_failed".into(),
                outcome: Err(kernel_launch_error("Launch grid required.")),
                elapsed: Duration::from_millis(88),
                jit_compiles: 1,
                disk_hits: 0,
                backend_compiles: 1,
                disk_io_errors: 0,
            },
            HookResult {
                label: "clean_fail".into(),
                outcome: Err(kernel_launch_error("bad config")),
                elapsed: Duration::from_millis(1),
                jit_compiles: 0,
                disk_hits: 0,
                backend_compiles: 0,
                disk_io_errors: 0,
            },
        ],
        total_elapsed: Duration::from_millis(89),
    };

    let text = report.to_string();
    let compiled = text
        .lines()
        .find(|l| l.contains("compiled_then_failed"))
        .unwrap();
    let jit_pos = compiled.find("jit+1 disk+0 backend+1").unwrap();
    let err_pos = compiled.find("error: Launch grid required.").unwrap();
    assert!(
        jit_pos < err_pos,
        "counters must precede the error: {compiled}"
    );

    // A failure with no compile activity carries no counter suffix.
    let clean = text.lines().find(|l| l.contains("clean_fail")).unwrap();
    assert!(!clean.contains("jit+"), "{clean}");
    assert!(clean.contains("error: bad config"), "{clean}");

    assert_eq!(report.total_backend_compiles(), 1);
    assert!(report.any_recompiled());
}
