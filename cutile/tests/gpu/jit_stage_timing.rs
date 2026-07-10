/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Gating experiment for the persistent on-disk kernel cache (issue #181).
//!
//! ## The question
//!
//! A persistent cache keyed on the **compiled bytecode** (call it `design B`,
//! the content-addressed key `sha256(.bc)`) can only skip work that happens
//! after serialization: the temp `.bc` write and the `tileiras` subprocess. It
//! can never skip the frontend, because the frontend is what produces the bytes
//! the key is computed from.
//!
//! A cache keyed on the **compilation inputs** (`design A`) can skip everything
//! down to `cuModuleLoad`, but its correctness rests on enumerating every input
//! that affects the cubin — a standing obligation, unlike design B's
//! correctness, which holds by construction.
//!
//! Design B is only worth its lower ceiling if `tileiras` dominates a cold
//! compile. This test measures whether it does.
//!
//! ## What it reports
//!
//! Per compile: module building (the `syn` re-parse of the kernel module and
//! its dependency closure), frontend, IR verification, bytecode serialization,
//! `.bc` write, `tileiras`, `cuModuleLoad`, `cuModuleGetFunction`.
//!
//! Then, per sample and aggregated:
//!
//! - **design B's capture rate** = `savings_B / savings_A` (a descriptive
//!   statistic; the verdict uses its minimum, never its median — see below),
//! - **Δ, design A's marginal saving over design B**, in absolute ms per
//!   kernel — the only measurable quantity that could ever justify design A's
//!   key-completeness burden,
//! - the `sha256` cost design B adds, measured over real bytecode sizes.
//!
//! ## Reading the output
//!
//! The verdict tests design B's premise — "`tileiras` dominates a cold
//! compile" — as a falsifiable predicate: in **every** sample, `tileiras` plus
//! the `.bc` write must be the **majority** (> 50%) of what a disk hit could
//! skip. The 50% is what "dominates" means, not a tuned constant, and the
//! every-sample quantifier is robust to the kernel mix: adding samples can
//! only tighten it, whereas a median moves whenever the corpus composition
//! does. (An earlier revision judged the median against an underived 70% and
//! was overridden the first time it fired — a gate overridden when it fires is
//! not a gate; hence the per-sample-minimum predicate here.)
//!
//! What no threshold on a benefit-side ratio can answer — whether Δ, roughly
//! the 55–70 ms frontend, justifies design A's standing correctness burden —
//! is a judgment, not a measurement. This test's job is to pin Δ down in
//! absolute terms; the A-vs-B decision itself (adopt design B, keep design A
//! as a possible future index layer) is recorded in the disk-cache PR (#193).
//!
//! `tileiras` timing is noisy and sits in the numerator of every capture
//! rate. When the minimum lands within a few points of 50%, re-run with
//! `CUTILE_JIT_TILEIRAS_SAMPLES=5` before drawing a conclusion.
//!
//! ## Running it
//!
//! ```bash
//! cargo test --release -p cutile --test gpu jit_stage_timing -- --ignored --nocapture
//! ```
//!
//! It is `#[ignore]`d so the default GPU suite, which builds in the dev profile,
//! does not run it. `--release` is mandatory, and the test refuses to run
//! without it. The
//! frontend is Rust code in `cutile-compiler`; `tileiras` is an external
//! binary. A debug build slows the first and not the second, which is exactly
//! the ratio this test exists to measure. See [`require_optimized_build`].
//!
//! The assertions are deliberately weak (they only catch a broken measurement);
//! the output table is the deliverable. The capture rate is reported, never
//! asserted — this test exists to produce that number, not to defend a
//! preferred answer.

use crate::common;
use cutile::api;
use cutile::core::f16;
use cutile::prelude::PartitionOp;
use cutile::tile_kernel::{
    jit_compile_count, take_last_jit_timings, CompileOptions, JitTimings, TileKernel,
};
use cutile_compiler::cuda_tile_runtime_utils::{
    get_gpu_name, tileiras_binary, TILEIRAS_SAMPLES_ENV,
};
use sha2::{Digest, Sha256};
use std::time::Instant;

/// Design B's premise as a falsifiable predicate: `tileiras` (plus the `.bc`
/// write) must be the majority of what a disk hit could skip, in every sample.
///
/// 0.50 is not a tuned constant — it is what "dominates" means. The predicate
/// is judged on the per-sample **minimum**: the sample corpus is an arbitrary
/// kernel mix, and a minimum only tightens as samples are added, while a
/// median moves whenever the mix does (adding a fourth gemm variant would
/// move the recorded run's median from 65.4% to 72.7% without touching the
/// system under test). A previous revision judged the median against 0.70, a
/// number with no derivation; it fired (65.4%) and was immediately overridden
/// by the absolute-time argument — a gate that is overridden when it fires is
/// not a gate.
const MAJORITY_CAPTURE: f64 = 0.50;

/// Refuses to report numbers from an unoptimized build.
///
/// Every frontend stage measured here (`syn`, `resolve`, `entrypt`, `typeck`,
/// `block`) is Rust code inside `cutile-compiler`, so `cargo test` without
/// `--release` measures it at `opt-level = 0`. `tileiras` is an external binary
/// and is unaffected. The ratio between the two — which is the entire point of
/// this test — is therefore skewed by however much the frontend is slowed down.
///
/// The workspace sets no `[profile.dev]` override, so the default `opt-level =
/// 0` applies. Measured difference: the frontend's fixed per-compile cost is
/// 236 ms in a debug build and 50 ms in a release build, while `tileiras` is
/// unchanged at ~66 ms. Four rounds of debug-build data led to the opposite
/// conclusion before this check existed.
fn require_optimized_build() {
    assert!(
        !cfg!(debug_assertions),
        "this test measures cutile-compiler's frontend against an external `tileiras` \
         binary. In a debug build the frontend is compiled at opt-level 0 and `tileiras` \
         is not, so every ratio below is meaningless. Re-run with:\n\n    \
         cargo test --release -p cutile --test gpu jit_stage_timing -- --ignored --nocapture\n"
    );
}

/// Distinct module name so these keys never collide with `warmup.rs` /
/// `warmup_bench.rs`, whose compiles would otherwise turn our misses into hits.
#[cutile::module]
mod jit_stage_module {
    use cutile::core::*;

    /// Trivial elementwise kernel: near-minimal frontend work.
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

    /// Reductions + broadcasts: moderate frontend work.
    /// Mirrors `cutile-benchmarks/benches/softmax.rs`.
    #[cutile::entry(unchecked_accesses = true)]
    unsafe fn softmax<const BM: i32, const BN: i32>(
        x: &Tensor<f16, { [-1, -1] }>,
        y: &mut Tensor<f16, { [BM, BN] }>,
    ) {
        let tile_x: Tile<f16, { [BM, BN] }> = load_tile_like(x, y);
        let tile_x_max: Tile<f16, { [BM] }> = reduce_max(tile_x, 1i32);
        let tile_x_max: Tile<f16, { [BM, BN] }> =
            tile_x_max.reshape(const_shape![BM, 1]).broadcast(y.shape());
        let num: Tile<f16, { [BM, BN] }> = exp(tile_x - tile_x_max);
        let denom: Tile<f16, { [BM] }> = reduce_sum(num, 1);
        let denom = denom.reshape(const_shape![BM, 1]).broadcast(y.shape());
        y.store(num / denom);
    }

    /// Loop + `mma`: the heaviest frontend of the three.
    /// Mirrors `cutile-benchmarks/benches/launch_overhead.rs`.
    #[cutile::entry(unchecked_accesses = true)]
    unsafe fn gemm<T: ElementType, const BM: i32, const BN: i32, const BK: i32>(
        z: &mut Tensor<T, { [BM, BN] }>,
        x: &Tensor<T, { [-1, -1] }>,
        y: &Tensor<T, { [-1, -1] }>,
        k: i32,
    ) {
        let part_x = x.partition(const_shape![BM, BK]);
        let part_y = y.partition(const_shape![BK, BN]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut tile_z: Tile<T, { [BM, BN] }> = z.load();
        for i in 0i32..(k / BK) {
            let tile_x = part_x.load([pid.0, i]);
            let tile_y = part_y.load([i, pid.1]);
            tile_z = mma(tile_x, tile_y, tile_z);
        }
        z.store(tile_z);
    }
}

struct Sample {
    label: String,
    t: JitTimings,
}

/// Compiles one specialization with `api::meta` inputs (no allocation, no
/// launch) and returns its stage breakdown.
///
/// Panics if the compile was a cache hit, which means this sample's cache key
/// equals an earlier one's. Silently dropping it would compute the aggregate
/// over the wrong set, so it is a hard error rather than a skip.
fn sample(label: impl Into<String>, compile: impl FnOnce()) -> Sample {
    let label = label.into();
    compile();
    let t = take_last_jit_timings().unwrap_or_else(|| {
        panic!(
            "{label}: no timings recorded, so this compile was a cache hit — its key \
             collides with an earlier sample's. Note that tensor *shape* is not part \
             of the key: `strides_hint` only records whether each stride is 1, and \
             `SpecializationBits` records power-of-2 divisibility clamped to 16. Vary \
             the generics or the compile options instead."
        )
    });
    Sample { label, t }
}

/// Cold-compile samples across three kernels of increasing frontend weight.
///
/// Each sample must have a distinct cache key. Tensor shape does **not** give
/// one: `TileFunctionKey` carries divisibility hints (clamped to 16) and
/// stride-is-one flags, not extents, because the cubin genuinely does not
/// depend on the extents — they are runtime kernel arguments. So the keys vary
/// by generics (tile shape, dtype) and compile options.
/// One throwaway compile, so the first measured sample is not the one that pays
/// the process's one-time costs.
///
/// The first `tileiras` invocation in a process pays for paging in the binary
/// and its shared libraries; it has been measured at 711 ms against 66 ms for
/// the identical kernel later in the same run. The first `cuModuleLoad`
/// likewise initializes the CUDA context. Neither belongs in a per-compile
/// figure. Its timings are taken and dropped.
fn warm_up() {
    let z = api::meta::<f32>(&[4096]).partition([32]);
    let x = api::meta::<f32>(&[4096]);
    let y = api::meta::<f32>(&[4096]);
    jit_stage_module::vector_add(z, x, y)
        .generics(vec!["f32".to_string(), "32".to_string()])
        .compile()
        .expect("warm-up .compile() failed");
    let _ = take_last_jit_timings();
}

fn collect_samples() -> Vec<Sample> {
    let mut out = Vec::new();

    // vector_add: vary the tile generic, as `warmup_bench.rs` does.
    for &(n, tile) in &[(4096usize, 64i32), (4096, 128), (4096, 256)] {
        let g = vec!["f32".to_string(), tile.to_string()];
        out.push(sample(format!("vector_add<f32,{tile}>"), || {
            let z = api::meta::<f32>(&[n]).partition([tile as usize]);
            let x = api::meta::<f32>(&[n]);
            let y = api::meta::<f32>(&[n]);
            jit_stage_module::vector_add(z, x, y)
                .generics(g)
                .compile()
                .expect("vector_add .compile() failed");
        }));
    }

    // softmax: `bn` is a generic, so varying the row length varies the key.
    // Tile shapes are ones `benches/softmax.rs` compiles.
    for &(m, n) in &[(4096usize, 1024usize), (4096, 2048), (4096, 4096)] {
        let (bm, bn) = (1usize, n);
        let g = vec![bm.to_string(), bn.to_string()];
        out.push(sample(format!("softmax<{bm},{bn}>"), || {
            let x = api::meta::<f16>(&[m, n]);
            let y = api::meta::<f16>(&[m, n]).partition([bm, bn]);
            unsafe {
                jit_stage_module::softmax(x, y)
                    .generics(g)
                    .compile()
                    .expect("softmax .compile() failed");
            }
        }));
    }

    // gemm: vary the tile shape, then the `max_divisibility` hint. Both tile
    // shapes are ones `benches/gemm.rs` compiles. `max_divisibility` is part of
    // the key and only *weakens* the alignment the compiler may assume, so it
    // cannot make an otherwise-valid kernel fail to compile.
    let n = 2048usize;
    let variants: [(usize, usize, usize, Option<i32>); 3] = [
        (128, 128, 64, None),
        (256, 256, 64, None),
        (128, 128, 64, Some(8)),
    ];
    for (bm, bn, bk, max_divisibility) in variants {
        let g = vec![
            "f16".to_string(),
            bm.to_string(),
            bn.to_string(),
            bk.to_string(),
        ];
        let div_label = match max_divisibility {
            Some(d) => format!(" maxdiv={d}"),
            None => String::new(),
        };
        out.push(sample(
            format!("gemm<f16,{bm},{bn},{bk}>{div_label}"),
            || {
                let z = api::meta::<f16>(&[n, n]).partition([bm, bn]);
                let x = api::meta::<f16>(&[n, n]);
                let y = api::meta::<f16>(&[n, n]);
                unsafe {
                    let launcher = jit_stage_module::gemm(z, x, y, n as i32).generics(g);
                    let launcher = match max_divisibility {
                        Some(d) => {
                            launcher.compile_options(CompileOptions::new().max_divisibility(d))
                        }
                        None => launcher,
                    };
                    launcher.compile().expect("gemm .compile() failed");
                }
            },
        ));
    }

    out
}

/// Milliseconds to `sha256` a buffer of `len` bytes, averaged over `iters`.
///
/// SHA-256 is data-independent in timing, so hashing a synthetic buffer of the
/// right size measures what hashing the real bytecode would cost.
fn sha256_ms(len: usize, iters: u32) -> f64 {
    let buf = vec![0xA5u8; len];
    let t0 = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(Sha256::digest(std::hint::black_box(&buf)));
    }
    t0.elapsed().as_secs_f64() * 1000.0 / f64::from(iters)
}

/// Milliseconds to read a `len`-byte file back from the filesystem, averaged
/// over `iters` after one warm-up read.
///
/// A disk-cache hit must read the cubin out of the store before
/// `cuModuleLoadData` can use it, under either key design. Measured warm, i.e.
/// served from the page cache, which is the steady state for a cache directory
/// that is being hit.
fn file_read_ms(len: usize, iters: u32) -> f64 {
    let path = std::env::temp_dir().join(format!("cutile_read_probe_{}", std::process::id()));
    std::fs::write(&path, vec![0xA5u8; len]).expect("write probe file");
    let _ = std::fs::read(&path).expect("warm the page cache");
    let t0 = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(std::fs::read(&path).expect("read probe file"));
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / f64::from(iters);
    let _ = std::fs::remove_file(&path);
    ms
}

/// The costs a design-A hit pays that the `design A` column above omits.
///
/// That column is `cuModuleLoad + cuModuleGetFunction` only. A real design-A
/// hit also has to:
///
/// 1. read the cubin out of the store (measured here),
/// 2. hash the canonical input key to address it (measured here; 512 bytes is a
///    generous size for the encoded `TileFunctionKey`),
/// 3. build that key from the launch arguments — `warmup_bench::cache_hit_path_cost`
///    measures this at roughly 5 µs, and it is already paid today on every launch,
/// 4. deserialize the `Validator`, which design A must persist because it skips
///    the frontend that produces it. **Not measured** — no serialization format
///    exists yet. `Validator` is a `Vec` of a few plain-data variants (element
///    type strings, `Vec<i32>` shapes), a few hundred bytes, so it belongs to
///    the same microsecond order as (1) and (2); that is an argument from size,
///    not a measurement.
///
/// Reported so the `design A` column is not mistaken for its true residual.
fn print_design_a_omitted_costs(mean_cubin_bytes: usize, design_a_load_ms: f64) {
    let read_ms = file_read_ms(mean_cubin_bytes, 200);
    let key_hash_ms = sha256_ms(512, 2000);
    println!("\nWhat the `design A` column above leaves out:");
    println!(
        "  read cubin from store       {read_ms:>7.3} ms   measured ({mean_cubin_bytes} B, page cache warm)"
    );
    println!(
        "  sha256 of the input key     {key_hash_ms:>7.3} ms   measured (512 B canonical encoding)"
    );
    println!("  build the input key             ~5 us   see warmup_bench::cache_hit_path_cost;");
    println!("                                          already paid today on every launch");
    println!("  deserialize the Validator          ?     NOT MEASURED - no format exists yet");
    println!(
        "  => design A residual >= {:.3} ms, against the {:.1} ms the column shows.",
        design_a_load_ms + read_ms + key_hash_ms,
        design_a_load_ms,
    );
    println!(
        "  This cannot change the choice: design B leaves the whole frontend behind, which\n  \
         is two orders of magnitude larger. Printed so the column is not read as exact."
    );
}

fn print_table(samples: &[Sample]) {
    println!(
        "\n{:<32} {:>9} {:>7} {:>8} {:>8} {:>8} {:>8} {:>7} {:>9} {:>10} {:>9}",
        "kernel",
        "total",
        "syn",
        "resolve",
        "entrypt",
        "typeck",
        "block",
        "tc_n",
        "tc_ms",
        "tileiras",
        "capture"
    );
    println!("{}", "-".repeat(140));
    for s in samples {
        let t = &s.t;
        println!(
            "{:<32} {:>9.1} {:>7.1} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>7} {:>9.1} {:>10.1} {:>8.1}%",
            s.label,
            t.total_ms,
            t.module_ast_ms,
            t.name_resolve_ms,
            t.compiler_new_ms,
            t.emit.typeck_ms,
            t.emit.block_ms,
            t.emit.typeck_calls,
            t.emit.typeck_total_ms,
            t.backend.tileiras_ms,
            t.capture_rate() * 100.0,
        );
    }
    println!(
        "(ms; syn+resolve = from_kernel; entrypt+typeck+block = frontend, where \
         typeck+block = compiler.compile().\n \
         tc_n = infer_function/infer_method calls per compile; tc_ms = total wall time in them.\n \
         `capture` = share of the achievable savings a content-addressed key gets)"
    );
}

/// Median of `xs`. Panics on empty input.
fn median(xs: &[f64]) -> f64 {
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in timings"));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// Design B's capture rate if `saved_ms` of frontend work were eliminated.
///
/// Removing frontend work shrinks the denominator (what design A could skip)
/// while leaving `tileiras` untouched, so design B's share rises.
fn capture_if_frontend_saved(t: &JitTimings, saved_ms: f64) -> f64 {
    let total = t.total_ms - saved_ms;
    let max_skippable = total - t.load_module_ms - t.load_function_ms;
    if max_skippable <= 0.0 {
        return 0.0;
    }
    (t.skippable_ms() / max_skippable).min(1.0)
}

/// Answers: **how does optimizing the frontend move Δ, design A's marginal
/// benefit over design B?**
///
/// It cannot move the verdict — removing frontend work only raises the share
/// `tileiras` holds — but every millisecond shaved off the frontend is a
/// millisecond removed from Δ, and Δ is the entire case for ever adding an
/// input-derived index layer on top of the content-addressed store. The
/// capture columns are kept for comparison with earlier recorded runs.
fn print_frontend_optimization_scenarios(samples: &[Sample]) {
    let scenarios: [(&str, fn(&JitTimings) -> f64); 5] = [
        ("today", |_| 0.0),
        ("+ memoized from_kernel", |t| t.module_build_ms()),
        ("+ all type inference gone", |t| {
            t.module_build_ms() + t.emit.typeck_total_ms
        }),
        ("+ entry-point gen gone", |t| {
            t.module_build_ms() + t.emit.typeck_total_ms + t.compiler_new_ms
        }),
        ("+ whole frontend gone (= design A)", |t| {
            t.module_build_ms() + t.frontend_ms()
        }),
    ];

    println!("\nHow does optimizing the frontend move Delta (design A's remaining edge)?");
    println!(
        "  {:<36} {:>10} {:>9} {:>9}",
        "scenario", "d median", "capture", "cap min"
    );
    println!("  {}", "-".repeat(68));
    for (label, saved) in scenarios {
        let rates: Vec<f64> = samples
            .iter()
            .map(|s| capture_if_frontend_saved(&s.t, saved(&s.t)))
            .collect();
        let deltas: Vec<f64> = samples
            .iter()
            .map(|s| {
                let t = &s.t;
                (t.total_ms - saved(t) - t.load_module_ms - t.load_function_ms - t.skippable_ms())
                    .max(0.0)
            })
            .collect();
        let med = median(&rates);
        let lo = rates.iter().cloned().fold(f64::INFINITY, f64::min);
        println!(
            "  {label:<36} {:>7.1} ms {:>8.1}% {:>8.1}%",
            median(&deltas),
            med * 100.0,
            lo * 100.0
        );
    }
    println!(
        "  Read as: frontend optimization is not a prerequisite for the #181 decision — it\n  \
         only shrinks what design A could still buy. The last row *is* design A by\n  \
         definition (d ~ 0)."
    );
}

// Excluded from the default GPU suite: `scripts/run_gpu_tests.sh` builds in the
// dev profile, where this test's numbers are meaningless (see
// `require_optimized_build`), and one large-tile GEMM alone spends ~17 s in
// `tileiras`. Run it explicitly:
//
//   cargo test --release -p cutile --test gpu jit_stage_timing -- --ignored --nocapture
#[test]
#[ignore = "requires --release; run explicitly, see the module docs"]
fn jit_stage_breakdown_gates_disk_cache_key_choice() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();

        require_optimized_build();

        let device_id = cutile::tile_kernel::get_default_device();
        println!("\n=== JIT stage breakdown — issue #181 gating experiment ===");
        println!("gpu      = {}", get_gpu_name(device_id));
        println!("tileiras = {}", tileiras_binary().display());
        println!(
            "samples  = {} tileiras run(s) per compile (set {} to raise it)",
            std::env::var(TILEIRAS_SAMPLES_ENV).unwrap_or_else(|_| "1".into()),
            TILEIRAS_SAMPLES_ENV,
        );

        // Discarded: the first tileiras spawn and the first cuModuleLoad in a
        // process both pay one-time costs that no per-compile figure should carry.
        warm_up();

        let c0 = jit_compile_count();
        let samples = collect_samples();
        let compiles = jit_compile_count() - c0;
        assert_eq!(
            compiles as usize,
            samples.len(),
            "every sample must be a real cold compile: expected {} compiles, saw {compiles}",
            samples.len()
        );

        print_table(&samples);

        // ── Aggregate ────────────────────────────────────────────────────────
        let n = samples.len() as f64;
        let sum = |f: fn(&JitTimings) -> f64| samples.iter().map(|s| f(&s.t)).sum::<f64>();

        let total = sum(|t| t.total_ms);
        let tileiras = sum(|t| t.backend.tileiras_ms);
        let write_bc = sum(|t| t.backend.write_bc_ms);
        let module_build = sum(|t| t.module_build_ms());
        let frontend = sum(|t| t.frontend_ms());
        let load_only = sum(|t| t.load_module_ms + t.load_function_ms);

        // Design B hits skip the `.bc` write and the `tileiras` subprocess.
        let savings_b = tileiras + write_bc;
        // Design A hits skip everything but the module load — its upper bound.
        let savings_a = total - load_only;
        // Time-weighted: what a warmup of *this exact kernel mix* would see.
        // Dominated by whichever sample compiles slowest, so read it alongside
        // the per-sample distribution below, never on its own.
        let weighted_capture = if savings_a > 0.0 {
            savings_b / savings_a
        } else {
            0.0
        };

        let pct = |x: f64| 100.0 * x / total;
        println!("\nAggregate over {} cold compiles:", samples.len());
        println!("  total                     {total:>9.1} ms");
        println!(
            "  module build (from_kernel){module_build:>9.1} ms  ({:>5.1}%)",
            pct(module_build)
        );
        println!(
            "  frontend                  {frontend:>9.1} ms  ({:>5.1}%)",
            pct(frontend)
        );
        println!(
            "  tileiras subprocess       {tileiras:>9.1} ms  ({:>5.1}%)",
            pct(tileiras)
        );
        println!(
            "  .bc temp write            {write_bc:>9.1} ms  ({:>5.1}%)",
            pct(write_bc)
        );
        println!(
            "  module load               {load_only:>9.1} ms  ({:>5.1}%)",
            pct(load_only)
        );
        println!("  {}", "-".repeat(56));
        println!(
            "  skippable by design B     {savings_b:>9.1} ms  ({:>5.1}%)",
            pct(savings_b)
        );
        println!(
            "  skippable by design A     {savings_a:>9.1} ms  ({:>5.1}%)",
            pct(savings_a)
        );

        // ── The specialization-independent floor ─────────────────────────────
        //
        // `from_kernel` and `generate_entry_point` do work that barely varies
        // with the kernel: re-parsing the dependency closure, re-resolving its
        // names. Approximate that fixed cost by the cheapest sample's, and
        // report it — it is paid once per *cache miss*, so a process compiling
        // N specializations of one module pays it N times, and it is exactly
        // the part a content-addressed key can never skip.
        let min_of = |f: fn(&JitTimings) -> f64| {
            samples
                .iter()
                .map(|s| f(&s.t))
                .fold(f64::INFINITY, f64::min)
        };
        let floor = min_of(|t| t.module_build_ms() + t.frontend_ms());
        println!("\nSpecialization-independent floor (redundant work, once per cache miss):");
        println!(
            "  syn parse    (min)        {:>9.1} ms",
            min_of(|t| t.module_ast_ms)
        );
        println!(
            "  name resolve (min)        {:>9.1} ms",
            min_of(|t| t.name_resolve_ms)
        );
        println!(
            "  entry point  (min)        {:>9.1} ms",
            min_of(|t| t.compiler_new_ms)
        );
        println!(
            "  type inference(min)       {:>9.1} ms",
            min_of(|t| t.emit.typeck_ms)
        );
        println!(
            "  block lowering(min)       {:>9.1} ms",
            min_of(|t| t.emit.block_ms)
        );
        println!(
            "  emit globals (min)        {:>9.1} ms",
            min_of(|t| t.emit.globals_ms)
        );
        println!("  {}", "-".repeat(40));
        // Where `block` actually goes. `tc_ms` counts *all* inference calls,
        // including the entry's, which runs before `compile_block` and is
        // already reported as `typeck`. Subtract it to get the inference that
        // happens inside `block` via `compile_inline`.
        let cheapest = samples
            .iter()
            .min_by(|a, b| a.t.total_ms.partial_cmp(&b.t.total_ms).unwrap())
            .expect("at least one sample");
        let c = &cheapest.t;
        let infer_in_block = (c.emit.typeck_total_ms - c.emit.typeck_ms).max(0.0);
        let nested_calls = c.emit.typeck_calls.saturating_sub(1);
        let other_lowering = (c.emit.block_ms - infer_in_block).max(0.0);
        println!(
            "\n  Where `block` goes, for the cheapest sample ({}):",
            cheapest.label
        );
        println!(
            "    inference inside block  {infer_in_block:>7.1} ms  ({:>4.1}% of block, {nested_calls} nested calls, {:.1} ms each)",
            100.0 * infer_in_block / c.emit.block_ms.max(f64::EPSILON),
            infer_in_block / (nested_calls.max(1) as f64),
        );
        println!(
            "    everything else         {other_lowering:>7.1} ms  ({:>4.1}% of block)",
            100.0 * other_lowering / c.emit.block_ms.max(f64::EPSILON),
        );
        println!(
            "    entry inference         {:>7.1} ms  (1 call, reported as `typeck`)",
            c.emit.typeck_ms
        );
        println!(
            "    => {:.1} ms per inference call on trivial DSL bodies. With only {} calls\n    \
             total, memoizing per-callee saves little; the per-call cost is the target.",
            c.emit.typeck_total_ms / (c.emit.typeck_calls.max(1) as f64),
            c.emit.typeck_calls,
        );
        println!("  {}", "-".repeat(40));
        println!(
            "  floor                     {floor:>9.1} ms  ({:>5.1}% of a mean cold compile)",
            100.0 * floor / (total / n)
        );
        println!(
            "  This is the ceiling on what optimizing the frontend could return,\n  \
             and the part design B is structurally unable to skip."
        );

        // ── Cost design B adds on every compile and every hit ─────────────────
        let mean_bc = sum(|t| t.backend.bytecode_len as f64) / n;
        let mean_cubin = sum(|t| t.backend.cubin_len as f64) / n;
        let hash_ms = sha256_ms(mean_bc.round() as usize, 200);
        println!("\nContent-addressed key overhead:");
        println!("  mean .bc size             {:>9.1} KiB", mean_bc / 1024.0);
        println!(
            "  mean cubin size           {:>9.1} KiB",
            mean_cubin / 1024.0
        );
        println!(
            "  sha256 over the .bc       {hash_ms:>9.3} ms  ({:>5.3}% of a mean cold compile)",
            100.0 * hash_ms / (total / n)
        );

        // ── Projection ───────────────────────────────────────────────────────
        let mean_total = total / n;
        let mean_b = (total - savings_b) / n + hash_ms;
        let mean_a = load_only / n;
        println!("\nProjected per-kernel cold-process JIT time (mean over samples):");
        println!("  today (no disk cache)     {mean_total:>9.1} ms");
        println!(
            "  design B (content-addr)   {mean_b:>9.1} ms   speedup {:>5.1}x",
            mean_total / mean_b.max(f64::EPSILON)
        );
        println!(
            "  design A (input-derived)  {mean_a:>9.1} ms   speedup {:>5.1}x   [upper bound]",
            mean_total / mean_a.max(f64::EPSILON)
        );
        println!(
            "  design B leaves {:.1} ms per kernel on the table.",
            mean_b - mean_a
        );

        // ── Verdict ──────────────────────────────────────────────────────────
        //
        // Design B's premise, tested per sample: `tileiras` + the `.bc` write
        // must be the majority of what a disk hit could skip. Judged on the
        // minimum, never the median: the corpus is an arbitrary kernel mix,
        // and the minimum is the only statistic here that adding samples
        // cannot relax. (Across four recorded runs the time-weighted rate
        // varied by 0.8 points and the median by 31.8 — the median lands on a
        // light kernel where `tileiras` sits only in the numerator.)
        let rates: Vec<f64> = samples.iter().map(|s| s.t.capture_rate()).collect();
        let med = median(&rates);
        let lo = rates.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = rates.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let unweighted = rates.iter().sum::<f64>() / n;

        println!("\nDesign B capture rate (share of design A's savings):");
        println!(
            "  time-weighted             {:>9.1}%   (what warming this exact kernel mix would \
             see; dominated by the slowest compile)",
            weighted_capture * 100.0
        );
        println!("  per-sample mean           {:>9.1}%", unweighted * 100.0);
        println!(
            "  per-sample median         {:>9.1}%   (descriptive only; moves with the kernel \
             mix)",
            med * 100.0
        );
        println!(
            "  per-sample min / max      {:>9.1}% / {:.1}%   <- verdict is judged on the min",
            lo * 100.0,
            hi * 100.0
        );

        // Which side dominates a cold compile is a fact about this run, not a
        // constant. Under a debug build the frontend led on every kernel; under
        // `--release` `tileiras` does. Derive it rather than asserting it.
        let frontend_bound: Vec<&Sample> = samples
            .iter()
            .filter(|s| s.t.skippable_ms() < s.t.max_skippable_ms() / 2.0)
            .collect();
        println!(
            "\n  {} of {} samples spend more of a cold compile in the frontend than in \
             `tileiras`.",
            frontend_bound.len(),
            samples.len()
        );
        if frontend_bound.is_empty() {
            println!(
                "  verdict: design B's premise holds on every sample — `tileiras` is the \
                 majority of\n  every cold compile (min capture {:.1}%, {:.1} points above the \
                 50% majority line).\n  What remains — whether Δ below justifies design A's \
                 key burden — is a judgment,\n  recorded with its revisit trigger in the \
                 disk-cache PR (#193).",
                lo * 100.0,
                (lo - MAJORITY_CAPTURE) * 100.0
            );
        } else {
            println!(
                "  verdict: design B's premise FAILS on {} of {} samples — the frontend, which \
                 a\n  content-addressed key can never skip, is the majority of their cold \
                 compile:",
                frontend_bound.len(),
                samples.len()
            );
            for s in &frontend_bound {
                println!(
                    "    {:<32} capture {:>5.1}%",
                    s.label,
                    s.t.capture_rate() * 100.0
                );
            }
            println!(
                "  For these kernels a disk hit leaves most of the wait in place. Weigh Δ \
                 below\n  against design A's correctness burden before settling on the \
                 content-addressed key."
            );
        }
        let n_samples = samples[0].t.backend.tileiras_samples;
        if n_samples <= 1 && (lo - MAJORITY_CAPTURE).abs() < 0.05 {
            println!(
                "  WARNING: the min sits within 5 points of the majority line and each \
                 tileiras\n  figure is a single run; re-run with {TILEIRAS_SAMPLES_ENV}=5 \
                 before trusting the verdict."
            );
        }

        // ── Δ: design A's marginal benefit over design B ─────────────────────
        //
        // The one quantity a different key could still buy, in the unit a user
        // actually waits: per kernel, once per cold process. An upper bound,
        // because `savings_a` is design A's upper bound (its true residual
        // additionally pays the store read, the key hash, and the unmeasured
        // Validator deserialization).
        let deltas: Vec<f64> = samples
            .iter()
            .map(|s| s.t.max_skippable_ms() - s.t.skippable_ms())
            .collect();
        let d_med = median(&deltas);
        let d_lo = deltas.iter().cloned().fold(f64::INFINITY, f64::min);
        let d_hi = deltas.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!(
            "\nDelta = design A's extra saving over design B (per kernel, once per cold \
             process):"
        );
        println!(
            "  median {d_med:.1} ms   min {d_lo:.1} ms   max {d_hi:.1} ms   (design B also \
             adds {hash_ms:.3} ms of sha256)"
        );
        println!(
            "  A cold process warming K kernels waits ~{d_med:.0}*K ms longer under design B \
             than\n  under design A: K = {:.0} crosses 100 ms, K = {:.0} crosses 1 s. This is \
             an upper\n  bound on what design A can still buy — its residual is itself a lower \
             bound\n  (Validator deserialization is unmeasured; no format exists yet).",
            (100.0 / d_med).ceil(),
            (1000.0 / d_med).ceil()
        );

        // `tileiras` noise propagates straight into every capture rate, since
        // it is the numerator. Surface the spread so a swing in the scenario
        // table below is never mistaken for a real change.
        let spread: Vec<f64> = samples
            .iter()
            .filter(|s| s.t.backend.tileiras_samples > 1)
            .map(|s| s.t.backend.tileiras_max_ms - s.t.backend.tileiras_min_ms)
            .collect();
        if spread.is_empty() {
            println!("\nWARNING: one `tileiras` run per compile. Its timing varies with machine");
            println!("  load and sits in the numerator of every capture rate below, so the");
            println!("  figures are indicative only. Re-run with {TILEIRAS_SAMPLES_ENV}=5 before");
            println!("  acting on any capture figure that lands near the 50% majority line.");
        } else {
            println!(
                "\ntileiras spread (max-min) across {} runs per compile: median {:.1} ms, max {:.1} ms",
                samples[0].t.backend.tileiras_samples,
                median(&spread),
                spread.iter().cloned().fold(0.0, f64::max),
            );
        }

        print_frontend_optimization_scenarios(&samples);

        // Absolute seconds, which is what a user actually waits. The capture
        // rate is a ratio and weights a 300 ms compile like a 17 s one.
        let worst = samples
            .iter()
            .max_by(|a, b| a.t.total_ms.partial_cmp(&b.t.total_ms).unwrap())
            .expect("at least one sample");
        let light = samples
            .iter()
            .min_by(|a, b| a.t.total_ms.partial_cmp(&b.t.total_ms).unwrap())
            .expect("at least one sample");
        println!("\nCold-process JIT time a disk-cache hit would leave (what a user waits):");
        println!(
            "  {:<32} {:>11} {:>13} {:>13}",
            "", "today", "design B", "design A"
        );
        for s in [worst, light] {
            println!(
                "  {:<32} {:>8.0} ms {:>10.0} ms {:>10.1} ms",
                s.label,
                s.t.total_ms,
                s.t.total_ms - s.t.skippable_ms(),
                s.t.load_module_ms + s.t.load_function_ms,
            );
        }
        println!(
            "  All three columns are time *remaining* after a hit, not time saved.\n  \
             design B leaves the frontend (it must run to produce the bytecode its key\n  \
             hashes); design A leaves only cuModuleLoad + cuModuleGetFunction, so its\n  \
             column is a lower bound — see the breakdown below.\n  \
             The capture rate treats these two rows equally; seconds do not."
        );

        print_design_a_omitted_costs(
            mean_cubin.round() as usize,
            light.t.load_module_ms + light.t.load_function_ms,
        );
        println!();

        // Sanity checks on the measurement itself, not on the result. The
        // capture rate is reported, never asserted: this test exists to produce
        // that number, not to enforce a preferred answer.
        assert!(
            tileiras > 0.0 && frontend > 0.0 && load_only > 0.0 && module_build > 0.0,
            "every stage must register nonzero time; a zero means the probe is broken"
        );
        assert!(
            samples.iter().all(|s| s.t.ir_text_ms < 1.0),
            "ir_text should be skipped entirely without print_ir / dump_mlir_dir"
        );
        assert!(
            savings_b <= savings_a + 1e-6,
            "design B cannot skip more than design A: {savings_b} > {savings_a}"
        );

        // `total_ms` must be built from the same `tileiras` measurement that
        // `tileiras_ms` reports, or the capture rate divides one by the other.
        // Two ways to get this wrong, both seen:
        //   1. the extra sampling runs leak into the outer timer;
        //   2. the outer timer sees the *first* run while `tileiras_ms` is the
        //      *median* across runs.
        // `BackendTimings::probe_overhead_ms` accounts for both.
        for s in &samples {
            let t = &s.t;
            let parts = t.module_build_ms()
                + t.frontend_ms()
                + t.backend.verify_ms
                + t.backend.serialize_ms
                + t.backend.write_bc_ms
                + t.backend.tileiras_ms
                + t.load_module_ms
                + t.load_function_ms;
            assert!(
                t.total_ms < parts * 1.25,
                "{}: total_ms {:.1} exceeds the sum of its parts {:.1} by more than 25% \
                 ({} tileiras sample(s), median {:.1} ms, min {:.1}, max {:.1}). Either the \
                 probe runs leaked into total_ms, or total_ms carries the first run while \
                 tileiras_ms carries the median. BackendTimings::probe_overhead_ms must \
                 cover both.",
                s.label,
                t.total_ms,
                parts,
                t.backend.tileiras_samples,
                t.backend.tileiras_ms,
                t.backend.tileiras_min_ms,
                t.backend.tileiras_max_ms,
            );
        }
    });
}
