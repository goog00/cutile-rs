/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Micro-benchmark settling one question: after `.compile()` (compile_warmup),
//! where does a kernel's device-side load land — already paid by `.compile()`,
//! or deferred to the first real launch (which is what `execute_warmup` covers)?
//!
//! `.compile()` here runs the JIT, `cuModuleLoad`, and `cuModuleGetFunction`,
//! but never launches the kernel. Under the default `CUDA_MODULE_LOADING=LAZY`
//! (CUDA 12.2+), holding a `CUfunction` handle does not finalize the device-side
//! load — that happens on first launch (or an explicit `cuFuncLoad`, which this
//! codebase never calls). Under `EAGER`, `cuModuleLoad` loads everything up
//! front, so `.compile()` gets slower and the first launch loses that cost.
//!
//! Everything except the vector_add launch is done before the timed region:
//! inputs allocated, output buffers pre-filled, the fill kernel primed, and
//! vector_add already compiled. So `first_launch - steady_state` isolates the
//! kernel's one-time launch cost — and the deferred device load, if any.
//!
//! Run it twice and compare `first-launch overhead`:
//!
//! ```bash
//! CUDA_MODULE_LOADING=LAZY  cargo run --release -p cutile-examples --example warmup_module_load_probe
//! CUDA_MODULE_LOADING=EAGER cargo run --release -p cutile-examples --example warmup_module_load_probe
//! ```
//!
//! Prediction: LAZY shows a clear first-launch overhead that EAGER shrinks
//! toward the residual (occupancy/argument setup). If the overhead moves out of
//! the launch and into `.compile()` when you switch to EAGER, the device load
//! was being deferred under LAZY — i.e. `execute_warmup` is the thing paying it,
//! and a doc that says `.compile()` already paid "module load" is imprecise for
//! the default configuration.

use cutile::api;
use cutile::prelude::*;
use cutile::tile_kernel::jit_compile_count;
use std::time::Instant;

#[cutile::module]
mod probe_kernels {
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

const LEN: usize = 1 << 20; // 1M f32 = 4 MiB; large enough to launch, small enough to stay cheap
const TILE: usize = 256;
const ITERS: usize = 20;

fn main() {
    println!("=== execute_warmup module-load probe ===");
    println!(
        "CUDA_MODULE_LOADING = {}",
        std::env::var("CUDA_MODULE_LOADING")
            .unwrap_or_else(|_| "(unset → driver default; LAZY on CUDA 12.2+)".into())
    );

    let generics = || vec!["f32".to_string(), TILE.to_string()];

    // 1) compile_warmup only: JIT + cuModuleLoad + cuModuleGetFunction, no launch.
    let t = Instant::now();
    {
        let z = api::meta::<f32>(&[LEN]).partition([TILE]);
        let x = api::meta::<f32>(&[LEN]);
        let y = api::meta::<f32>(&[LEN]);
        probe_kernels::vector_add(z, x, y)
            .generics(generics())
            .compile()
            .expect("compile warmup");
    }
    let compile_ms = t.elapsed().as_secs_f64() * 1e3;
    println!(".compile() (meta, no launch)      : {compile_ms:9.3} ms");

    // 2) Do ALL allocation / fill-kernel work up front so none of it leaks into
    //    the timed launches: real inputs, and one output buffer per iteration
    //    (vector_add consumes its output partition, so we cannot reuse one).
    //    The first `ones`/`zeros` here also compiles+loads the fill kernel.
    let x = api::ones::<f32>(&[LEN]).sync().expect("alloc x");
    let y = api::ones::<f32>(&[LEN]).sync().expect("alloc y");
    let outputs: Vec<_> = (0..ITERS)
        .map(|_| {
            api::zeros::<f32>(&[LEN])
                .partition([TILE])
                .sync()
                .expect("alloc z")
        })
        .collect();

    // vector_add must already be warm; the timed loop should measure launch, not
    // compilation. We print the delta as a guard rather than asserting, so an
    // unexpected key mismatch shows up as data instead of a panic.
    let jit_before = jit_compile_count();

    // 3) Time each vector_add launch. Iteration 0 is the kernel's first-ever
    //    launch — the point at which a deferred device load would be paid.
    let mut times_ms = Vec::with_capacity(ITERS);
    for z in outputs {
        let t = Instant::now();
        probe_kernels::vector_add(z, &x, &y)
            .generics(generics())
            .sync()
            .expect("launch");
        times_ms.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let jit_delta = jit_compile_count() - jit_before;

    let first = times_ms[0];
    let steady = median(&times_ms[1..]);
    println!(
        "jit compiles during timed loop    : {jit_delta}   (want 0: launches measure load, not compile)"
    );
    println!("first vector_add launch           : {first:9.3} ms");
    println!(
        "steady-state launch (median of {})  : {steady:9.3} ms",
        ITERS - 1
    );
    println!(
        "first-launch overhead (first−steady): {:9.3} ms",
        first - steady
    );
    if jit_delta != 0 {
        println!(
            "  note: a compile happened inside the timed region — the first-launch\n\
             number includes JIT, not just device load. Meta and real keys likely differ."
        );
    }
}

fn median(xs: &[f64]) -> f64 {
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).expect("no NaNs in timings"));
    let mid = v.len() / 2;
    if v.len() % 2 == 0 {
        (v[mid - 1] + v[mid]) / 2.0
    } else {
        v[mid]
    }
}
