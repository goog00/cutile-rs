/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Runtime launch warmup at service startup.
//!
//! The full warmup sequence a long-running service wants at startup:
//!
//! 1. `jit_cache::enable*` — opt into the on-disk cubin cache, so compilation
//!    survives restarts;
//! 2. meta `.compile()` per specialization — pre-pay all compilation with zero
//!    device allocation;
//! 3. `execute_warmup` — really launch the latency-critical kernels once with
//!    realistic shapes and data, paying the runtime first-launch costs (real
//!    allocation, launch-path driver init, occupancy/smem setup) that no cache
//!    can absorb.
//!
//! The report tells you whether the warmup actually warmed: a hook with
//! `jit_compiles > 0` compiled something step 2 should have covered — its
//! shapes/generics do not match the `.compile()` list. **Run this example
//! twice**: on the second run step 2 is served from disk (`backend == 0`
//! in its printout) and the hooks are fully warm either way.

use cutile::api;
use cutile::jit_cache::{self, FileSystemJitStore};
use cutile::prelude::*;
use cutile::warmup::{execute_warmup, WarmupHook};

#[cutile::module]
mod service_kernels {
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

fn main() {
    // 1) Opt into disk persistence. A fixed location, so the next run of this
    //    example finds what this run stored; real programs usually want
    //    `jit_cache::enable_default()`.
    let cache_dir = std::env::temp_dir().join("cutile-execute-warmup-example");
    println!("cache directory: {}", cache_dir.display());
    jit_cache::enable(std::sync::Arc::new(
        FileSystemJitStore::new(&cache_dir).expect("open cache directory"),
    ));

    // 2) Pre-pay compilation for every specialization the service uses:
    //    the same calls production makes, with meta inputs and `.compile()`.
    let before = jit_cache::stats();
    for tile in [128usize, 64] {
        let z = api::meta::<f32>(&[1024]).partition([tile]);
        let x = api::meta::<f32>(&[1024]);
        let y = api::meta::<f32>(&[1024]);
        service_kernels::vector_add(z, x, y)
            .generics(vec!["f32".into(), tile.to_string()])
            .compile()
            .expect("compile warmup");
    }
    let after = jit_cache::stats();
    println!(
        "compile warmup: disk hits {}, backend compiles absorbed by disk on the second run",
        after.hits - before.hits,
    );

    // 3) Really launch the latency-critical subset once, with real tensors.
    //
    // The fill kernel behind `ones`/`zeros` is warmed by this first real
    // allocation. Without it, the first hook's counters would attribute the
    // fill kernel's compile to that hook — counters bill every kernel compiled
    // during the hook, helper kernels included.
    let _prime = api::ones::<f32>(&[1024]).sync().expect("prime fill kernel");

    let report = execute_warmup(vec![
        WarmupHook::new("vector_add.f32.128", |_cx| {
            let x = api::ones::<f32>(&[1024]).sync()?;
            let y = api::ones::<f32>(&[1024]).sync()?;
            let z = api::zeros::<f32>(&[1024]).partition([128]).sync()?;
            service_kernels::vector_add(z, &x, &y)
                .generics(vec!["f32".into(), "128".into()])
                .sync()?;
            Ok(())
        }),
        WarmupHook::new("vector_add.f32.64", |_cx| {
            let x = api::ones::<f32>(&[1024]).sync()?;
            let y = api::ones::<f32>(&[1024]).sync()?;
            let z = api::zeros::<f32>(&[1024]).partition([64]).sync()?;
            service_kernels::vector_add(z, &x, &y)
                .generics(vec!["f32".into(), "64".into()])
                .sync()?;
            Ok(())
        }),
    ]);

    println!(
        "\nexecute_warmup: {}/{} ok in {:.1?}",
        report.ok_count(),
        report.results.len(),
        report.total_elapsed,
    );
    for r in &report.results {
        println!(
            "  {:<24} {:>9.1?}  jit+{} disk+{} backend+{}  {}",
            r.label,
            r.elapsed,
            r.jit_compiles,
            r.disk_hits,
            r.backend_compiles,
            match (&r.outcome, r.recompiled()) {
                (Err(e), _) => format!("FAILED: {e}"),
                (Ok(()), true) => "RECOMPILED — warmup list does not match".to_string(),
                (Ok(()), false) if r.fully_warm() => "fully warm".to_string(),
                (Ok(()), false) => "warm via disk".to_string(),
            },
        );
    }
    assert!(
        report.all_ok() && report.all_warm(),
        "warmup must be clean: {report:?}"
    );
    println!("\nfirst production request now pays steady-state latency only");
}
