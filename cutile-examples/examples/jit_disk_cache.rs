/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Example: opt-in JIT disk cache.
 *
 * Disk persistence is never enabled by default. An application turns it on by
 * constructing a store and installing it once, before any kernel compilation:
 *
 *     set_jit_store(Some(Box::new(FileSystemJitStore::new(path)?)));
 *
 * After that, a compiled kernel's cubin is written to `path`. A later process
 * (or, as shown here, the same process after the in-memory cache is dropped)
 * loads the cubin from disk and skips the `tileiras` compile step.
 *
 * Run with:
 *   cargo run -p cutile-examples --example jit_disk_cache
 */

use cuda_async::device_operation::*;
use cutile::api;
use cutile::error::Error;
use cutile::jit_store::FileSystemJitStore;
use cutile::tensor::*;
use cutile::tile_kernel::*;
use my_module::vector_add as vector_add_kernel;

#[cutile::module]
mod my_module {
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

fn launch() -> Result<(), Error> {
    let x = api::ones::<f32>(&[256]).sync()?;
    let y = api::ones::<f32>(&[256]).sync()?;
    let z = api::zeros::<f32>(&[256]).partition([256]).sync()?;
    vector_add_kernel(z, &x, &y)
        .generics(vec!["f32".into(), "256".into()])
        .sync()?;
    Ok(())
}

fn main() -> Result<(), Error> {
    // Choose where compiled cubins are cached. Override with CUTILE_CACHE_DIR.
    let cache_dir = std::env::var("CUTILE_CACHE_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("cutile_jit_cache_example"));
    println!("Using JIT disk cache at: {}", cache_dir.display());

    // Opt in to disk persistence. This is the only switch; there is no
    // environment variable and no default-on behavior.
    let store = FileSystemJitStore::new(cache_dir).expect("failed to create JIT store");
    cuda_async::jit_store::set_jit_store(Some(Box::new(store)));

    // First launch: nothing on disk yet, so this performs a full JIT compile
    // and then persists the cubin.
    let (c0, h0) = (jit_compile_count(), jit_disk_hit_count());
    launch()?;
    println!(
        "after first launch:  compiled +{}, disk hits +{}",
        jit_compile_count() - c0,
        jit_disk_hit_count() - h0
    );

    // Drop the in-memory cache to simulate a fresh process. The cubin is still
    // on disk, so the next launch is served from the disk cache without
    // re-running codegen.
    clear_kernel_cache();

    let (c1, h1) = (jit_compile_count(), jit_disk_hit_count());
    launch()?;
    println!(
        "after second launch: compiled +{}, disk hits +{}",
        jit_compile_count() - c1,
        jit_disk_hit_count() - h1
    );

    Ok(())
}
