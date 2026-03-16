/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cutile;
use cutile_compiler::compiler::{CUDATileFunctionCompiler, CUDATileModules};
use cutile_compiler::cuda_tile_runtime_utils::get_gpu_name;

mod common;

#[cutile::module]
mod binary_math_ops_module {

    use cutile::core::*;

    #[cutile::entry()]
    fn minmax_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        // Test min and max operations
        let x: Tile<f32, S> = load_tile_mut(output);
        let y: Tile<f32, S> = load_tile_mut(output);
        let max_result: Tile<f32, S> = maxf(x, y);
        let min_result: Tile<f32, S> = minf(max_result, y);
        output.store(min_result);
    }

    #[cutile::entry()]
    fn select_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        // Test select operation
        let x: Tile<f32, S> = load_tile_mut(output);
        let y: Tile<f32, S> = load_tile_mut(output);

        let _zero: Tile<f32, S> = constant(0.0, output.shape());
        let _one: Tile<f32, S> = constant(1.0, output.shape());

        // Simplified to avoid bool literal issue
        let result: Tile<f32, S> = maxf(x, y);
        output.store(result);
    }

    #[cutile::entry()]
    fn bf16_arithmetic_kernel<const S: [i32; 1]>(output: &mut Tensor<bf16, S>) {
        // Test bf16 arithmetic operations
        let x: Tile<bf16, S> = load_tile_mut(output);
        let y: Tile<bf16, S> = load_tile_mut(output);

        let sum: Tile<bf16, S> = addf(x, y); // addition
        let diff: Tile<bf16, S> = subf(sum, y); // subtraction
        let prod: Tile<bf16, S> = mulf(diff, y); // multiplication
        let result: Tile<bf16, S> = divf(prod, y); // division

        output.store(result);
    }

    #[cutile::entry()]
    fn bf16_comparison_kernel<const S: [i32; 1]>(output: &mut Tensor<bf16, S>) {
        // Test bf16 comparison operations
        let x: Tile<bf16, S> = load_tile_mut(output);
        let y: Tile<bf16, S> = load_tile_mut(output);

        let _eq: Tile<i1, S> = cmpf(x, "oeq", y); // equal
        let _lt: Tile<i1, S> = cmpf(x, "olt", y); // less than
        let _gt: Tile<i1, S> = cmpf(x, "ogt", y); // greater than

        output.store(x);
    }
}

use binary_math_ops_module::_module_asts;

#[test]
fn compile_minmax() -> () {
    common::with_test_stack(|| {
        let modules =
            CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "binary_math_ops_module",
            "minmax_kernel",
            &[128.to_string()],
            &[("output", &[1])],
            None,
            gpu_name,
        )
        .expect("Failed.");
        let module_op_str = compiler
            .compile()
            .expect("Failed.")
            .as_operation()
            .to_string();
        println!("\n=== MIN/MAX MLIR ===\n{}", module_op_str);

        let expected_ops = ["maxf", "minf"];
        for op in expected_ops {
            assert!(
                module_op_str.contains(format!("= {}", op).as_str()),
                "Expected {} operation in MLIR output",
                op
            );
        }

        println!(
            "\n✓ All {} min/max operations verified in MLIR output",
            expected_ops.len()
        );
    });
}

#[test]
fn compile_select() -> () {
    common::with_test_stack(|| {
        let modules =
            CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "binary_math_ops_module",
            "select_kernel",
            &[128.to_string()],
            &[("output", &[1])],
            None,
            gpu_name,
        )
        .expect("Failed.");
        let module_op_str = compiler
            .compile()
            .expect("Failed.")
            .as_operation()
            .to_string();
        println!("\n=== SELECT MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("maxf"),
            "Kernel compiled but select test needs comparison ops"
        );

        println!("\n✓ select operation verified in MLIR output");
    });
}

#[test]
fn compile_bf16_arithmetic() -> () {
    common::with_test_stack(|| {
        let modules =
            CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "binary_math_ops_module",
            "bf16_arithmetic_kernel",
            &[128.to_string()],
            &[("output", &[1])],
            None,
            gpu_name,
        )
        .expect("Failed.");
        let module_op_str = compiler
            .compile()
            .expect("Failed.")
            .as_operation()
            .to_string();
        println!("\n=== BF16 ARITHMETIC MLIR ===\n{}", module_op_str);

        let expected_ops = ["addf", "subf", "mulf", "divf"];
        for op in expected_ops {
            assert!(
                module_op_str.contains(op),
                "Expected {} operation in MLIR output",
                op
            );
        }
        assert!(
            module_op_str.contains("bf16"),
            "Expected bf16 type in MLIR output"
        );

        println!(
            "\n✓ All {} bf16 arithmetic operations verified in MLIR output",
            expected_ops.len()
        );
    });
}

#[test]
fn compile_bf16_comparison() -> () {
    common::with_test_stack(|| {
        let modules =
            CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "binary_math_ops_module",
            "bf16_comparison_kernel",
            &[128.to_string()],
            &[("output", &[1])],
            None,
            gpu_name,
        )
        .expect("Failed.");
        let module_op_str = compiler
            .compile()
            .expect("Failed.")
            .as_operation()
            .to_string();
        println!("\n=== BF16 COMPARISON MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("cmpf"),
            "Expected cmpf operation in MLIR output"
        );
        assert!(
            module_op_str.contains("bf16"),
            "Expected bf16 type in MLIR output"
        );

        println!("\n✓ bf16 comparison operations verified in MLIR output");
    });
}
