/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Unified Host Closure Kernel Example
//!
//! Tests passing closures from host to generic GPU kernels.
//!
//! Build and run with:
//!   cargo oxide run host_closure
//!
//! ## What This Tests
//!
//! 1. Generic kernel with `Fn` trait bound: `fn map<F: Fn(T) -> T + Copy>(...)`
//! 2. Closure with captures: `move |x| x * factor`
//! 3. Closure values passed through the call-site `cuda_launch!` macro
//! 4. Scalarization of closure captures to PTX parameters
//!
//! ## Expected Flow
//!
//! 1. Macro parses closure, extracts captures (e.g., `factor`)
//! 2. Backend sees `map<{closure@...}>` with closure type
//! 3. Closure captures scalarized to PTX params
//! 4. Host passes captures as kernel arguments

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::{cuda_launch, load_kernel_module};

#[derive(Clone, Copy)]
struct MixedCapture {
    small: u8,
    wide: f64,
    scale: f32,
}

// =============================================================================
// CLOSURE-ACCEPTING GENERIC KERNEL
// =============================================================================
/// Generic map kernel - applies a function to each element.
///
/// The key feature: `F` can be a closure with captures!
/// When called with `map(move |x| x * factor, ...)`, rustc monomorphizes
/// this to `map<f32, {closure capturing factor}>`.
#[kernel]
pub fn map<T: Copy, F: Fn(T) -> T + Copy>(f: F, input: &[T], mut out: DisjointSlice<T>) {
    let idx = thread::index_1d();
    let idx_raw = idx.get();
    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = f(input[idx_raw]);
    }
}

// =============================================================================
// HOST CODE
// =============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Unified Closure Kernel Test ===\n");

    // Initialize CUDA
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    // Test data
    const N: usize = 1024;
    let input_data: Vec<f32> = (0..N).map(|i| i as f32).collect();

    // Allocate device memory
    let input_dev = DeviceBuffer::from_host(&stream, &input_data)?;
    let mut output_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;

    let module = load_kernel_module(&ctx, "host_closure")
        .map_err(|err| format!("failed to load host_closure kernel module: {err}"))?;

    let mut failed = false;

    // =========================================================================
    // TEST 1: Closure with single capture
    // =========================================================================
    println!("Test 1: Single capture (scale by factor)");
    {
        let factor = 2.5f32;
        println!("  factor = {}", factor);
        println!("  N = {}", N);

        // THE KEY: closure with capture passed to generic kernel!
        // The macro should:
        // 1. Parse the closure
        // 2. Extract `factor` as a captured variable
        // 3. Pass `factor` as a kernel argument
        cuda_launch! {
            kernel: map::<f32, _>,
            stream: stream,
            module: module,
            config: LaunchConfig::for_num_elems(N as u32),
            args: [move |x: f32| x * factor, slice(input_dev), slice_mut(output_dev)]
        }?;

        // Verify
        let output_host = output_dev.to_host_vec(&stream)?;

        let errors = (0..N)
            .filter(|&i| (output_host[i] - input_data[i] * factor).abs() > 1e-5)
            .count();

        if errors == 0 {
            println!("  ✓ SUCCESS: All {} elements correct!\n", N);
        } else {
            println!("  ✗ FAILED: {} errors\n", errors);
            failed = true;
        }
    }

    // =========================================================================
    // TEST 2: Closure with multiple captures
    // =========================================================================
    println!("Test 2: Multiple captures (affine transform)");
    {
        let scale = 2.0f32;
        let offset = 10.0f32;
        println!("  scale = {}, offset = {}", scale, offset);

        // Reset output
        output_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;

        // Closure captures both `scale` and `offset`
        cuda_launch! {
            kernel: map::<f32, _>,
            stream: stream,
            module: module,
            config: LaunchConfig::for_num_elems(N as u32),
            args: [move |x: f32| x * scale + offset, slice(input_dev), slice_mut(output_dev)]
        }?;

        // Verify
        let output_host = output_dev.to_host_vec(&stream)?;

        let errors = (0..N)
            .filter(|&i| (output_host[i] - (input_data[i] * scale + offset)).abs() > 1e-5)
            .count();

        if errors == 0 {
            println!("  ✓ SUCCESS: All {} elements correct!\n", N);
        } else {
            println!("  ✗ FAILED: {} errors\n", errors);
            failed = true;
        }
    }

    // =========================================================================
    // TEST 3: Zero-capture closure (inline constant)
    // =========================================================================
    println!("Test 3: Zero captures (double each element)");
    {
        // Reset output
        output_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;

        // No captures - just inline computation
        cuda_launch! {
            kernel: map::<f32, _>,
            stream: stream,
            module: module,
            config: LaunchConfig::for_num_elems(N as u32),
            args: [|x: f32| x * 2.0, slice(input_dev), slice_mut(output_dev)]
        }?;

        // Verify
        let output_host = output_dev.to_host_vec(&stream)?;

        let errors = (0..N)
            .filter(|&i| (output_host[i] - input_data[i] * 2.0).abs() > 1e-5)
            .count();

        if errors == 0 {
            println!("  ✓ SUCCESS: All {} elements correct!\n", N);
        } else {
            println!("  ✗ FAILED: {} errors\n", errors);
            failed = true;
        }
    }

    // =========================================================================
    // TEST 4: Closure with 3 captures (polynomial transform)
    // =========================================================================
    println!("Test 4: Three captures (polynomial: a*x^2 + b*x + c)");
    {
        let a = 0.5f32;
        let b = 2.0f32;
        let c = 1.0f32;
        println!("  a = {}, b = {}, c = {}", a, b, c);

        // Reset output
        output_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;

        // Closure captures a, b, c (3 captures)
        cuda_launch! {
            kernel: map::<f32, _>,
            stream: stream,
            module: module,
            config: LaunchConfig::for_num_elems(N as u32),
            args: [move |x: f32| a * x * x + b * x + c, slice(input_dev), slice_mut(output_dev)]
        }?;

        // Verify: f(x) = 0.5*x^2 + 2*x + 1
        let output_host = output_dev.to_host_vec(&stream)?;

        let errors = (0..N)
            .filter(|&i| {
                let x = input_data[i];
                let expected = a * x * x + b * x + c;
                (output_host[i] - expected).abs() > 1e-3
            })
            .count();

        if errors == 0 {
            println!("  ✓ SUCCESS: All {} elements correct!\n", N);
        } else {
            println!("  ✗ FAILED: {} errors\n", errors);
            failed = true;
            // Debug: show first few mismatches
            for i in 0..N.min(5) {
                let x = input_data[i];
                let expected = a * x * x + b * x + c;
                println!("    [{i}]: got {}, expected {}", output_host[i], expected);
            }
        }
    }

    // =========================================================================
    // TEST 5: Closure with 4 captures (to ensure arbitrary count works)
    // =========================================================================
    println!("Test 5: Four captures (weighted sum: w1*x + w2 + w3*w4)");
    {
        let w1 = 3.0f32;
        let w2 = 5.0f32;
        let w3 = 2.0f32;
        let w4 = 7.0f32;
        println!("  w1 = {}, w2 = {}, w3 = {}, w4 = {}", w1, w2, w3, w4);

        // Reset output
        output_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;

        // Closure captures w1, w2, w3, w4 (4 captures)
        cuda_launch! {
            kernel: map::<f32, _>,
            stream: stream,
            module: module,
            config: LaunchConfig::for_num_elems(N as u32),
            args: [move |x: f32| w1 * x + w2 + w3 * w4, slice(input_dev), slice_mut(output_dev)]
        }?;

        // Verify: f(x) = 3*x + 5 + 2*7 = 3*x + 19
        let output_host = output_dev.to_host_vec(&stream)?;

        let errors = (0..N)
            .filter(|&i| {
                let x = input_data[i];
                let expected = w1 * x + w2 + w3 * w4;
                (output_host[i] - expected).abs() > 1e-3
            })
            .count();

        if errors == 0 {
            println!("  ✓ SUCCESS: All {} elements correct!\n", N);
        } else {
            println!("  ✗ FAILED: {} errors\n", errors);
            failed = true;
            for i in 0..N.min(5) {
                let x = input_data[i];
                let expected = w1 * x + w2 + w3 * w4;
                println!("    [{i}]: got {}, expected {}", output_host[i], expected);
            }
        }
    }

    // =========================================================================
    // TEST 6: Struct capture with internal padding
    // =========================================================================
    println!("Test 6: Struct capture with internal padding");
    {
        let mixed = MixedCapture {
            small: 9,
            wide: 2.25,
            scale: 0.75,
        };
        println!(
            "  small = {}, scale = {}, wide = {}",
            mixed.small, mixed.scale, mixed.wide
        );

        output_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;

        // Repro: `cuda_launch!` extracts only the root identifier `mixed`
        // from this closure, but rustc lowers the closure environment as
        // separate disjoint captures for the fields used below.
        cuda_launch! {
            kernel: map::<f32, _>,
            stream: stream,
            module: module,
            config: LaunchConfig::for_num_elems(N as u32),
            args: [
                move |x: f32| x * mixed.scale + mixed.wide as f32 + mixed.small as f32,
                slice(input_dev),
                slice_mut(output_dev)
            ]
        }?;

        let output_host = output_dev.to_host_vec(&stream)?;

        let errors = (0..N)
            .filter(|&i| {
                let x = input_data[i];
                let expected = x * mixed.scale + mixed.wide as f32 + mixed.small as f32;
                (output_host[i] - expected).abs() > 1e-5
            })
            .count();

        if errors == 0 {
            println!("  ✓ SUCCESS: All {} elements correct!\n", N);
        } else {
            println!("  ✗ FAILED: {} errors\n", errors);
            failed = true;
            for i in 0..N.min(5) {
                let x = input_data[i];
                let expected = x * mixed.scale + mixed.wide as f32 + mixed.small as f32;
                println!("    [{i}]: got {}, expected {}", output_host[i], expected);
            }
        }
    }

    println!("=== All Tests Complete ===");
    if failed {
        Err("one or more host_closure tests failed".into())
    } else {
        Ok(())
    }
}
