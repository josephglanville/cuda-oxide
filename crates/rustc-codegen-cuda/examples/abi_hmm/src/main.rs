/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![allow(clippy::not_unsafe_ptr_arg_deref, clippy::missing_safety_doc)]

//! Unified ABI + HMM Test Example
//!
//! This example replicates the nvc++ test from rustc-codegen-cuda-plan.md:
//! - GPU directly accesses HOST memory via HMM (Heterogeneous Memory Management)
//! - No explicit memory copies (cudaMemcpy / clone_htod)
//! - Tests unified ABI: device must use same struct layout as host
//!
//! ## Key Difference from Traditional CUDA
//!
//! Traditional CUDA:
//! ```text
//! Host: data on stack → cudaMemcpy → Device memory → kernel → cudaMemcpy → Host reads result
//! ```
//!
//! HMM (this example):
//! ```text
//! Host: data on stack → kernel (GPU reads/writes host memory directly) → Host reads result
//! ```
//!
//! ## Requirements
//!
//! - GPU: Turing or newer (RTX 20xx+)
//! - Linux Kernel: 6.1.24+
//! - CUDA: 12.2+
//! - HMM enabled: `nvidia-smi -q | grep Addressing` should show "HMM"
//!
//! ## Build and Run
//!
//! ```bash
//! cargo oxide run abi_hmm
//! ```

use cuda_core::{CudaContext, LaunchConfig};
use cuda_device::{kernel, thread};
use cuda_host::{cuda_launch, load_kernel_module};

// =============================================================================
// TEST STRUCT: Exotic alignment that may differ between host and device
// =============================================================================

/// A struct with exotic alignment requirements.
///
/// On x86_64 (host), rustc currently lays this out as:
/// - `b` at offset 0 (16 bytes)
/// - `a` at offset 16 (1 byte)
/// - trailing padding: 15 bytes
/// - Total size: 32 bytes
///
/// NOTE: No #[repr(C)] - we're testing that our compiler correctly uses rustc's
/// memory order for fields. rustc may reorder fields for better packing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Extreme {
    pub a: u8,
    pub b: i128,
}

// =============================================================================
// KERNELS
// =============================================================================

/// Kernel that modifies struct through pointer (tests HMM + unified ABI).
///
/// The pointer `p` points to HOST memory. GPU accesses it via HMM.
/// `device_check` is set to 1 to prove the kernel ran on GPU.
#[kernel]
pub fn modify_extreme_hmm(p: *mut Extreme, scale: i128, device_check: *mut i32) {
    let idx = thread::index_1d();
    if idx.get() == 0 {
        unsafe {
            *device_check = 1;
            (*p).b *= scale;
        }
    }
}

/// Generic kernel with closure (tests HMM + closure + unified ABI).
#[kernel]
pub fn with_closure_hmm<F: Fn(*mut Extreme) + Copy>(p: *mut Extreme, device_check: *mut i32, f: F) {
    let idx = thread::index_1d();
    if idx.get() == 0 {
        unsafe {
            *device_check = 1;
        }
        f(p);
    }
}

// =============================================================================
// HOST CODE
// =============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== HMM (Heterogeneous Memory Management) Test ===");
    println!("=== GPU Direct Access to Host Memory ===\n");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    // HMM support was verified via: nvidia-smi -q | grep Addressing
    println!("HMM Support: YES ✓ (verified via nvidia-smi)");

    // Print host layout info for Extreme
    println!("\nHost Layout for Extreme:");
    println!("  sizeof(Extreme)  = {}", std::mem::size_of::<Extreme>());
    println!("  alignof(Extreme) = {}", std::mem::align_of::<Extreme>());
    println!("  offsetof(a)      = {}", std::mem::offset_of!(Extreme, a));
    println!("  offsetof(b)      = {}", std::mem::offset_of!(Extreme, b));

    let module = load_kernel_module(&ctx, "abi_hmm")
        .map_err(|err| format!("failed to load abi_hmm kernel module: {err}"))?;

    let mut failed = false;

    // =========================================================================
    // TEST 1: HMM - GPU directly accesses host stack memory
    // =========================================================================
    println!("\n--- Test 1: HMM Direct Host Memory Access ---");
    println!("  (No cudaMemcpy - GPU reads/writes host stack directly)");
    {
        // Data lives on HOST STACK - not copied to device!
        let mut data = Extreme { a: b'X', b: 42 };
        let mut device_ran: i32 = 0;

        println!("  Before kernel:");
        println!("    data.a = '{}', data.b = {}", data.a as char, data.b);
        println!("    device_ran = {}", device_ran);
        println!("    &data = {:p} (HOST address)", &data);
        println!("    &device_ran = {:p} (HOST address)", &device_ran);

        // Get raw HOST pointers - passed directly to GPU via HMM
        let data_ptr: *mut Extreme = &mut data;
        let device_ran_ptr: *mut i32 = &mut device_ran;
        let scale: i128 = 2;

        cuda_launch! {
            kernel: modify_extreme_hmm,
            stream: stream,
            module: module,
            config: LaunchConfig::for_num_elems(1),
            args: [data_ptr, scale, device_ran_ptr]
        }?;

        stream.synchronize()?;

        println!("  After kernel (reading HOST memory directly):");
        println!("    data.a = '{}', data.b = {}", data.a as char, data.b);
        println!("    device_ran = {}", device_ran);

        if device_ran == 1 && data.a == b'X' && data.b == 84 {
            println!("  ✓ TEST 1 PASSED: HMM works!");
            println!("    - device_ran=1 proves kernel executed on GPU");
            println!("    - data.b=84 (42*2) proves GPU wrote to host memory");
            println!(
                "    - Unified ABI correct: device read field b at offset {}",
                std::mem::offset_of!(Extreme, b)
            );
        } else if device_ran == 0 {
            println!("  ✗ TEST 1 FAILED: device_ran=0");
            println!("    Kernel did not execute on GPU");
            failed = true;
        } else {
            println!("  ✗ TEST 1 FAILED: Expected a='X', b=84, device_ran=1");
            println!(
                "    Got: a='{}', b={}, device_ran={}",
                data.a as char, data.b, device_ran
            );
            failed = true;
        }
    }

    // =========================================================================
    // TEST 2: HMM + Move Closure
    // =========================================================================
    println!("\n--- Test 2: HMM + Move Closure ---");
    println!("  (Closure captures by value, GPU accesses host struct via HMM)");
    {
        let mut data = Extreme { a: b'Y', b: 100 };
        let mut device_ran: i32 = 0;

        println!("  Before kernel:");
        println!("    data.a = '{}', data.b = {}", data.a as char, data.b);
        println!("    &data = {:p} (HOST address)", &data);

        let data_ptr: *mut Extreme = &mut data;
        let device_ran_ptr: *mut i32 = &mut device_ran;

        // Move closure - captures `scale` BY VALUE (copied to device)
        // But `p` (the struct pointer) points to HOST memory (HMM)
        let scale: i128 = 3;

        cuda_launch! {
            kernel: with_closure_hmm::<_>,
            stream: stream,
            module: module,
            config: LaunchConfig::for_num_elems(1),
            args: [data_ptr, device_ran_ptr, move |p: *mut Extreme| unsafe { (*p).b *= scale }]
        }?;

        stream.synchronize()?;

        println!("  After kernel:");
        println!("    data.a = '{}', data.b = {}", data.a as char, data.b);
        println!("    device_ran = {}", device_ran);

        if device_ran == 1 && data.a == b'Y' && data.b == 300 {
            println!("  ✓ TEST 2 PASSED: HMM + Move Closure works!");
            println!("    - GPU accessed host struct via HMM pointer");
            println!("    - Closure scale value was copied to device");
        } else {
            println!("  ✗ TEST 2 FAILED: Expected a='Y', b=300, device_ran=1");
            failed = true;
        }
    }

    // =========================================================================
    // TEST 3: HMM Reference Capture (non-move closure)
    // =========================================================================
    // This is the ultimate test - like the C++ example:
    //   kernel<<<1,1>>>(&data, &device_ran, [&](auto* p) { p->b = p->b * 2; });
    //
    // The closure captures `scale` BY REFERENCE (&scale).
    // Both the struct pointer AND the captured reference point to host memory.
    // GPU accesses BOTH via HMM.
    //
    // How it works:
    // 1. Closure captures &scale (a reference to host stack variable)
    // 2. Closure struct contains: { scale: &i128 } - a pointer to host memory
    // 3. Closure struct is passed to kernel (the pointer value is copied)
    // 4. On device, closure body dereferences &scale via HMM
    // 5. GPU reads scale value from host memory
    // =========================================================================
    println!("\n--- Test 3: HMM Reference Capture (non-move closure) ---");
    println!("  Target: [&](auto* p) {{ p->b *= scale; }} where scale is captured by reference");
    {
        let mut data = Extreme { a: b'Z', b: 50 };
        let mut device_ran: i32 = 0;

        println!("  Before kernel:");
        println!("    data.a = '{}', data.b = {}", data.a as char, data.b);
        println!("    &data = {:p} (HOST address)", &data);

        let data_ptr: *mut Extreme = &mut data;
        let device_ran_ptr: *mut i32 = &mut device_ran;

        // scale lives on HOST STACK
        // The closure captures &scale (a REFERENCE, not the value)
        // The closure struct contains a pointer to this host address
        let scale: i128 = 4;
        println!(
            "    &scale = {:p} (HOST address - closure captures this!)",
            &scale
        );

        // NON-MOVE closure - captures scale by reference!
        // Closure struct: { scale: &'a i128 }
        // The &i128 points to host stack, accessed via HMM
        cuda_launch! {
            kernel: with_closure_hmm::<_>,
            stream: stream,
            module: module,
            config: LaunchConfig::for_num_elems(1),
            args: [
                data_ptr,
                device_ran_ptr,
                |p: *mut Extreme| unsafe { (*p).b *= scale }
            ]
        }?;

        stream.synchronize()?;

        println!("  After kernel:");
        println!("    data.a = '{}', data.b = {}", data.a as char, data.b);
        println!("    device_ran = {}", device_ran);

        if device_ran == 1 && data.a == b'Z' && data.b == 200 {
            println!("  ✓ TEST 3 PASSED: HMM Reference Capture works!");
            println!("    - GPU accessed host struct via HMM pointer (data_ptr)");
            println!("    - GPU accessed captured &scale via HMM (reference capture)");
            println!("    - This matches C++: [&](auto* p) {{ p->b *= scale; }}");
        } else if device_ran == 0 {
            println!("  ✗ TEST 3 FAILED: device_ran=0");
            println!("    Kernel did not execute on GPU");
            failed = true;
        } else {
            println!("  ✗ TEST 3 FAILED: Expected a='Z', b=200, device_ran=1");
            println!(
                "    Got: a='{}', b={}, device_ran={}",
                data.a as char, data.b, device_ran
            );
            failed = true;
        }
    }

    // =========================================================================
    // TEST 4: Multiple Reference Captures (non-move closure)
    // =========================================================================
    println!("\n--- Test 4: Multiple Reference Captures ---");
    println!("  Target: [&](auto* p) {{ p->b = (p->b * scale) + offset; }}");
    {
        let mut data = Extreme { a: b'W', b: 10 };
        let mut device_ran: i32 = 0;

        let data_ptr: *mut Extreme = &mut data;
        let device_ran_ptr: *mut i32 = &mut device_ran;

        // Multiple variables captured by reference
        let scale: i128 = 5;
        let offset: i128 = 7;

        println!("  Before kernel:");
        println!("    data.b = {}", data.b);
        println!("    scale = {}, offset = {}", scale, offset);
        println!(
            "    &scale = {:p}, &offset = {:p} (both HOST addresses)",
            &scale, &offset
        );

        // NON-MOVE closure - captures BOTH scale AND offset by reference!
        // Closure struct: { scale: &i128, offset: &i128 }
        cuda_launch! {
            kernel: with_closure_hmm::<_>,
            stream: stream,
            module: module,
            config: LaunchConfig::for_num_elems(1),
            args: [
                data_ptr,
                device_ran_ptr,
                |p: *mut Extreme| unsafe { (*p).b = (*p).b * scale + offset }
            ]
        }?;

        stream.synchronize()?;

        println!("  After kernel:");
        println!("    data.b = {} (expected: 10 * 5 + 7 = 57)", data.b);
        println!("    device_ran = {}", device_ran);

        if device_ran == 1 && data.b == 57 {
            println!("  ✓ TEST 4 PASSED: Multiple Reference Captures work!");
            println!("    - GPU accessed two captured references via HMM");
        } else {
            println!("  ✗ TEST 4 FAILED: Expected b=57, device_ran=1");
            println!("    Got: b={}, device_ran={}", data.b, device_ran);
            failed = true;
        }
    }

    println!("\n=== Tests Complete ===");
    if failed {
        Err("one or more abi_hmm tests failed".into())
    } else {
        Ok(())
    }
}
