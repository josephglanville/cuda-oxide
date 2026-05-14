# Closures and Generics

Rust's zero-cost abstractions -- generics, closures, and trait bounds -- work on
the GPU. This is one of cuda-oxide's most distinctive capabilities: you can write
a single generic kernel that operates on any numeric type, or pass a closure from
the host to customize GPU behavior, all without runtime overhead.

## Generic kernels

A kernel can be generic over types and trait bounds, just like any Rust function.
The compiler monomorphizes each instantiation into a separate PTX entry point:

```rust
use cuda_device::{kernel, thread, DisjointSlice};
use core::ops::Mul;

#[kernel]
pub fn scale<T: Copy + Mul<Output = T>>(
    factor: T,
    input: &[T],
    mut out: DisjointSlice<T>,
) {
    let idx = thread::index_1d();
    let i = idx.get();
    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = input[i] * factor;
    }
}
```

### PTX naming

Each monomorphization produces a distinct PTX entry point. The name is derived
from the function name and the concrete type parameters:

| Instantiation      | PTX entry point name |
|:-------------------|:---------------------|
| `scale::<f32>`     | `scale__f32`         |
| `scale::<i32>`     | `scale__i32`         |
| `scale::<MyType>`  | `scale__MyType`      |

### Launching generic kernels

When launching, specify the type parameter on the generated typed method. That
forces the concrete instantiation and lets the loader look up the matching PTX
entry point:

```rust
use cuda_core::LaunchConfig;

module
    .scale::<f32>(
        &stream,
        LaunchConfig::for_num_elems(N as u32),
        2.0f32,
        &input_dev,
        &mut output_dev,
    )
    .expect("Launch failed");
```

The generated method forces monomorphization of `scale::<f32>` so the
instantiation appears in the compiled PTX even though it is never called
directly on the CPU.

## Host closures as kernel arguments

cuda-oxide supports passing closures from the host to the GPU. This enables
powerful `map`-style patterns where the kernel's behavior is parameterized by
a function:

```rust
#[kernel]
pub fn map<F: Fn(i32) -> i32>(f: F, input: &[i32], mut out: DisjointSlice<i32>) {
    let idx = thread::index_1d();
    let i = idx.get();
    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = f(input[i]);
    }
}
```

Launch with a closure:

```rust
let factor = 3i32;
module
    .map::<_>(&stream, config, move |x| x * factor, &input_dev, &mut output_dev)
    .expect("Launch failed");
```

### How closure launch works

The launch path treats the closure environment as a single Rust value:

1. **Monomorphize the closure wrapper** -- the host launch code preserves the
   closure expression and forces a typed kernel wrapper instantiation for the
   inferred closure type.
2. **Pass one opaque argument** -- the closure environment crosses the CUDA
   kernel ABI as one argument in rustc's host layout. A `move` closure stores
   copied captures in that environment; a non-move closure stores references to
   the captured host values.
3. **Rebuild the logical closure** -- at kernel entry, lowering accepts the
   host-layout value and reconstructs the logical closure value used by the MIR
   body, including rustc's declaration-order field semantics.

```{figure} images/closure-capture-flow.svg
:align: center
:width: 100%

Closure kernel launch: the host passes the closure environment as one opaque
kernel argument. The device entry reconstructs the logical closure value from
that host-layout environment before running the kernel body.
```

### PTX naming for closures

Each closure kernel instantiation has a PTX entry name derived from rustc's
type-identity fingerprint for the kernel's generic argument tuple. Typed module
methods and lower-level `cuda_launch_*` macros compute that same name from the
concrete closure type, so they resolve to the same monomorphized entry:

| Instantiation                         | PTX entry point            |
|:--------------------------------------|:---------------------------|
| `map::<i32, {closure type A}>`        | `map__typed_<type-id-A>`   |
| `map::<i32, {closure type B}>`        | `map__typed_<type-id-B>`   |

## Move vs reference closures

The `move` keyword determines how captures are transferred to the GPU:

### Move closures (recommended default)

```rust
let factor = 3i32;
move |x| x * factor   // `factor` is copied to the GPU
```

- Each capture is **copied by value** into the closure environment, and that
  environment is copied to the device as one opaque kernel argument.
- The host value can be dropped after launch.
- Works on all systems -- no special hardware support needed.

### Reference closures (HMM)

```rust
let factor = 3i32;
|x| x * factor   // `factor` stays on host; GPU accesses via pointer
```

- The closure environment stores **references to host memory**.
- The GPU reads them through **Hardware-Managed Memory (HMM)** -- automatic
  page migration from host to device on access.
- The host variable **must remain alive** until the kernel completes.
- Requires HMM support (Turing+ GPU, Linux 6.1.24+, CUDA 12.2+).

### When to use which

| Scenario                                       | Use                                                               |
|:-----------------------------------------------|:------------------------------------------------------------------|
| Small scalar captures (numbers, booleans)      | `move` (zero-copy overhead)                                       |
| Large struct captures                          | `move` if the kernel reads it many times; HMM if rarely accessed  |
| Prototyping                                    | Either works; `move` is more portable                             |
| Shared mutable state between host and device   | Reference (HMM) -- but beware synchronization                     |

:::{tip}
When in doubt, use `move` closures. They are simpler to reason about, work
everywhere, and avoid the synchronization hazards of shared host/device memory.
:::

## In-kernel closures

Closures defined and called entirely within device code work with normal Rust
semantics -- no host closure ABI handling or argument scalarization is involved
because everything is already on the GPU:

```rust
#[kernel]
pub fn apply_transform(input: &[f32], mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();

    let transform = |x: f32| -> f32 {
        let clamped = if x < 0.0 { 0.0 } else if x > 1.0 { 1.0 } else { x };
        clamped * clamped
    };

    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = transform(input[idx.get()]);
    }
}
```

In-kernel closures are inlined by the compiler and have zero overhead. They are
useful for factoring logic within a kernel without introducing a separate device
function.

## Cross-crate kernels

Kernels can be defined in a library crate and launched from a binary crate:

```rust
// In lib crate `my_kernels`:
use cuda_device::{cuda_module, kernel, thread, DisjointSlice};

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(c_elem) = c.get_mut(idx) {
            *c_elem = a[i] + b[i];
        }
    }
}
```

```rust
// In binary crate:
use my_kernels::kernels;

let module = kernels::load(&ctx)?;
module
    .vecadd(&stream, config, &a, &b, &mut c)
    .expect("Launch failed");
```

The compiler handles cross-crate kernel discovery through the marker traits
generated by `#[kernel]`. The typed module resolves the PTX name at compile time
and caches the loaded function handle.

:::{tip}
For generic cross-crate kernels, the monomorphization happens in the **calling**
crate (where the concrete type is known), so the PTX is generated as part of
the binary's compilation.
:::
