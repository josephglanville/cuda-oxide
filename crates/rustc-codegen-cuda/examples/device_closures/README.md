# Unified Device Closures Example

Demonstrates closure patterns in CUDA kernels using unified compilation.

## What This Tests

1. **Inline closures** - Closures defined and called within the kernel
2. **Closures with captures** - Closures that capture kernel parameters
3. **Closures passed to device functions** - Using `FnOnce`/`Fn` trait bounds

## Build & Run

```bash
cargo oxide run device_closures
```

## How Closures Work in GPU Code

### The Problem: `no_std` vs `std` MIR Differences

When you write:

```rust
fn apply_closure<F: FnOnce(u32) -> u32>(f: F, x: u32) -> u32 {
    f(x)
}

#[kernel]
pub fn test_closure_fnonce(mut out: DisjointSlice<u32>) {
    let triple = |x: u32| x * 3;
    let idx = thread::index_1d();
    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = apply_closure(triple, 7);
    }
}
```

Rustc generates **different MIR** depending on `no_std` vs `std`:

#### `no_std` Crate (Separate Kernel Crate Approach)

```text
bb1: {
    _14 = &mut _6;  // Create REFERENCE to closure
    _5 = <closure as core::ops::FnMut>::call_mut(move _14, move _13)
    //                                           ^^^^^^^^
    //                                           Reference passed
}
```

#### `std` Crate (Unified Compilation)

```text
bb1: {
    // NO reference created!
    _5 = <closure as std::ops::FnOnce>::call_once(const ZeroSized: {closure}, move _12)
    //                                            ^^^^^^^^^^^^^^^^^^^^^^^^^
    //                                            Value passed directly
}
```

### Why This Difference?

> **⚠️ UNVERIFIED**: The following explanation is based on observed behaviour and
> needs verification against a local checkout of the rustc source
> (`git clone https://github.com/rust-lang/rust.git`).

In `no_std` environments, `core::ops::FnOnce` appears to lack full intrinsic support for
`call_once`, so rustc desugars through `FnMut::call_mut(&mut self)` as a fallback. In `std`
environments, `FnOnce::call_once(self)` is explicitly lowered with the special `rust-call` ABI.

**TODO**: Verify this in rustc source - look for where closure trait method lowering differs
based on `std` vs `no_std` crate type. Relevant areas may include:
- `compiler/rustc_mir_build/src/thir/cx/expr.rs` (closure call lowering)
- `compiler/rustc_middle/src/ty/instance.rs` (instance resolution)
- Lang items for `fn_once`, `fn_mut`, `fn` traits

### The Closure Body

Both pipelines generate a **separate closure body function**:

```text
fn _RNCNv...test_closure_fnonce0B3_ {
    let mut _1: &'{erased} Closure(...);  // Expects REFERENCE
    let _2: u32;
    bb0: {
        _0 = Mul(copy _2, const 3_u32)
        return
    }
}
```

**Key observation**: The closure body expects `_1` to be a **reference** (`&Closure`).

- In `no_std`: We pass a reference → ✓ works
- In `std`: We pass a value → ✗ type mismatch!

### Understanding the Closure Type

```text
&'{erased} Closure(
    DefId(0:66 ~ device_closures[eb71]::...test_closure_fnonce::{closure#0}),
    [
        i8,                                              // Internal marker
        Binder { extern "RustCall" fn((u32,)) -> u32 },  // Signature
        ()                                               // Captures (empty = ZST)
    ]
)
```

For a closure WITH captures:

```text
[..., (&'{erased} u32,)]  // Captures tuple: holds &u32 (the captured factor)
```

### How Trait Calls Resolve to Closure Bodies

When the MIR says `<closure as FnMut>::call_mut(...)`, how does this become a call to the
closure body function? There is **no intermediate trait method** - the closure body IS the
implementation. Here's how the resolution works:

#### Step 1: Every Closure Gets a Unique Anonymous Type

```rust
let triple = |x: u32| x * 3;
```

Rustc creates a unique type: `{closure@src/main.rs:157:22: 157:30}`. Two identical closures
have different types.

#### Step 2: Rustc Auto-Implements `Fn*` Traits

For this closure type, rustc generates (conceptually):

```rust
impl FnOnce<(u32,)> for {closure@157:22} {
    fn call_once(self, args: (u32,)) -> u32 { /* points to closure body */ }
}
impl FnMut<(u32,)> for {closure@157:22} {
    fn call_mut(&mut self, args: (u32,)) -> u32 { /* points to SAME body */ }
}
```

The trait impl doesn't contain code - it points to the closure body function.

#### Step 3: The MIR Call is Monomorphic

```text
_5 = <{closure@src/main.rs:157:22} as FnMut<(u32,)>>::call_mut(...)
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^
      Concrete type, not generic - rustc knows exactly which function at compile time
```

#### Step 4: The Closure Body Has a Mangled Name

```text
fn _RNCNvCskdg9vxy9oFe_23device_closures46cuda_oxide_kernel_246e25db_test_closure_fnonce0B3_
   ^^^                                                                                            ^
   Rust symbol                                                                              closure index
```

The host-side identifier `cuda_oxide_kernel_246e25db_test_closure_fnonce` is the
hash-suffixed reserved form the `#[kernel]` macro emits; the byte count after
`device_closures` (`46`) is the length of that segment in the demangled symbol.

Encodes: `_R` (Rust) + `NC` (Nested Closure) + parent path + `0` (first closure) + crate hash.

#### Step 5: Our Backend Resolution

In `translate_closure_call()` and `collector.rs`, we:

```rust
// 1. Extract closure type from the call
if let TyKind::Closure(closure_def_id, substs) = closure_ty.kind() {

    // 2. Resolve to monomorphized instance
    let instance = Instance::resolve(tcx, closure_def_id, substs);

    // 3. Get the mangled symbol name
    let mangled = tcx.symbol_name(instance);  // "_RNCNv...0B3_"

    // 4. Generate direct call to that function
    generate_call(mangled, args);
}
```

**Key insight**: `<ClosureType as FnMut>::call_mut` resolves directly to `_RNCNv...closure_body`.
The trait syntax in MIR is just how rustc represents the call - by codegen time, it's a direct
function call with no indirection.

## The Fixes Required

### Fix 1: Collector - Discover Closure Bodies

Closures inside kernels have names like `cuda_oxide_kernel_<hash>_foo::{closure#0}`. We skip them
as kernel entry points, but we still need to collect them for compilation.

When the collector sees `FnOnce::call_once` / `FnMut::call_mut` / `Fn::call` with a closure
type argument, it extracts and collects the closure body:

```rust
// In collector.rs
if fn_name.contains("call_once") || fn_name.contains("call_mut") {
    for arg in args.iter() {
        if let TyKind::Closure(def_id, substs) = arg.as_type().kind() {
            // Add closure body to collection
            self.worklist.push_back(CollectedFunction {
                instance: closure_instance,
                export_name: mangled_name,
            });
        }
    }
}
```

### Fix 2: Translator - Create Reference for `call_once`

When translating `FnOnce::call_once(self, args)` in `std` mode, the self is passed by value
but the closure body expects a reference. We create a `MirRefOp`:

```rust
// In translate_closure_call()
let is_call_once = call_name.contains("call_once");

let self_arg = if is_call_once {
    // Create &mut self for the closure body
    let ref_op = MirRefOp::new(ctx, self_value, /*mutable=*/true);
    ref_op.result()
} else {
    // call_mut/call already pass reference
    self_value
};
```

## Why Only Closures Need This

Regular helper functions (like `#[device] fn helper(...)`) don't need special handling:

| Aspect             | Closures                              | Regular Functions    |
|--------------------|---------------------------------------|----------------------|
| Call mechanism     | Trait method dispatch (`Fn*::call*`)  | Direct function call |
| ABI                | `rust-call` (special tuple unpacking) | Standard Rust ABI    |
| Self parameter     | Implicit, varies by trait             | Explicit parameters  |
| `std` vs `no_std`  | Different MIR generation              | Same MIR             |

## Test Cases

| Test                          | Description                         | Pattern              |
|-------------------------------|-------------------------------------|----------------------|
| `test_inline_closure`         | `\|x\| x * 2` inside kernel         | Inline, no captures  |
| `scale_kernel`                | `input[i] * factor`                 | Explicit parameter   |
| `transform_kernel`            | `(x + offset) * scale`              | Multiple parameters  |
| `inline_with_param`           | Inline closure using kernel param   | Capture + inline     |
| `test_closure_constant`       | `\|\| 42`                           | No args, no captures |
| `test_closure_multi_arg`      | `\|a, b\| a + b`                    | Multiple args        |
| `test_closure_fnonce`         | Passed to `FnOnce` function         | Trait-based call     |
| `test_closure_capture_fnonce` | Captured + passed to fn             | Combined pattern     |
