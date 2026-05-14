# host_closure

## Host Closures - Passing Closures from Host to Kernel

Tests passing closures from host code to generic GPU kernels. This enables functional programming patterns where behavior is parameterized at launch time.

## What This Example Does

- Defines a generic `map<T, F>` kernel that applies a function to each element
- Host code passes move and non-move closures with 0-4 captured variables
- Sync and async typed launches and `cuda_launch_*` macros all use the same
  closure kernel ABI
- The backend passes each closure environment as one opaque kernel argument and
  reconstructs the logical closure value at kernel entry

## Key Concepts Demonstrated

### Generic Kernel with Closure Parameter

```rust
#[kernel]
pub fn map<T: Copy, F: Fn(T) -> T + Copy>(f: F, input: &[T], mut out: DisjointSlice<T>) {
    let idx = thread::index_1d();
    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = f(input[idx.get()]);
    }
}
```

### Launching with Host Closure

```rust
let factor = 2.5f32;

module.map::<f32, _>(
    stream.as_ref(),
    LaunchConfig::for_num_elems(N as u32),
    move |x: f32| x * factor, // _ infers closure type
    &input_dev,
    &mut output_dev,
)?;
```

### How Closure Launch Works

1. The launch API preserves the closure expression and monomorphizes the typed
   closure wrapper for the inferred closure type.
2. Host launch code passes the closure environment as one CUDA argument. A
   `move` closure stores copied values in that environment; a non-move closure
   stores references to the captured host values.
3. The backend names the PTX entry with the same type-identity fingerprint used
   by host launch code.
4. Kernel lowering accepts the host-layout closure environment at the ABI
   boundary and rebuilds the logical closure value before calling the wrapper
   body.

## Build and Run

```bash
cargo oxide run host_closure
```

## Expected Output

```text
=== Unified Closure Kernel Test ===

Test 1: Single capture (scale by factor)
  ✓ SUCCESS: cuda_launch single capture
  ✓ SUCCESS: typed launch single capture
  ✓ SUCCESS: cuda_launch_async single capture
  ✓ SUCCESS: typed async launch single capture

...

Test 9: non-move reference captures
  ✓ SUCCESS: cuda_launch non-move reference captures
  ✓ SUCCESS: typed launch non-move reference captures
  ✓ SUCCESS: cuda_launch_async non-move reference captures
  ✓ SUCCESS: typed async launch non-move reference captures

=== All Tests Complete ===
```

## Hardware Requirements

- **Minimum GPU**: Any CUDA-capable GPU
- **CUDA Driver**: 11.0+

## Closure Tests

Each case runs through four launch surfaces: `cuda_launch!`, typed sync,
`cuda_launch_async!`, and typed async.

| Test | Closure                                      | Captures | Formula            |
|------|----------------------------------------------|---------:|:-------------------|
| 1    | `move \|x\| x * factor`                      |        1 | `x * 2.5`          |
| 2    | `move \|x\| x * scale + offset`              |        2 | `x * 2.0 + 10.0`   |
| 3    | `\|x\| x * 2.0`                              |        0 | `x * 2.0`          |
| 4    | `move \|x\| a*x*x + b*x + c`                 |        3 | `0.5*x² + 2*x + 1` |
| 5    | `move \|x\| w1*x + w2 + w3*w4`               |        4 | `3*x + 5 + 14`     |
| 6    | `move \|x\| x * scale - bias`                |        2 | `1.5*x - 4`        |
| 7    | `move \|x\| x * scale + wide + small`        |        3 | mixed-size fields  |
| 8    | `move \|x\| x * mixed.scale + ...`           |        1 | padded struct      |
| 9    | `\|x\| x * scale + bias`                     |        2 | borrowed captures  |

## The Closure Story

### CUDA C++ Approach

```cpp
float factor = 5.0f;
auto scale = [=](float x) { return x * factor; };
kernel<<<1, N>>>(scale, input, output);
// nvc++ handles closure serialization automatically
```

### cuda-oxide Approach

```rust
let factor = 5.0f32;
module.map::<f32, _>(
    stream.as_ref(),
    LaunchConfig::for_num_elems(N as u32),
    move |x: f32| x * factor,
    &input,
    &mut output,
)?;
// All closure launch APIs pass the closure environment as one kernel argument.
```

## Supported Closure Types

| Type     | Captures   | Callable       |
|----------|------------|----------------|
| `Fn`     | By ref     | Multiple times |
| `FnMut`  | By mut ref | Multiple times |
| `FnOnce` | By value   | Once           |

For GPU kernels, `FnOnce` with `Copy` bound is most common (closures are copied to each thread).

## Generated PTX

For `map::<f32, {closure capturing factor}>`:

```ptx
.entry map__typed_<type_id> (
    .param .align N .b8 %closure_env[SIZE], // Host-layout closure environment
    .param .u64 %input_ptr,
    .param .u64 %input_len,
    .param .u64 %out_ptr,
    .param .u64 %out_len
) {
    // Rebuild logical closure value from the opaque environment.
    // Load input
    ld.global.f32 %f_x, [%input_ptr + %offset];
    // Apply closure: x * factor
    mul.f32 %f_result, %f_x, %factor;
    // Store output
    st.global.f32 [%out_ptr + %offset], %f_result;
}
```

## Common Patterns

### Parameterized Transforms

```rust
let threshold = 0.5f32;
module.map::<f32, _>(
    stream.as_ref(),
    cfg,
    move |x: f32| if x > threshold { 1.0 } else { 0.0 },
    &input,
    &mut output,
)?;
```

### Runtime Configuration

```rust
fn launch_with_config(scale: f32, offset: f32, ...) {
    module.map::<f32, _>(
        stream.as_ref(),
        cfg,
        move |x: f32| x * scale + offset,
        &input,
        &mut output,
    )?;
}
```

### Composition

```rust
let f = |x: f32| x.sin();
let g = |x: f32| x * 2.0;
module.map::<f32, _>(
    stream.as_ref(),
    cfg,
    move |x: f32| g(f(x)), // sin(x) * 2
    &input,
    &mut output,
)?;
```
