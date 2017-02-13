# gpgpu.js

This is a toy utility for [this blog post](https://amoffat.github.io/held-karp-demo/).
It is designed to abstract away WebGL
as a graphics technology and make it behave more like a general-purpose
computing technology, like CUDA or OpenCL.

I will probably not be expanding on this utility unless there is substantial
interest.

## Benchmarks

Running the following kernel on 100M floats:

* **CPU Kernel:** `(num) -> num + Math.tan(Math.cos(Math.sin(num * num)))` 
* **GPU Kernel:** `dst = src + tan(cos(sin(src * src)));`

```
CPU: 6851.25ms
GPU Total: 1449.29ms
GPU Execution: 30.64ms
GPU IO: 1418.65ms
Theoretical Speedup: 223.59x
Actual Speedup: 4.73x
```


## Basic Usage

```coffeescript
root = $("#gpgpu")
engine = getEngine root

kernel = "dst = src + tan(cos(sin(src * src)));"
inputs = new Float32Array [1, 2, 3, 4, 5]

outputs = engine.execute kernel, inputs
```

## User-defined functions

```coffeescript
root = $("#gpgpu")
engine = getEngine root

inputs = new Float32Array [1, 2, 3, 4, 5]
preamble = """
float cube(float v) {
    return v * v * v;
}
"""
kernel = "dst = cube(src);"
outputs = engine.execute inputs, kernel, preamble
```

## Uniforms

```coffeescript
inputData = new Float32Array [1, 2, 3, 4, 5]
inputs = Storage.fromData engine, inputData

kernel = "dst = src * float(num);"
preamble = "uniform int num;"

comp = engine.createComputation inputData.length, kernel, preamble
outputs = comp.execute inputs, {num:["1i", 3]}
```
