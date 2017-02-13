# gpgpu.js

This is a toy utility for this blog post.  It is designed to abstract away WebGL
as a graphics technology and make it behave more like a general-purpose
computing technology, like CUDA or OpenCL.

I will probably not be expanding on this utility unless there is substantial
interest.

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
