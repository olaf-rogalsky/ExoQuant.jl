# Exoquant.jl

A high-quality, native Julia port of the **Exoquant** color quantization and dithering library.

## Features
* **Native Julia**: Zero C dependencies.
* **Polymorphic**: Supports `RGB`, `RGBA`, `Gray`, `Lab`, etc., via `Colors.jl`.
* **Multithreaded**: Parallelized K-Means++ and Spatial Refinement.
* **Flexible Dithering**: Floyd-Steinberg, Randomized Rounding, or None.
* **Memory Efficient**: Returns `IndirectArrays`.

## Usage
```julia
using Exoquant, Colors

img = rand(RGB{N0f8}, 256, 256)
exq = Exoquantizer(256, iterations=2, dither=Exoquant.FloydSteinberg)

# Get an IndirectArray (Indexed Image)
indexed_img = quantize(exq, img)
```
