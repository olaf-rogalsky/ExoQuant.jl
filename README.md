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
## Credits
Original source in C by Dennis Ranke / exoticorn, see [https://github.com/exoticorn/exoquant](https://github.com/exoticorn/exoquant).
Vibe translated to julia with help of [Google Gemini](https://gemini.google.com/app).


## MIT License
Derived from ExoQuant v0.7

Copyright (c) 2004 Dennis Ranke

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
