module Exoquant

#using Colors, FixedPointNumbers, IndirectArrays, LinearAlgebra, Statistics, Random, Base.Threads
using Colors, IndirectArrays

export Exoquantizer, quantize, generate_palette, DitherMethod, InitMethod

@enum DitherMethod None FloydSteinberg RandomizedRounding
@enum InitMethod TreeSplit KMeansPlusPlus

struct Exoquantizer
    max_colors::Int
    iterations::Int
    dither::DitherMethod
    init::InitMethod
    
    Exoquantizer(n=256; iterations=2, dither=FloydSteinberg, init=KMeansPlusPlus) = 
        new(n, iterations, dither, init)
end

function quantize(exq::Exoquantizer, img::AbstractArray{T}; 
                  dither::DitherMethod=exq.dither, 
                  refine::Bool=(exq.iterations > 0)) where T<:Colorant
    
    palette = generate_palette(exq, img)
    
    if refine
        palette = refine_palette(exq, img, palette)
    end
    
    return map_to_palette(img, palette, dither)
end

# --- Initialization ---

function generate_palette(exq::Exoquantizer, img::AbstractArray{T}) where T<:Colorant
    if exq.init == KMeansPlusPlus
        return kmeans_pp_init(img, exq.max_colors)
    else
        return tree_split_init(img, exq.max_colors)
    end
end

function kmeans_pp_init(img::AbstractArray{T}, k::Int) where T
    pixels = vec(img)
    n = length(pixels)
    k = min(k, n)
    
    centroids = T[pixels[rand(1:n)]]
    distances = fill(Inf, n)

    for _ in 2:k
        latest_centroid = centroids[end]
        @threads for i in 1:n
            d = colordiff(pixels[i], latest_centroid)
            if d < distances[i]
                distances[i] = d
            end
        end

        weights = distances .^ 2
        total_weight = sum(weights)
        if total_weight == 0 break end
        
        target = rand() * total_weight
        cumulative = 0.0
        for i in 1:n
            cumulative += weights[i]
            if cumulative >= target
                push!(centroids, pixels[i])
                break
            end
        end
    end
    return centroids
end

function tree_split_init(img::AbstractArray{T}, k::Int) where T
    pixels = unique(vec(img))
    if length(pixels) <= k return pixels end
    step = length(pixels) ÷ k
    return pixels[1:step:end][1:k]
end

# --- Refinement ---

function refine_palette(exq::Exoquantizer, img, palette::Vector{T}) where T
    refined = copy(palette)
    n_colors = length(palette)
    n_threads = nthreads()

    for _ in 1:exq.iterations
        thread_sums = [fill(zero(RGB{Float64}), n_colors) for _ in 1:n_threads]
        thread_counts = [fill(0, n_colors) for _ in 1:n_threads]
        
        @threads for i in eachindex(img)
            tid = threadid()
            p = img[i]
            c = RGB(p)
            idx = argmin(map(entry -> colordiff(c, RGB(entry)), refined))
            thread_sums[tid][idx] += c
            thread_counts[tid][idx] += 1
        end
        
        final_sums = sum(thread_sums)
        final_counts = sum(thread_counts)
        
        for i in 1:n_colors
            if final_counts[i] > 0
                refined[i] = T(final_sums[i] / final_counts[i])
            end
        end
    end
    return refined
end

# --- Mapping & Dithering ---

function map_to_palette(img::AbstractArray{T}, palette, method::DitherMethod) where T
    if method == None
        return map_nearest_parallel(img, palette)
    elseif method == RandomizedRounding
        return dither_randomized_parallel(img, palette)
    elseif method == FloydSteinberg
        return dither_floyd_steinberg(img, palette)
    end
end

function map_nearest_parallel(img, palette)
    indices = Matrix{UInt32}(undef, size(img)...)
    @threads for i in eachindex(img)
        indices[i] = UInt32(argmin([colordiff(img[i], c) for c in palette]))
    end
    return IndirectArray(indices, palette)
end

function dither_randomized_parallel(img, palette)
    indices = Matrix{UInt32}(undef, size(img)...)
    @threads for i in eachindex(img)
        noise = (rand() - 0.5) * 0.05
        p = img[i]
        noisy_p = RGB(clamp(p.r + noise, 0, 1), clamp(p.g + noise, 0, 1), clamp(p.b + noise, 0, 1))
        indices[i] = UInt32(argmin([colordiff(noisy_p, c) for c in palette]))
    end
    return IndirectArray(indices, palette)
end

function dither_floyd_steinberg(img::AbstractArray{T}, palette) where T
    R, C = size(img)
    indices = Matrix{UInt32}(undef, R, C)
    # Convert image to RGB Float for error diffusion
    work_img = RGB{Float64}.(img)
    pal_rgb = RGB{Float64}.(palette)
    
    for c in 1:C, r in 1:R
        old_pixel = work_img[r, c]
        best_idx = argmin([colordiff(old_pixel, p) for p in pal_rgb])
        indices[r, c] = best_idx
        
        quant_error = old_pixel - pal_rgb[best_idx]
        
        # Diffuse
        if c < C;          work_img[r, c+1] += quant_error * (7/16); end
        if r < R && c > 1; work_img[r+1, c-1] += quant_error * (3/16); end
        if r < R;          work_img[r+1, c] += quant_error * (5/16); end
        if r < R && c < C; work_img[r+1, c+1] += quant_error * (1/16); end
    end
    return IndirectArray(indices, palette)
end

end # module
