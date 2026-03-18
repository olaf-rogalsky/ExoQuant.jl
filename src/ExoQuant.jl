module ExoQuant

using Colors, IndirectArrays

export quantize, DitherMethod, InitMethod
using Base.Threads: @spawn

# helper functions
@inline color_dist(c1::Oklab, c2::Oklab) = @fastmath (c1.l - c2.l)^2 + (c1.a - c2.a)^2 + (c1.b - c2.b)^2
@inline Base.:+(c1::Oklab{T}, c2::Oklab{T}) where T = @fastmath Oklab{T}(c1.l + c2.l, c1.a + c2.a, c1.b + c2.b)
@inline Base.:-(c1::Oklab{T}, c2::Oklab{T}) where T = @fastmath Oklab{T}(c1.l - c2.l, c1.a - c2.a, c1.b - c2.b)
@inline Base.:-(c1::Oklab{T}) where T = @fastmath Oklab{T}(-c1.l, -c1.a, -c1.b)
@inline Base.:*(x, c::Oklab{T}) where T = @fastmath Oklab{T}(x * c.l, x * c.a, x * c.b)
@inline Base.:*(c::Oklab{T}, x) where T = @fastmath Oklab{T}(x * c.l, x * c.a, x * c.b)
@inline Base.:/(c::Oklab{T}, x) where T = @fastmath Oklab{T}(c.l / x, c.a / x, c.b / x)
@inline Base.:/(x, c::Oklab{T}) where T = @fastmath Oklab{T}(x / c.l, x / c.a / x, x / c.b)

@enum DitherMethod None FloydSteinberg RandomizedRounding
@enum InitMethod TreeSplit KMeansPlusPlus

# Single Image
function quantize(img::AbstractMatrix{T};
                  ncolors::Integer = 256,
                  iterations::Integer = 2, # number of iterations in refinement of palette
                  method::DitherMethod = FloydSteinberg,
                  nthreads::Integer = 1,
                  init::InitMethod = KMeansPlusPlus) where T<:Colorant

    @assert ncolors > 0
    @assert iterations >= 0
    @assert 1 <= nthreads <= Base.Threads.nthreads()
    
    img_oklab = Oklab{Float32}.(img)
    palette = generate_palette(img_oklab, ncolors, init, nthreads)
    palette = refine_palette(img_oklab, palette, iterations, nthreads = nthreads)
    indices = dither(img_oklab, palette, method = method, nthreads = nthreads)
    return IndirectArray(indices, T.(palette))
end

# Batch API with Global Palette option
function quantize(images::AbstractVector{<:AbstractMatrix{T}};
                  ncolors::Integer = 256,
                  iterations::Integer = 2, # number of iterations in refinement of palette
                  method::DitherMethod = FloydSteinberg,
                  nthreads::Integer = 1,
                  init::InitMethod = KMeansPlusPlus,
                  global_palette::Bool = true) where T<:Colorant

    @assert ncolors > 0
    @assert iterations >= 0
    @assert 1 <= nthreads <= Base.Threads.nthreads()

    if !global_palette || length(images) == 1
        return map(images) do img
            quantize(img,
                     ncolors = ncolors,
                     iterations = iterations,
                     method = method,
                     nthreads = nthreads,
                     init = init)
        end
    end

    # 1. Aggregate a representative sample from all images
    sample_pixels = Oklab{Float32}[]
    for img in images
        inc = (length(img) + 1) ÷ length(images)
        len = length(img)
        i = 1
        while i <= len
            append!(sample_pixels, Oklab{Float32}(img[i]))
            i = i + max(1, inc + rand(-1:1))
        end
    end

    # 2. Generate ONE master palette
    master_palette = generate_palette(sample_pixels, ncolors, init = init)
    master_palette = refine_palette(sample_pixels, master_palette, iterations, nthreads = nthreads)
    master_palette = T.(master_palette)
    
    # 3. Map all images to this master palette
    return map(images) do img
        img_oklab = Oklab{Float32}.(img)
        indices = map_to_palette_oklab(img_oklab, master_palette_oklab, dither, nthreads)
        IndirectArray(indices, master_palette)
    end
end

# Internal Logic
function generate_palette(pixels, ncolors; init = KMeansPlusPlus)
    if init == TreeSplit
        return generate_palette_treesplit(pixels, ncolors)
    elseif init == KMeansPlusPlus
        return generate_palette_kmeans_pp(pixels, ncolors)
    else
        @assert init == TreeSplit || init == KMeansPlusPlus
    end
end

function generate_palette_treesplit(pixels, ncolors)
    u_pixels = unique(pixels)
    step = max(1, length(u_pixels) ÷ ncolors)
    return u_pixels[1:step:end][1:min(length(u_pixels), ncolors)]
end

function generate_palette_kmeans_pp(pixels, ncolors)
    n = length(pixels)
    k = min(ncolors, n)
    infty = typemax(eltype(valtype(pixels)))

    centroids = [pixels[rand(1:n)]]
    distances = fill(infty, n)

    @fastmath for _ in 2:k
        latest = centroids[end]

        for i in 1:n
            dist = color_dist(pixels[i], latest)
            if dist < distances[i]
                distances[i] = dist
            end
        end

        # Weighted random selection (Sequential, but O(N))
        total_w = sum(distances)
        if total_w == 0 break end
        target = rand() * total_w
        curr = 0.0
        for i in 1:n
            curr += distances[i]
            if curr >= target
                push!(centroids, pixels[i])
                break
            end
        end
    end
    return centroids
end


# For any palette color find all image pixels, which are closest to
# the palette entry and replace it with the centroid of the image
# pixel colors. Repeat this procedure _iterations_ times.
#
# In single threaded operation the function is very close to a
# function w/o paralellization. With multiple threads it performes
# almost linear with the number of threads. I there outcomment the
# single threaded version.
function refine_palette(img, palette, iterations; nthreads = 1)
    refined = copy(palette)
    infty = typemax(eltype(valtype(palette)))
    nc = length(palette)
    n = length(img)
    
    # Determine chunk size (target one chunk per available thread)
    n_tasks = nthreads
    chunk_size = max(1, n ÷ n_tasks)
    chunks = Iterators.partition(eachindex(img), chunk_size)

    @inbounds @fastmath for _ in 1:iterations
        # Define the work for a single chunk
        tasks = map(chunks) do chunk
            @spawn begin
                # Local accumulators for this specific task/chunk
                loc_L = zeros(Float64, nc)
                loc_a = zeros(Float64, nc)
                loc_b = zeros(Float64, nc)
                loc_count = zeros(Int, nc)
                
                for i in chunk
                    p = img[i]
                    best_dist = infty
                    best_index = chunk[1]
                    for j in 1:nc
                        dist = color_dist(p, refined[j])
                        if dist < best_dist
                            best_dist = dist
                            best_index = j
                        end
                    end
                    loc_L[best_index] += p.l
                    loc_a[best_index] += p.a
                    loc_b[best_index] += p.b
                    loc_count[best_index] += 1
                end
                return (loc_L, loc_a, loc_b, loc_count)
            end
        end

        # Fetch and reduce results from all tasks
        results = fetch.(tasks)
        
        # Reset master accumulators
        master_L = zeros(Float64, nc)
        master_a = zeros(Float64, nc)
        master_b = zeros(Float64, nc)
        master_count = zeros(Int, nc)

        for (L, a, b, count) in results
            master_L .+= L
            master_a .+= a
            master_b .+= b
            master_count .+= count
        end

        for i in 1:nc
            if master_count[i] > 0
                @inbounds refined[i] = Oklab{Float32}(
                    master_L[i] / master_count[i], 
                    master_a[i] / master_count[i], 
                    master_b[i] / master_count[i])
            end
        end
    end
    return refined
end



function dither(img::AbstractMatrix{T},
                palette::AbstractVector{S};
                method::DitherMethod = FloydSteinberg,
                nthreads::Int = 1) where {T <: Integer, S <: Colorant}
    if method == FloydSteinberg
        return dither_fs(img, palette)
    else
        @assert method == FloydSteinberg
    end
end        


function dither_fs(img::AbstractMatrix{T}, palette::AbstractVector{S}) where {T <: Oklab, S <: Oklab}
    # Oklab color distance squared
   
    (ny, nx) = size(img)

    idx = Matrix{UInt32}(undef, ny, nx)
    cpy = copy(img)
    infty = typemax(eltype(valtype(palette)))
    
    for x = 1:nx
        for y = 1:ny
            old_color = cpy[y, x]

            # find closest color in palette
            # the following line was slower in my benchmark
            #   (index, new_color) = argmin(idx_color -> dist(old_color, idx_color.second), pairs(palette))
            # multithreading turned out to be slower as well
            index = 1
            min_dist = infty
            for i in 2:length(palette)
                d = color_dist(old_color, palette[i])
                if d < min_dist
                    min_dist = d
                    index = i
                end
            end
                    
            idx[y, x] = index
            cpy[y, x] = palette[index]
            err = old_color - palette[index]
            if x < nx;     cpy[y    , x + 1] = cpy[y    , x + 1] + 7 / 16 * err; end
            if y < ny 
                if x > 1;  cpy[y + 1, x - 1] = cpy[y + 1, x - 1] + 3 / 16 * err; end
                cpy[y + 1, x    ] = cpy[y + 1, x    ] + 5 / 16 * err
                if x < nx; cpy[y + 1, x + 1] = cpy[y + 1, x + 1] + 1 / 16 * err; end
            end
        end
    end
    return IndirectArray(idx, palette)
end

end # module






#=
function dither_fs_parallel(img, palette; nthreads = 1)
    # Oklab color distance squared
    dist(c1::Oklab, c2::Oklab) = @fastmath (c1.l - c2.l)^2 + (c1.a - c2.a)^2 + (c1.b - c2.b)^2
   
    (ny, nx) = size(img)

    idx = Matrix{UInt32}(undef, ny, nx)
    cpy = copy(img)
    infty = typemax(eltype(valtype(palette)))

    # This version is parallelized with a @spawn macro. In my little
    # benchmarking experiments this yields a better performance with
    # multiple threads, but compared to a version without threading it
    # is still slower. At least, the reordering of the loops with the
    # wavefront counter instead of simple loops around the cartesian
    # coodinates x and y did not slow down the code.
    tasks = []
    for wfc in 1:(3nx + ny - 3) # wavefront counter
        for x = 1:nx
            y = wfc - 3x + 3
            if 1 <= y <= ny
                task = @spawn let
                    old_color = cpy[y, x]

                    # find closest color n palette
                    #   the following line was slower in my benchmark
                    #   (index, new_color) = argmin(idx_color -> dist(old_color, idx_color.second), pairs(palette))
                    index = 1
                    min_dist = infty
                    for i in 2:length(palette)
                        d = dist(old_color, palette[i])
                        if d < min_dist
                            min_dist = d
                            index = i
                        end
                    end
                    
                    idx[y, x] = index
                    cpy[y, x] = palette[index]
                    err = old_color - palette[index]
                    if x < nx;     cpy[y    , x + 1] = cpy[y    , x + 1] + 7 / 16 * err; end
                    if y < ny 
                        if x > 1;  cpy[y + 1, x - 1] = cpy[y + 1, x - 1] + 3 / 16 * err; end
                                   cpy[y + 1, x    ] = cpy[y + 1, x    ] + 5 / 16 * err
                        if x < nx; cpy[y + 1, x + 1] = cpy[y + 1, x + 1] + 1 / 16 * err; end
                    end
                end
                push!(tasks, task)
            end
            if length(tasks) > nthreads
                wait.(tasks)
                tasks = []
            end
        end
        wait.(tasks)
        tasks = []
    end
    return IndirectArray(idx, palette)
end
=#

#=
# For any palette color find all image pixels, which are closest to
# the palette entry and replace it with the centroid of the image
# pixel colors. Repeat this procedure _iterations_ times.
#
# single threaded
function refine_palette(img, palette, iterations; nthreads = 1)
    refined = copy(palette)
    infty = typemax(eltype(valtype(palette)))
    nc = length(palette)
    @inbounds for _ in 1:iterations
        # Re-map pixels to nearest palette entry and update means
        sums = fill(zero(eltype(img)), length(palette))
        counts = fill(0, length(palette))
        
        for p in img
            best_index = 1
            best_dist = infty
            for j = 1:nc
                dist = color_dist(p, refined[j])
                if dist < best_dist
                    best_index = j
                    best_dist = dist
                end
            end
            
            sums[best_index] += p
            counts[best_index] += 1
        end
        
        for i in 1:nc
            if counts[i] > 0
                refined[i] = sums[i] / counts[i]
            end
        end
    end
    return refined
end
=#



#=
# In my benchmark, the parallel version was slower than the single
# threaded.  Also only the first loop over all pixels in
# multithreaded. Therefore (presumibly), speedup of multithreading
# beyonnt two threads was diminishing or even worsening.
function generate_palette_kmeans_pp_parallel(pixels, ncolors; nthreads = 1)
    Random.seed!(1)
    n = length(pixels)
    k = min(ncolors, n)
    infty = typemax(eltype(valtype(pixels)))

    centroids = [pixels[rand(1:n)]]
    distances = fill(infty, n)

    # Manual partitioning for task-based parallelism
    chunk_size = max(1, n ÷ nthreads)
    chunks = Iterators.partition(1:n, chunk_size)

    @fastmath for _ in 2:k
        latest = centroids[end]

        # Update distances to the nearest centroid in parallel
        tasks = map(chunks) do chunk
            @spawn begin
                for i in chunk
                    # Euclidean distance squared (Fastest on 64-bit)
                    dist = color_dist(pixels[i], latest)
                    if dist < distances[i]
                        distances[i] = dist
                    end
                end
            end
        end
        fetch.(tasks)

        # Weighted random selection (Sequential, but O(N))
        total_w = sum(distances)
        if total_w == 0 break end
        target = rand() * total_w
        curr = 0.0
        for i in 1:n
            curr += distances[i]
            if curr >= target
                push!(centroids, pixels[i])
                break
            end
        end
    end
    return centroids
end
=#
