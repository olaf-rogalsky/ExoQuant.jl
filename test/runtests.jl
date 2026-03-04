using Exoquant
using Test
using Colors

@testset "Exoquant.jl Full Suite" begin
    img = rand(RGB{N0f8}, 50, 50)
    
    @testset "Initialization Methods" begin
        for init in [Exoquant.TreeSplit, Exoquant.KMeansPlusPlus]
            exq = Exoquantizer(8, init=init, iterations=0, dither=Exoquant.None)
            res = quantize(exq, img)
            @test length(res.values) <= 8
        end
    end

    @testset "Dithering Methods" begin
        for dither in [Exoquant.None, Exoquant.FloydSteinberg, Exoquant.RandomizedRounding]
            exq = Exoquantizer(4, dither=dither)
            res = quantize(exq, img)
            @test size(res) == (50, 50)
        end
    end

    @testset "Polymorphism" begin
        img_gray = rand(Gray{Float32}, 20, 20)
        res = quantize(Exoquantizer(2), img_gray)
        @test eltype(res.values) <: Gray
    end
end
