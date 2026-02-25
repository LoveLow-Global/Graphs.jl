"""
Edge-case tests for the weighted diameter algorithm introduced in PR 495.
Tests compare the optimized iFUB-based diameter against the naive
`maximum(eccentricity(g, vertices(g), distmx))` baseline.
"""

using Graphs
using Graphs.SimpleGraphs
using Test
using Random
using Logging: NullLogger, with_logger
using StableRNGs

# Reference: naive O(n * Dijkstra) diameter
function diameter_naive(g, distmx)
    return maximum(Graphs.eccentricity(g, vertices(g), distmx))
end

# Symmetric random weight matrix (positive entries)
function symmetric_weights(rng, n; lo=0.1, hi=10.0)
    W = lo .+ (hi - lo) .* rand(rng, n, n)
    return (W + W') / 2
end

@testset "Weighted diameter edge cases" begin

    # ------------------------------------------------------------------ #
    #  1. Empty graph (0 vertices)                                       #
    # ------------------------------------------------------------------ #
    @testset "Empty graph (0 vertices)" begin
        g = SimpleGraph(0)
        distmx = zeros(Float64, 0, 0)
        @test diameter(g, distmx) == 0.0

        gd = SimpleDiGraph(0)
        @test diameter(gd, distmx) == 0.0
    end

    # ------------------------------------------------------------------ #
    #  2. Single vertex                                                  #
    # ------------------------------------------------------------------ #
    @testset "Single vertex" begin
        g = SimpleGraph(1)
        distmx = fill(0.0, 1, 1)
        @test diameter(g, distmx) == 0.0

        gd = SimpleDiGraph(1)
        @test diameter(gd, distmx) == 0.0
    end

    # ------------------------------------------------------------------ #
    #  3. Two vertices, one undirected edge                              #
    # ------------------------------------------------------------------ #
    @testset "Two vertices, undirected edge" begin
        g = SimpleGraph(2)
        add_edge!(g, 1, 2)
        distmx = [0.0 3.5; 3.5 0.0]
        @test diameter(g, distmx) ≈ 3.5
        @test diameter(g, distmx) ≈ diameter_naive(g, distmx)
    end

    # ------------------------------------------------------------------ #
    #  4. Two vertices, directed cycle                                   #
    # ------------------------------------------------------------------ #
    @testset "Two vertices, directed cycle" begin
        gd = SimpleDiGraph(2)
        add_edge!(gd, 1, 2)
        add_edge!(gd, 2, 1)
        distmx = [0.0 3.0; 7.0 0.0]
        d = diameter(gd, distmx)
        @test d ≈ 7.0
        @test d ≈ diameter_naive(gd, distmx)
    end

    # ------------------------------------------------------------------ #
    #  5. Two vertices, one directed edge (not strongly connected)       #
    # ------------------------------------------------------------------ #
    @testset "Two vertices, one directed edge (disconnected)" begin
        gd = SimpleDiGraph(2)
        add_edge!(gd, 1, 2)
        distmx = [0.0 5.0; Inf 0.0]
        @test diameter(gd, distmx) == Inf
    end

    # ------------------------------------------------------------------ #
    #  6. Disconnected undirected graph                                  #
    # ------------------------------------------------------------------ #
    @testset "Disconnected undirected" begin
        g = SimpleGraph(4)
        add_edge!(g, 1, 2)
        add_edge!(g, 3, 4)
        distmx = [0.0 1.0 Inf Inf;
                   1.0 0.0 Inf Inf;
                   Inf Inf 0.0 1.0;
                   Inf Inf 1.0 0.0]
        @test diameter(g, distmx) == Inf
    end

    # ------------------------------------------------------------------ #
    #  7. Isolated vertices (no edges at all)                            #
    # ------------------------------------------------------------------ #
    @testset "No edges, multiple vertices" begin
        g = SimpleGraph(5)
        distmx = fill(Inf, 5, 5)
        for i in 1:5; distmx[i, i] = 0.0; end
        @test diameter(g, distmx) == Inf
    end

    # ------------------------------------------------------------------ #
    #  8. Path graph – undirected weighted                               #
    # ------------------------------------------------------------------ #
    @testset "Path graph undirected weighted" begin
        # 1 -- 2 -- 3 -- 4 -- 5  with known weights
        g = path_graph(5)
        # weights on edges: 1-2: 2, 2-3: 3, 3-4: 5, 4-5: 1
        distmx = fill(Inf, 5, 5)
        for i in 1:5; distmx[i, i] = 0.0; end
        distmx[1,2] = distmx[2,1] = 2.0
        distmx[2,3] = distmx[3,2] = 3.0
        distmx[3,4] = distmx[4,3] = 5.0
        distmx[4,5] = distmx[5,4] = 1.0
        # Diameter should be d(1,5) = 2+3+5+1 = 11
        @test diameter(g, distmx) ≈ 11.0
        @test diameter(g, distmx) ≈ diameter_naive(g, distmx)
    end

    # ------------------------------------------------------------------ #
    #  9. Path digraph – directed weighted                               #
    # ------------------------------------------------------------------ #
    @testset "Path digraph weighted (not strongly connected)" begin
        gd = path_digraph(4)
        distmx = fill(Inf, 4, 4)
        for i in 1:4; distmx[i, i] = 0.0; end
        distmx[1,2] = 1.0; distmx[2,3] = 2.0; distmx[3,4] = 3.0
        # Not strongly connected → Inf
        @test diameter(gd, distmx) == Inf
    end

    # ------------------------------------------------------------------ #
    #  10. Cycle graph – undirected weighted                             #
    # ------------------------------------------------------------------ #
    @testset "Cycle graph undirected weighted" begin
        g = cycle_graph(4)  # 1-2-3-4-1
        distmx = fill(Inf, 4, 4)
        for i in 1:4; distmx[i, i] = 0.0; end
        distmx[1,2] = distmx[2,1] = 1.0
        distmx[2,3] = distmx[3,2] = 1.0
        distmx[3,4] = distmx[4,3] = 1.0
        distmx[4,1] = distmx[1,4] = 1.0
        # All unit weights, so diameter = 2 (opposite corners)
        @test diameter(g, distmx) ≈ 2.0
        @test diameter(g, distmx) ≈ diameter_naive(g, distmx)
    end

    # ------------------------------------------------------------------ #
    #  11. Cycle digraph – directed weighted                             #
    # ------------------------------------------------------------------ #
    @testset "Cycle digraph weighted" begin
        gd = cycle_digraph(4)  # 1→2→3→4→1
        distmx = fill(Inf, 4, 4)
        for i in 1:4; distmx[i, i] = 0.0; end
        distmx[1,2] = 1.0; distmx[2,3] = 2.0
        distmx[3,4] = 3.0; distmx[4,1] = 4.0
        # d(1,2)=1, d(1,3)=3, d(1,4)=6
        # d(2,3)=2, d(2,4)=5, d(2,1)=9
        # d(3,4)=3, d(3,1)=7, d(3,2)=8
        # d(4,1)=4, d(4,2)=5, d(4,3)=7
        # max = d(2,1) = 9
        @test diameter(gd, distmx) ≈ 9.0
        @test diameter(gd, distmx) ≈ diameter_naive(gd, distmx)
    end

    # ------------------------------------------------------------------ #
    #  12. Star graph – undirected weighted                              #
    # ------------------------------------------------------------------ #
    @testset "Star graph undirected weighted" begin
        g = star_graph(6)  # center=1, leaves=2..6
        rng = StableRNG(42)
        distmx = fill(Inf, 6, 6)
        for i in 1:6; distmx[i, i] = 0.0; end
        spoke_weights = [2.0, 5.0, 1.0, 3.0, 4.0]
        for (j, w) in enumerate(spoke_weights)
            distmx[1, j+1] = distmx[j+1, 1] = w
        end
        # Diameter is between two farthest leaves via center:
        # max_{i,j} (w_i + w_j) = 5+4 = 9 (leaves 3 and 6)
        @test diameter(g, distmx) ≈ 9.0
        @test diameter(g, distmx) ≈ diameter_naive(g, distmx)
    end

    # ------------------------------------------------------------------ #
    #  13. Complete graph – undirected weighted                          #
    # ------------------------------------------------------------------ #
    @testset "Complete graph undirected weighted" begin
        n = 8
        g = complete_graph(n)
        rng = StableRNG(123)
        distmx = symmetric_weights(rng, n)
        d = diameter(g, distmx)
        d_ref = diameter_naive(g, distmx)
        @test d ≈ d_ref
    end

    # ------------------------------------------------------------------ #
    #  14. Complete digraph – directed weighted                          #
    # ------------------------------------------------------------------ #
    @testset "Complete digraph directed weighted" begin
        n = 8
        gd = complete_digraph(n)
        rng = StableRNG(456)
        distmx = 0.1 .+ 9.9 .* rand(rng, n, n)
        d = diameter(gd, distmx)
        d_ref = diameter_naive(gd, distmx)
        @test d ≈ d_ref
    end

    # ------------------------------------------------------------------ #
    #  15. Unit weights on weighted path should match unweighted         #
    # ------------------------------------------------------------------ #
    @testset "Unit weight matrix matches unweighted diameter" begin
        for n in [3, 10, 50]
            g = path_graph(n)
            distmx = zeros(Int, n, n)
            for e in edges(g)
                distmx[src(e), dst(e)] = 1
                distmx[dst(e), src(e)] = 1
            end
            @test diameter(g, distmx) == n - 1
            @test diameter(g, distmx) == diameter(g)
        end
    end

    # ------------------------------------------------------------------ #
    #  16. Diameter NOT on hub vertex's shortest path tree               #
    # ------------------------------------------------------------------ #
    @testset "Diameter between low-degree peripheral vertices" begin
        # Create a barbell-like graph: two cliques connected by a long path
        # The hub has high degree, but the diameter is between the far ends
        n_clique = 5
        g = complete_graph(n_clique)  # vertices 1..5
        # Add a path 5-6-7-8-9-10
        for v in 6:10
            add_vertex!(g)
        end
        add_edge!(g, 5, 6)
        for v in 6:9
            add_edge!(g, v, v+1)
        end

        n = nv(g)
        distmx = fill(Inf, n, n)
        for i in 1:n; distmx[i,i] = 0.0; end
        for e in edges(g)
            distmx[src(e), dst(e)] = 1.0
            distmx[dst(e), src(e)] = 1.0
        end
        # Make the path edges have large weight
        distmx[5,6] = distmx[6,5] = 10.0
        for v in 6:9
            distmx[v, v+1] = distmx[v+1, v] = 10.0
        end
        # Vertex 1 has highest degree (4 in the clique)
        # but diameter is d(1, 10) which goes through the long weighted path
        d = diameter(g, distmx)
        d_ref = diameter_naive(g, distmx)
        @test d ≈ d_ref
    end

    # ------------------------------------------------------------------ #
    #  17. Integer weight matrix                                         #
    # ------------------------------------------------------------------ #
    @testset "Integer weight matrix" begin
        g = path_graph(5)
        distmx = zeros(Int, 5, 5)
        distmx[1,2] = distmx[2,1] = 3
        distmx[2,3] = distmx[3,2] = 7
        distmx[3,4] = distmx[4,3] = 2
        distmx[4,5] = distmx[5,4] = 5
        @test diameter(g, distmx) == 17
        @test diameter(g, distmx) == diameter_naive(g, distmx)
    end

    # ------------------------------------------------------------------ #
    #  18. Very large weights                                            #
    # ------------------------------------------------------------------ #
    @testset "Large weights" begin
        g = path_graph(3)
        distmx = [0.0 1e15 Inf; 1e15 0.0 1e15; Inf 1e15 0.0]
        @test diameter(g, distmx) ≈ 2e15
        @test diameter(g, distmx) ≈ diameter_naive(g, distmx)
    end

    # ------------------------------------------------------------------ #
    #  19. Very small weights (near zero but positive)                   #
    # ------------------------------------------------------------------ #
    @testset "Very small weights" begin
        g = path_graph(4)
        distmx = fill(Inf, 4, 4)
        for i in 1:4; distmx[i,i] = 0.0; end
        distmx[1,2] = distmx[2,1] = 1e-15
        distmx[2,3] = distmx[3,2] = 1e-15
        distmx[3,4] = distmx[4,3] = 1e-15
        d = diameter(g, distmx)
        d_ref = diameter_naive(g, distmx)
        @test d ≈ d_ref
    end

    # ------------------------------------------------------------------ #
    #  20. All edges same weight (should be proportional to unweighted)  #
    # ------------------------------------------------------------------ #
    @testset "Uniform edge weights" begin
        g = cycle_graph(10)
        w = 3.7
        distmx = fill(Inf, 10, 10)
        for i in 1:10; distmx[i,i] = 0.0; end
        for e in edges(g)
            distmx[src(e), dst(e)] = w
            distmx[dst(e), src(e)] = w
        end
        @test diameter(g, distmx) ≈ w * diameter(g)
        @test diameter(g, distmx) ≈ diameter_naive(g, distmx)
    end

    # ------------------------------------------------------------------ #
    #  21. Directed cycle with highly asymmetric weights                 #
    # ------------------------------------------------------------------ #
    @testset "Directed cycle with asymmetric weights" begin
        # 1→2→3→1 with very different weights
        gd = SimpleDiGraph(3)
        add_edge!(gd, 1, 2); add_edge!(gd, 2, 3); add_edge!(gd, 3, 1)
        distmx = [Inf 1.0 Inf; Inf Inf 1.0; 100.0 Inf Inf]
        # d(1,2)=1, d(1,3)=2
        # d(2,3)=1, d(2,1)=101
        # d(3,1)=100, d(3,2)=101
        # diameter = 101
        d = diameter(gd, distmx)
        @test d ≈ 101.0
        @test d ≈ diameter_naive(gd, distmx)
    end

    # ------------------------------------------------------------------ #
    #  22. Strongly connected directed graph – complete tournament       #
    # ------------------------------------------------------------------ #
    @testset "Complete tournament (directed)" begin
        # All n*(n-1) directed edges present
        n = 6
        gd = complete_digraph(n)
        rng = StableRNG(789)
        distmx = 0.1 .+ rand(rng, n, n) .* 9.9
        d = diameter(gd, distmx)
        d_ref = diameter_naive(gd, distmx)
        @test d ≈ d_ref
    end

    # ------------------------------------------------------------------ #
    #  23. Grid / Lattice graph                                          #
    # ------------------------------------------------------------------ #
    @testset "Grid graph weighted" begin
        g = Graphs.grid([5, 5])  # 5x5 grid
        n = nv(g)
        rng = StableRNG(999)
        distmx = symmetric_weights(rng, n; lo=1.0, hi=5.0)
        d = diameter(g, distmx)
        d_ref = diameter_naive(g, distmx)
        @test d ≈ d_ref
    end

    # ------------------------------------------------------------------ #
    #  24. Tree graph (no cycles)                                        #
    # ------------------------------------------------------------------ #
    @testset "Binary tree weighted" begin
        # Build a binary tree of depth 4
        g = SimpleGraph(15)
        for i in 1:7
            add_edge!(g, i, 2i)
            add_edge!(g, i, 2i+1)
        end
        n = nv(g)
        rng = StableRNG(2024)
        distmx = symmetric_weights(rng, n)
        d = diameter(g, distmx)
        d_ref = diameter_naive(g, distmx)
        @test d ≈ d_ref
    end

    # ------------------------------------------------------------------ #
    #  25. Barbell graph (two cliques connected by one edge)             #
    # ------------------------------------------------------------------ #
    @testset "Barbell graph weighted" begin
        g1 = complete_graph(5)
        g2 = complete_graph(5)
        g = blockdiag(g1, g2)
        add_edge!(g, 5, 6)  # bridge
        n = nv(g)
        rng = StableRNG(314)
        distmx = symmetric_weights(rng, n)
        d = diameter(g, distmx)
        d_ref = diameter_naive(g, distmx)
        @test d ≈ d_ref
    end

    # ------------------------------------------------------------------ #
    #  26. Directed graph with bridge that creates long detour           #
    # ------------------------------------------------------------------ #
    @testset "Directed graph with detour" begin
        # 1⇄2⇄3, plus shortcut 1→3 with weight 0.1
        # and reverse 3→1 with weight 100
        gd = SimpleDiGraph(3)
        add_edge!(gd, 1, 2); add_edge!(gd, 2, 1)
        add_edge!(gd, 2, 3); add_edge!(gd, 3, 2)
        add_edge!(gd, 1, 3); add_edge!(gd, 3, 1)
        distmx = [Inf 1.0 0.1;
                   1.0 Inf 1.0;
                   100.0 1.0 Inf]
        d = diameter(gd, distmx)
        d_ref = diameter_naive(gd, distmx)
        @test d ≈ d_ref
    end

    # ------------------------------------------------------------------ #
    #  27. Wheel graph (hub + cycle of spokes)                           #
    # ------------------------------------------------------------------ #
    @testset "Wheel graph weighted" begin
        g = wheel_graph(10)  # vertex 1 is center
        n = nv(g)
        rng = StableRNG(555)
        distmx = symmetric_weights(rng, n)
        d = diameter(g, distmx)
        d_ref = diameter_naive(g, distmx)
        @test d ≈ d_ref
    end

    # ------------------------------------------------------------------ #
    #  28. Caterpillar graph (long spine with degree-1 leaves)           #
    # ------------------------------------------------------------------ #
    @testset "Caterpillar graph weighted" begin
        # Spine: 1-2-3-4-5, each spine vertex gets a leaf
        g = path_graph(5)
        for v in 1:5
            add_vertex!(g)
            add_edge!(g, v, v + 5)
        end
        n = nv(g)
        rng = StableRNG(777)
        distmx = symmetric_weights(rng, n)
        d = diameter(g, distmx)
        d_ref = diameter_naive(g, distmx)
        @test d ≈ d_ref
    end

    # ------------------------------------------------------------------ #
    #  29. All vertices same degree (regular graph)                      #
    # ------------------------------------------------------------------ #
    @testset "Regular graph (cycle) weighted" begin
        for n in [5, 10, 20]
            g = cycle_graph(n)
            rng = StableRNG(n)
            distmx = symmetric_weights(rng, n)
            d = diameter(g, distmx)
            d_ref = diameter_naive(g, distmx)
            @test d ≈ d_ref
        end
    end

    # ------------------------------------------------------------------ #
    #  30. Petersen graph – 3-regular, diameter 2 unweighted             #
    # ------------------------------------------------------------------ #
    @testset "Petersen graph weighted" begin
        g = smallgraph(:petersen)
        n = nv(g)
        rng = StableRNG(1001)
        distmx = symmetric_weights(rng, n)
        d = diameter(g, distmx)
        d_ref = diameter_naive(g, distmx)
        @test d ≈ d_ref
    end

    # ------------------------------------------------------------------ #
    #  31. Barabasi-Albert model (power-law degree)                      #
    # ------------------------------------------------------------------ #
    @testset "Barabasi-Albert weighted" begin
        rng = StableRNG(2025)
        for _ in 1:5
            n = rand(rng, 20:60)
            g = barabasi_albert(n, 3; rng=rng)
            while !is_connected(g)
                g = barabasi_albert(n, 3; rng=rng)
            end
            nv_g = nv(g)
            distmx = symmetric_weights(rng, nv_g)
            d = diameter(g, distmx)
            d_ref = diameter_naive(g, distmx)
            @test d ≈ d_ref
        end
    end

    # ------------------------------------------------------------------ #
    #  32. Random Erdos-Renyi undirected – many samples                  #
    # ------------------------------------------------------------------ #
    @testset "Random ER undirected weighted (many samples)" begin
        rng = StableRNG(3000)
        with_logger(NullLogger()) do
            for _ in 1:30
                n = rand(rng, 5:40)
                p = 0.1 + 0.3 * rand(rng)
                g = erdos_renyi(n, p; rng=rng)
                # Use largest connected component
                ccs = connected_components(g)
                largest = ccs[argmax(length.(ccs))]
                gsub, _ = induced_subgraph(g, largest)
                nv(gsub) <= 1 && continue
                nv_s = nv(gsub)
                distmx = symmetric_weights(rng, nv_s)
                d = diameter(gsub, distmx)
                d_ref = diameter_naive(gsub, distmx)
                @test d ≈ d_ref
            end
        end
    end

    # ------------------------------------------------------------------ #
    #  33. Random ER directed weighted – many samples                    #
    # ------------------------------------------------------------------ #
    @testset "Random ER directed weighted (many samples)" begin
        rng = StableRNG(4000)
        with_logger(NullLogger()) do
            for _ in 1:30
                n = rand(rng, 5:40)
                p = 0.15 + 0.3 * rand(rng)
                gd = erdos_renyi(n, p; is_directed=true, rng=rng)
                sccs = strongly_connected_components(gd)
                largest = sccs[argmax(length.(sccs))]
                gsub, _ = induced_subgraph(gd, largest)
                nv(gsub) <= 1 && continue
                nv_s = nv(gsub)
                distmx = 0.1 .+ 9.9 .* rand(rng, nv_s, nv_s)
                d = diameter(gsub, distmx)
                d_ref = diameter_naive(gsub, distmx)
                @test d ≈ d_ref
            end
        end
    end

    # ------------------------------------------------------------------ #
    #  34. Random disconnected undirected (should return Inf)            #
    # ------------------------------------------------------------------ #
    @testset "Disconnected random undirected weighted" begin
        rng = StableRNG(5000)
        for _ in 1:10
            n = rand(rng, 10:30)
            g = erdos_renyi(n, 0.02; rng=rng)
            if !is_connected(g)
                nv_g = nv(g)
                distmx = symmetric_weights(rng, nv_g)
                @test diameter(g, distmx) == Inf
            end
        end
    end

    # ------------------------------------------------------------------ #
    #  35. Random disconnected directed (should return typemax)          #
    # ------------------------------------------------------------------ #
    @testset "Disconnected random directed weighted" begin
        rng = StableRNG(6000)
        for _ in 1:10
            n = rand(rng, 10:30)
            gd = erdos_renyi(n, 0.02; is_directed=true, rng=rng)
            if !is_strongly_connected(gd)
                nv_g = nv(gd)
                distmx = 0.1 .+ 9.9 .* rand(rng, nv_g, nv_g)
                @test diameter(gd, distmx) == Inf
            end
        end
    end

    # ------------------------------------------------------------------ #
    #  36. Directed graph where reverse gives the diameter               #
    # ------------------------------------------------------------------ #
    @testset "Directed: diameter found in reverse direction" begin
        # 1→2 (w=1), 2→3 (w=1), 3→1 (w=1), plus 1→3 (w=0.5)
        # Forward from 1: d(1,2)=1, d(1,3)=0.5
        # From 2: d(2,3)=1, d(2,1)=2
        # From 3: d(3,1)=1, d(3,2)=2
        # Diameter = 2
        gd = SimpleDiGraph(3)
        add_edge!(gd, 1, 2); add_edge!(gd, 2, 3)
        add_edge!(gd, 3, 1); add_edge!(gd, 1, 3)
        distmx = [Inf 1.0 0.5;
                   Inf Inf 1.0;
                   1.0 Inf Inf]
        d = diameter(gd, distmx)
        d_ref = diameter_naive(gd, distmx)
        @test d ≈ d_ref
    end

    # ------------------------------------------------------------------ #
    #  37. Graph where highest-degree vertex is far from diameter path   #
    # ------------------------------------------------------------------ #
    @testset "Hub far from diameter endpoints" begin
        # Hub vertex 1 connected to 2,3,4,5 (degree 4)
        # Long chain: 6-7-8-9-10 (degree 1 or 2)
        # Connect hub to chain via vertex 5-6
        g = SimpleGraph(10)
        # Hub edges
        for v in 2:5
            add_edge!(g, 1, v)
        end
        # Chain
        add_edge!(g, 5, 6)
        for v in 6:9
            add_edge!(g, v, v+1)
        end
        n = nv(g)
        distmx = fill(Inf, n, n)
        for i in 1:n; distmx[i,i] = 0.0; end
        for e in edges(g)
            distmx[src(e), dst(e)] = 100.0
            distmx[dst(e), src(e)] = 100.0
        end
        # Make hub edges very short
        for v in 2:5
            distmx[1, v] = distmx[v, 1] = 0.1
        end
        # Chain edges are long (100 each)
        # d(2, 10) = 0.1 + 0.1 + 100*5 = 500.2
        # Diameter is between peripheral vertices
        d = diameter(g, distmx)
        d_ref = diameter_naive(g, distmx)
        @test d ≈ d_ref
    end

    # ------------------------------------------------------------------ #
    #  38. Float32 weight matrix                                         #
    # ------------------------------------------------------------------ #
    @testset "Float32 weight matrix" begin
        g = cycle_graph(5)
        distmx = Float32[0 1 Inf Inf 1;
                         1 0 2 Inf Inf;
                         Inf 2 0 3 Inf;
                         Inf Inf 3 0 4;
                         1 Inf Inf 4 0]
        d = diameter(g, distmx)
        d_ref = diameter_naive(g, distmx)
        @test d ≈ d_ref
    end

    # ------------------------------------------------------------------ #
    #  39. Weight matrix with some zero-weight edges                     #
    # ------------------------------------------------------------------ #
    @testset "Zero-weight edges" begin
        g = SimpleGraph(4)
        add_edge!(g, 1, 2)
        add_edge!(g, 2, 3)
        add_edge!(g, 3, 4)
        add_edge!(g, 1, 4)
        distmx = [0.0  0.0  Inf  5.0;
                   0.0  0.0  0.0  Inf;
                   Inf  0.0  0.0  0.0;
                   5.0  Inf  0.0  0.0]
        # All paths through 1-2-3-4 have zero weight except going 1→4 direct (5.0)
        # d(1,4) = min(5.0, 0+0+0) = 0.0
        # Diameter = max over all pairs = 5.0? No...
        # Actually with zero-weight edges: d(1,2)=0, d(1,3)=0, d(1,4)=0, d(2,3)=0, d(2,4)=0, d(3,4)=0
        # All distances are 0. Diameter = 0.
        # But d(4,1) going 4→1 is 5.0 or 4→3→2→1 = 0+0+0 = 0. So d(4,1)=0.
        # OK for undirected: d(1,4) = min(5.0, 0+0+0) = 0.
        d = diameter(g, distmx)
        d_ref = diameter_naive(g, distmx)
        @test d ≈ d_ref
    end

    # ------------------------------------------------------------------ #
    #  40. Graph with bridge edge (biconnected component test)           #
    # ------------------------------------------------------------------ #
    @testset "Bridge edge with high weight" begin
        # Two triangles connected by a bridge
        g = SimpleGraph(6)
        add_edge!(g, 1, 2); add_edge!(g, 2, 3); add_edge!(g, 1, 3)
        add_edge!(g, 4, 5); add_edge!(g, 5, 6); add_edge!(g, 4, 6)
        add_edge!(g, 3, 4)  # bridge
        n = nv(g)
        distmx = fill(Inf, n, n)
        for i in 1:n; distmx[i,i] = 0.0; end
        for e in edges(g)
            distmx[src(e), dst(e)] = 1.0
            distmx[dst(e), src(e)] = 1.0
        end
        distmx[3,4] = distmx[4,3] = 1000.0  # expensive bridge
        d = diameter(g, distmx)
        d_ref = diameter_naive(g, distmx)
        @test d ≈ d_ref
        # Diameter should be ≈ 1002 (e.g., d(1,6) = 1 + 1000 + 1 = 1002)
        @test d ≈ 1002.0
    end

    # ------------------------------------------------------------------ #
    #  41. Directed graph where all vertices have equal total degree     #
    # ------------------------------------------------------------------ #
    @testset "Directed regular graph weighted" begin
        # Directed cycle: all vertices have in-degree 1, out-degree 1
        n = 7
        gd = cycle_digraph(n)
        rng = StableRNG(8888)
        distmx = zeros(Float64, n, n)
        for e in edges(gd)
            distmx[src(e), dst(e)] = 0.1 + 9.9 * rand(rng)
        end
        d = diameter(gd, distmx)
        d_ref = diameter_naive(gd, distmx)
        @test d ≈ d_ref
    end

    # ------------------------------------------------------------------ #
    #  42. Very dense random undirected graph                            #
    # ------------------------------------------------------------------ #
    @testset "Dense random undirected" begin
        rng = StableRNG(9999)
        n = 30
        g = erdos_renyi(n, 0.8; rng=rng)
        while !is_connected(g)
            g = erdos_renyi(n, 0.8; rng=rng)
        end
        distmx = symmetric_weights(rng, n)
        d = diameter(g, distmx)
        d_ref = diameter_naive(g, distmx)
        @test d ≈ d_ref
    end

    # ------------------------------------------------------------------ #
    #  43. Very sparse random undirected graph (tree-like)               #
    # ------------------------------------------------------------------ #
    @testset "Sparse random undirected (tree-like)" begin
        rng = StableRNG(7777)
        n = 40
        # Build a random tree by connecting each new vertex to a random existing one
        g = SimpleGraph(n)
        for v in 2:n
            u = rand(rng, 1:v-1)
            add_edge!(g, u, v)
        end
        distmx = symmetric_weights(rng, n)
        d = diameter(g, distmx)
        d_ref = diameter_naive(g, distmx)
        @test d ≈ d_ref
    end

    # ------------------------------------------------------------------ #
    #  44. Graph where the naive lb = ecc(u) already equals diameter     #
    # ------------------------------------------------------------------ #
    @testset "Initial lower bound equals diameter" begin
        # Star graph: hub has eccentricity = max spoke weight
        # Diameter = sum of two largest spoke weights (through hub)
        # So lb = ecc(hub) < diameter ... unless graph is K2
        g = SimpleGraph(2)
        add_edge!(g, 1, 2)
        distmx = [0.0 42.0; 42.0 0.0]
        # Here ecc(1) = ecc(2) = 42, diameter = 42. lb starts at 42, correct immediately
        @test diameter(g, distmx) ≈ 42.0
    end

    # ------------------------------------------------------------------ #
    #  45. Type stability / inference tests                              #
    # ------------------------------------------------------------------ #
    @testset "Type inference" begin
        g = path_graph(3)
        distmx_f64 = [0.0 1.0 Inf; 1.0 0.0 1.0; Inf 1.0 0.0]
        @test @inferred(diameter(g, distmx_f64)) isa Float64

        distmx_int = [0 1 100; 1 0 1; 100 1 0]
        @test @inferred(diameter(g, distmx_int)) isa Int

        gd = SimpleDiGraph(2)
        add_edge!(gd, 1, 2); add_edge!(gd, 2, 1)
        distmx_dir = [0.0 1.0; 2.0 0.0]
        @test @inferred(diameter(gd, distmx_dir)) isa Float64
    end

    # ------------------------------------------------------------------ #
    #  46. Asymmetric weight matrix on undirected graph                  #
    # ------------------------------------------------------------------ #
    @testset "Asymmetric weights on undirected graph" begin
        # Dijkstra on undirected graph uses distmx[u,v] for edge (u→v)
        # and distmx[v,u] for edge (v→u), creating asymmetric shortest paths.
        # The naive approach computes this correctly; check optimized does too.
        g = path_graph(4)
        distmx = [0.0 1.0 Inf Inf;
                   5.0 0.0 1.0 Inf;
                   Inf 5.0 0.0 1.0;
                   Inf Inf 5.0 0.0]
        # Undirected graph, but asymmetric weights:
        # d(1→2) = 1, d(2→1) = 5
        # d(1→3) = 1+1=2, d(3→1) = 5+5=10
        # d(1→4) = 1+1+1=3, d(4→1) = 5+5+5=15
        # Diameter (max over all directed pairs) = 15
        d = diameter(g, distmx)
        d_ref = diameter_naive(g, distmx)
        @test d ≈ d_ref
    end

    # ------------------------------------------------------------------ #
    #  47. Larger directed graph with mixed structure                    #
    # ------------------------------------------------------------------ #
    @testset "Mixed directed graph" begin
        # Create a directed graph that is strongly connected
        # but has both short and long paths between vertices
        n = 15
        gd = SimpleDiGraph(n)
        # Hamiltonian cycle to ensure strong connectivity
        for v in 1:n-1
            add_edge!(gd, v, v+1)
        end
        add_edge!(gd, n, 1)
        # Add some shortcuts
        rng = StableRNG(1111)
        for _ in 1:n
            u, v = rand(rng, 1:n, 2)
            u != v && add_edge!(gd, u, v)
        end
        distmx = 0.1 .+ 9.9 .* rand(rng, n, n)
        d = diameter(gd, distmx)
        d_ref = diameter_naive(gd, distmx)
        @test d ≈ d_ref
    end

    # ------------------------------------------------------------------ #
    #  48. Graph where diameter pair is at distance 1 from hub (tricky)  #
    # ------------------------------------------------------------------ #
    @testset "Diameter pair both adjacent to hub" begin
        # Star: hub=1, spokes to 2,3,4,5
        # Edge 2-3 also exists (triangle 1-2-3)
        # Weights: spoke(1→2)=10, spoke(1→3)=10, edge(2→3)=100
        g = SimpleGraph(5)
        for v in 2:5; add_edge!(g, 1, v); end
        add_edge!(g, 2, 3)
        n = 5
        distmx = fill(Inf, n, n)
        for i in 1:n; distmx[i,i] = 0.0; end
        for v in 2:5
            distmx[1,v] = distmx[v,1] = 10.0
        end
        distmx[2,3] = distmx[3,2] = 100.0
        # d(2,3) = min(100, 10+10) = 20 (via hub)
        # d(2,4) = 10+10 = 20, d(2,5) = 20, d(3,4) = 20, d(3,5) = 20, d(4,5) = 20
        # diameter = 20
        d = diameter(g, distmx)
        d_ref = diameter_naive(g, distmx)
        @test d ≈ d_ref
        @test d ≈ 20.0
    end

    # ------------------------------------------------------------------ #
    #  49. Directed graph with multiple SCCs (not strongly connected)    #
    # ------------------------------------------------------------------ #
    @testset "Directed graph with multiple SCCs" begin
        gd = SimpleDiGraph(4)
        add_edge!(gd, 1, 2)
        add_edge!(gd, 2, 1)
        add_edge!(gd, 3, 4)
        add_edge!(gd, 4, 3)
        # Two SCCs: {1,2} and {3,4}, no edges between them
        distmx = [Inf 1.0 Inf Inf;
                   2.0 Inf Inf Inf;
                   Inf Inf Inf 3.0;
                   Inf Inf 4.0 Inf]
        @test diameter(gd, distmx) == Inf
    end

    # ------------------------------------------------------------------ #
    #  50. Stress test: Random ER with various seeds and sizes           #
    # ------------------------------------------------------------------ #
    @testset "Stress test random graphs" begin
        with_logger(NullLogger()) do
            for seed in 10001:10020
                rng = StableRNG(seed)
                n = rand(rng, 3:25)
                p = 0.2 + 0.4 * rand(rng)

                # Undirected
                g = erdos_renyi(n, p; rng=rng)
                if is_connected(g) && nv(g) > 1
                    nv_g = nv(g)
                    distmx = symmetric_weights(rng, nv_g)
                    d = diameter(g, distmx)
                    d_ref = diameter_naive(g, distmx)
                    @test d ≈ d_ref
                end

                # Directed
                gd = erdos_renyi(n, p; is_directed=true, rng=rng)
                if is_strongly_connected(gd) && nv(gd) > 1
                    nv_gd = nv(gd)
                    distmx_d = 0.1 .+ 9.9 .* rand(rng, nv_gd, nv_gd)
                    d = diameter(gd, distmx_d)
                    d_ref = diameter_naive(gd, distmx_d)
                    @test d ≈ d_ref
                end
            end
        end
    end
end
