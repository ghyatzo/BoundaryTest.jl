"""
    ω(d)

Unit volume of an hyper-sphere in d-dimensions.
i.e. ω(2) = π, ω(3) = 4π/3 ...
Volume of d-ball of radius r: ω(d)r^d
"""
ω(d) = π^(d/2)/gamma(d/2 + 1)
"""
    density_estimation(M; samples=0, k::Integer=3, knndata=nothing)

estimates the density of a point cloud by checking the radius of the ball that contains
at least `k` nearest neighbours of `samples` number of points taken at random.
If left at 0, the default number of `samples` is `N ÷ log(N)`, where `N` is the total number of points.
"""
function density_estimation(M; dim=0, samples=0, k=0, knndata=nothing)
    d, n = size(M)
    @assert d < n
    @assert n > 3^d "We need enough points to ensure that we have points in the inside."


    # if the points are meant to be on a submanifold then the volume should match
    # the manifold dimension
    d = iszero(dim) ? d : dim
    k = iszero(k) ? d : k
    tree = isnothing(knndata) ? KDTree(M) : knndata
    idxs = iszero(samples) ? rand(1:n, Int(n ÷ log(n))) : rand(1:n, samples)
    # K closest points plus the point itself.
    _, dists = knn(tree, M[:, idxs], k+1)

    # Max distance and K points
    r = sum(extrema(maximum.(dists))) / 2

    vol = ω(d)r^d

    k/vol, r
end

function manifold_dimension_estimation(M; samples=0, k=size(M,1), knndata=nothing)
    d, n = size(M)
    @assert d < n

    tree = isnothing(knndata) ? KDTree(M) : knndata
    idxs = iszero(samples) ? rand(1:n, Int(n ÷ log(n))) : rand(1:n, samples)

    Ips, _ = knn(tree, M[:, idxs], k + 1)
    res = 0
    for (neighbors, i) in zip(Ips, idxs)
        Y = M[:, neighbors] .- M[:, i]
        E = eigen(Y*Y')
        _, m = findmax([E.values[i]/E.values[i-1] for i in length(E.values):-1:2])
        res += m
    end

    round(Int, res / length(idxs))
end

function grid_density_estimation(M; dim=0, knndata=nothing)
    d, n = size(M)
    @assert d < n
    @assert n > 3^d "We need enough points to ensure that we have points in the inside."

    d = iszero(dim) ? d : dim
    tree = isnothing(knndata) ? KDTree(M) : knndata
    idxs = rand(1:n, Int(n ÷ log(n)))
    _, dists = knn(tree, M[:, idxs], d + 1)

    # one distance will always be 0, the center point itself.
    r = sum(sum.(dists)/d) / length(dists)

    # we assume that the data is regularly spaced in a hyper-cube
    internal_p = (n^inv(d)-2)^d
    vertices_p = 2^d
    face_p = n - internal_p - vertices_p

    internal_ratio = internal_p / n
    face_ratio = face_p / n
    vertices_ratio = vertices_p / n

    # the sphere centered on a point can be on an internal point, a face point or a
    # vertex point. The number of points of a regular grid within that sphere changes
    # based on the type of point
    # we don't add the point inside the sphere because it would mean counting
    # it multiple times.
    internal_k = 2^d
    face_k = 2^(d-1) + floor(2^(d-2)) # the floor is to make it work for d=1
    vertex_k = 2^(d-1)

    # we will take a weighted average based on the distribution of points in the hypercube
    k = internal_k*internal_ratio + face_k*face_ratio + vertex_k*vertices_ratio

    # we use the big box volume
    vol = (2r)^d

    k/vol, r
end

# Assuming rather uniform distribution of the point sphere
# we can take a choice of k = ω(d)rᵈρ where ω(d) is the volume
# of the unitary hypersphere in d dimensions. And ρ is the average density of the
# datapoints.
"""
    K(r, ρ; dim = 2, grid = false)

Assuming a uniform distribution `ρ` of a point cloud, we can get a rough approximation of
how many points are within a hyper-sphere of radius `r`.
The `grid` keyword can be specified to use the volume of a unit hyper-box instead.
"""
K(r, ρ; dim = 2, grid = false) = begin
    Cvol = ifelse(grid, 2^dim , ω(dim))
    Cvol * ρ * r^dim
end
"""
    r(k, ρ; dim = 2, grid = false)

Assuming uniform distribution `ρ` of a point cloud, we can get the average radius such that
the hyper-sphere of that radius will contain roughly `k` elements.
The `grid` keyword can be specified to use the volume of a unit hyper-box instead.
"""
r(k, ρ; dim = 2, grid = false) = begin
    Cvol = ifelse(grid, 2^dim , ω(dim))
    (k/(Cvol*ρ))^inv(dim)
end
"""
    mean_spread(ρ; dim = 2, grid = false)

the average distance between points in the point cloud. equivalent to `r(1, ρ; dim)`.
The `grid` keyword can be specified to use the volume of a unit hyper-box instead.
"""
mean_spread(ρ; dim = 2, grid = false) = r(1, ρ; dim, grid)


# !!!! WARNING: OPTIMAL PARAMETERS ARE VALID FOR N->inf, VERY VERY FUCKING BIG N
# !!!! IN PRACTICE N IS ALMOST NEVER REACHED FOR THESE OPTIMAL VALUES
# !!!! USE JUST ENOUGH POINTS IN THE SPHERES TO GET THE CORRECT BEHAVIOUR
# 	|Ball(x0, 3/√2 * ε)| / |Ball(x0, r)| ≤ S
#
#	|Ball(x0, 3/√2 * eps)| / |Ball(x0, r)| = (3/√2 ε/r)^d
#	(3/√2 ε/r)^d ≤ S
#	ε/r ≤ S^(1/d) √2/3
#	Given Assumption 1.1 in the paper we want
#	S^(1/d) √2/3 ≤ 1/(3√d)	=> 	S ≤ (2d)^-(d/2) or S^(1/d) ≤ (2d)^-(1/2)
#
#	this gives us
#		ε ≤ r*S^(1/d)*√2/3 ≤ r / 3√d
#		r ≥ 3 √d ε ≥ 3/√2 S^(-1/d) ε
"""
    min_r(ρ; dim = 2)

The algorithm needs enough points to be reliable, but not too many otherwise it will be biased
by the curvature of the point cloud. A good estimate is taking "the smallest radius with enough points".
This implementations considers selecting the radius that matches the assumption from the paper
of `ε/r ≤ 1/3√d` where `ε == 3mean_spread(ρ)`.

The algorithm can in theory be more precise, but it requires careful parameter tuning.
This approximations has shown to be a nice compromise between accuracy, stability, and generality.
"""
min_r(ρ; dim = 2, grid=false) = 2*3√(dim) * mean_spread(ρ; dim, grid)


"""
    eps_from_r(r; dim, S = (2dim)^(-dim/2))

Given a radius `r`, it outputs the maximum `ε` that satisfies the empirical ratio of
`|Ball(x0, 3/√2 * ε)| / |Ball(x0, r)| ≤ S` where S is taken by default such that the assumption
from the paper of `ε/r ≤ 1/3√d` is satisfied.
"""
eps_from_r(r; dim, S = (2dim)^(-dim/2)) = √2/3 * r * S^inv(dim)


"""
    r_from_eps(eps; dim, S = (2dim)^(-dim/2))

Given a border `ε`, it outputs the minimum radius `r` that satisfies the empirical ratio of
`|Ball(x0, 3/√2 * ε)| / |Ball(x0, r)| ≤ S` where S is taken by default such that the assumption
from the paper of `ε/r ≤ 1/3√d` is satisfied.
"""
r_from_eps(eps; dim, S = (2dim)^(-dim/2)) = 3/√2 * eps * (S)^-inv(dim)
