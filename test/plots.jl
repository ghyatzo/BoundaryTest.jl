using Plots
using BoundaryTest
import BoundaryTest: ω, density_estimation, grid_density_estimation

dist_idx(distances, eps) = filter(i -> distances[i] < eps, eachindex(distances))
map_single!(f::F, inout) where F = map!(f, inout, inout)

# uniform points on a hypersphere
point_nball(r, t = rand(); d = 2) = map_single!(p -> r * t^inv(d) * p, normalize!(randn(d)))
point_2ball(r, θ = 2π*rand(), t = rand()) = [r * sqrt(t) * cos(θ), r * sqrt(t) * sin(θ)]

point_nsphere(r; d = 2) = point_nball(r, 1; d)
point_2sphere(r, θ = 2π*rand()) = point_2ball(r, θ, 1)

# uniform points in a rectangle
point_2rect(a, b) = [a * rand() - a/2, b * rand() - b/2]

R = 5
N = 80000
grid = false

## -- wavey-2-ball
# N_base = rand(N);
# points = [point_2ball(R + R/6*sin(2pi*5*i), 2pi*i) for i in N_base ];
# true_distances = [abs(R + R/6*sin(2pi*5*N_base[i]) - sqrt(p[1]^2+p[2]^2)) for (i,p) in enumerate(points)];
# ρ_true = length(points) / (ω(2)R^2)

## -- 2-ball
# points = [point_2ball(R) for i in 1:N ];
# true_distances = [abs(R) - sqrt(p[1]^2+p[2]^2) for (i,p) in enumerate(points)];
# ρ_true = length(points) / (ω(2)R^2)

## -- grid
points = vec([[a, b] for a in -R:0.5:R, b in -R:0.5:R]); #grid
true_distances = [min(R - abs(p[1]), R - abs(p[2])) for p in points];
ρ_true = length(points) / (2R)^2
grid = true

## -- box
# points = [point_2rect(2R, 2R) for _ in 1:N];
# true_distances = [min(R - abs(p[1]), R - abs(p[2])) for p in points];
# ρ_true = length(points) / (2R)^d

M = stack(points);
d, n = size(M);

ρ_nn, _ = density_estimation(M)

test_1st, normals_1st, max_epsi = boundary_dists(M; grid)
test_2nd, normals_2nd, max_epsi = boundary_dists(M; grid, second_order=true)
test_epsi = grid ? 5/4 * max_epsi : max_epsi

s = Plots.scatter(Tuple.(points),
            markersize=1.5, markeralpha=0.5, markershape=:+, markerstrokewidth=0.1, color=:black, aspect_ratio=:equal, label=:none)

Plots.scatter!(s, Tuple.(points[dist_idx(true_distances, max_epsi)]),
         markersize=3.5, markeralpha=0.2, markerstrokewidth=0, color=:red, aspect_ratio=:equal, label="truth ε")

# scatter!(s, Tuple.(points[dist_idx(true_distances, 2 * 2/3 * max_epsi)]),
#          markersize=3.5, markeralpha=0.1, markerstrokewidth=0, color=:yellow, aspect_ratio=:equal, label="truth 2ε")

Plots.scatter!(s, Tuple.(points[dist_idx(test_2nd, test_epsi)]),
         markersize=1.8, markeralpha=0.4, markerstrokewidth=0.1, color=:cyan, aspect_ratio=:equal, label="2nd")

Plots.scatter!(s, Tuple.(points[dist_idx(test_1st, test_epsi)]),
         markersize=0.8, markeralpha=0.9, markerstrokewidth=0.1, color=:blue, aspect_ratio=:equal, label="1st")

outmost_normals = [eachcol(M)[i] .+ eachcol(normals_2nd)[i] for i in dist_idx(test_2nd, test_epsi)]
Plots.scatter!(s, Tuple.(outmost_normals), shape=:<, markersize=1, markerstrokewidth=0.1, aspect_ratio=:equal, color=:green, label="normals")

first_order_test = (length(dist_idx(true_distances, max_epsi)) - length(dist_idx(test_1st, test_epsi))) / length(dist_idx(true_distances, max_epsi))

secnd_order_test = (length(dist_idx(true_distances, max_epsi)) - length(dist_idx(test_2nd, test_epsi))) / length(dist_idx(true_distances, max_epsi))



### MANIFOLD TESTS
using GLMakie

## -- cut ball
points = [point_nball(3, 1, d=3) for _ in 1:20000]
# points = [point_2ball(3, t, 1) for t in 0:0.01:2π]
filter!(p -> p[2] < 0.5, points)
M = stack(points)
tree = KDTree(M)

m = BoundaryTest.manifold_dimension_estimation(M)
ρ, _ = BoundaryTest.density_estimation(M, dim=m)
r = 3/2*BoundaryTest.min_r(ρ, dim = m)
mineps = BoundaryTest.eps_from_r(r, dim = m)

kn = BoundaryTest.K(r, ρ, dim=m)
real_kn = sum(inrangecount(tree, M, r))/size(M,2)

dists, norms = BoundaryTest._second_order_manifold(M, r, m)
b_idxs = filter(i -> dists[i] < mineps, eachindex(dists))

GLMakie.scatter(M, marker=:+, color=:black, alpha=0.2)
GLMakie.scatter!(M[:, b_idxs], color=:blue)
normals = eachcol(M) .+ 0.2 .* eachcol(norms)
# GLMakie.scatter!(stack(normals)[:, b_idxs], color=:green)

P = M[:, b_idxs[4]]
Yidx = inrange(tree, P, r)
scatter!(M[:, Yidx], color=:orange)
scatter!(P..., color=:red)

## TEST FIRST ORDER
Y = M[:, Yidx] .- P
Ybar = mean(M[:, Yidx], dims=2)
E = eigvecs((M[:, Yidx].-Ybar)*(M[:, Yidx].-Ybar)')
T = E[:, end:-1:end-(m-1)]
normal = -normalize(T*T'*sum(Y, dims=2))

arrows!([Point2f(P)], [Vec2f(vec(normal))], arrowsize=(0.05,0.05,0.05), color=:green)
# arrows!(fill(P[1], 2), fill(P[2], 2), fill(P[3], 2), T[1,:], T[2,:], T[3,:], arrowsize=(0.04,0.04,0.05), color=:orange)
arrows!(Point2f.(eachcol(M[:, Yidx])), Vec2f.(eachcol(-Y)), arrowsize=(0.01,0.01), alpha=0.2)

## TEST SECOND ORDER
Y = M[:, Yidx] .- P
Ybar = mean(M[:, Yidx], dims=2)

E = eigvecs((M[:, Yidx].-Ybar)*(M[:, Yidx].-Ybar)')
T = E[:, end:-1:end-(m-1)]
# arrows!([Point2f(P)], [Vec2f(vec(T))], arrowsize=(0.04,0.04,0.05))

for j in 1:length(Yidx)
    view(Y, :, j) ./= inrangecount(tree, M[:, Yidx[j]], r/2)
end
normal = -normalize!(T*T'*sum(Y, dims=2))
arrows!([Point3f(P)], [Vec3f(vec(normal))], arrowsize=(0.05,0.05,0.05), color=:green)
arrows!([Point3f(P)], [Vec3f(norms[:, b_idxs[4]])], arrowsize=(0.05,0.05,0.05), color=:purple)
