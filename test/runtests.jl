using Test

using BoundaryTest
import BoundaryTest: ω, density_estimation

dist_idx(distances, eps) = filter(i -> distances[i] < eps, eachindex(distances))
map_single(f::F, inout) where F = map!(f, inout, inout)

# uniform points on a hypersphere
point_nball(r, t = rand(); d = 2) = map_single!(p -> r * t^inv(d) * p, normalize!(randn(d)))
point_2ball(r, θ = 2π*rand(), t = rand()) = [r * sqrt(t) * cos(θ), r * sqrt(t) * sin(θ)]

point_nsphere(r; d = 2) = point_nball(r, 1; d)
point_2sphere(r, θ = 2π*rand()) = point_2ball(r, θ, 1)

# uniform points in a rectangle
point_2rect(a, b) = [a * rand() - a/2, b * rand() - b/2]

@testset "Wavey ball" begin
	R = 5
	N = 8000

	N_base = rand(N)
	points = [point_2ball(R + R/6*sin(2pi*9*i), 2pi*i) for i in N_base ]
	true_distances = [abs(R + R/6*sin(2pi*9*N_base[i]) - sqrt(p[1]^2+p[2]^2)) for (i,p) in enumerate(points)]
	ρ_true = length(points) / (ω(2)R^2)

	M = stack(points)
	d, n = size(M)

	test_1st, normals_1st, max_epsi = boundary_dists(M)
	test_2nd, normals_2nd, max_epsi = boundary_dists(M; second_order=true)

	first_order_test = abs(length(dist_idx(true_distances, max_epsi)) - length(dist_idx(test_1st, max_epsi))) / n
	secnd_order_test = abs(length(dist_idx(true_distances, max_epsi)) - length(dist_idx(test_2nd, max_epsi))) / n

	@info first_order_test
	@info secnd_order_test
	@test first_order_test < 0.05
	@test secnd_order_test < 0.05
end

@testset "Normal ball" begin
	R = 5
	N = 8000

	points = [point_2ball(R) for i in 1:N ];
	true_distances = [abs(R) - sqrt(p[1]^2+p[2]^2) for (i,p) in enumerate(points)];
	ρ_true = length(points) / (ω(2)R^2)

	M = stack(points)
	d, n = size(M)

	test_1st, normals_1st, max_epsi = boundary_dists(M)
	test_2nd, normals_2nd, max_epsi = boundary_dists(M; second_order=true)

	first_order_test = abs(length(dist_idx(true_distances, max_epsi)) - length(dist_idx(test_1st, max_epsi))) / n
	secnd_order_test = abs(length(dist_idx(true_distances, max_epsi)) - length(dist_idx(test_2nd, max_epsi))) / n

	@info first_order_test
	@info secnd_order_test
	@test first_order_test < 0.05
	@test secnd_order_test < 0.05
end

@testset "Box" begin
	R = 5
	N = 8000

	points = [point_2rect(2R, 2R) for _ in 1:N];
	true_distances = [min(R - abs(p[1]), R - abs(p[2])) for p in points];
	ρ_true = length(points) / (2R)^2

	M = stack(points)
	d, n = size(M)

	test_1st, normals_1st, max_epsi = boundary_dists(M)
	test_2nd, normals_2nd, max_epsi = boundary_dists(M; second_order=true)

	first_order_test = abs(length(dist_idx(true_distances, max_epsi)) - length(dist_idx(test_1st, max_epsi))) / n
	secnd_order_test = abs(length(dist_idx(true_distances, max_epsi)) - length(dist_idx(test_2nd, max_epsi))) / n

	@info first_order_test
	@info secnd_order_test
	@test first_order_test < 0.05
	@test secnd_order_test < 0.05
end

@testset "Regular Grid" begin
	R = 5
	grid = true
	# N = 8000

	points = vec([[a, b] for a in -R:0.2:R, b in -R:0.2:R]); #grid
	true_distances = [min(R - abs(p[1]), R - abs(p[2])) for p in points];
	ρ_true = length(points) / (2R)^2

	M = stack(points)
	d, n = size(M)

	test_1st, normals_1st, max_epsi = boundary_dists(M; grid=true)
	test_2nd, normals_2nd, max_epsi = boundary_dists(M; grid=true, second_order=true)

	first_order_test = abs(length(dist_idx(true_distances, max_epsi)) - length(dist_idx(test_1st, 5/4*max_epsi))) / n
	secnd_order_test = abs(length(dist_idx(true_distances, max_epsi)) - length(dist_idx(test_2nd, 5/4*max_epsi))) / n

	@info first_order_test
	@info secnd_order_test
	@test first_order_test < 0.05
	@test secnd_order_test < 0.05
end
