module BoundaryTest


using NearestNeighbors
using LinearAlgebra
using SpecialFunctions: gamma
using ProgressMeter

export boundary_dists

include("utils.jl")

include("first_order.jl")

include("second_order.jl")

"""
    boundary_dists(X::AbstractMatrix [, scale=1; grid=false, second_order=false, knndata=nothing])
    -> distances::Vector, normals::Matrix, epsilon::Float

The algorithm is from the paper: [Boundary Estimation from Points Clouds: Algorithms, Guarantees and Applications](
https://doi.org/10.48550/arXiv.2111.03217
)
Computes the approximate distances to the border of the point cloud.
Also returns approximated normals of all points.

The point data matrix is assumed to be in column major order, that is, **all points are along the columns**


It attempts to automatically select a suitable computing radius by approximating the point cloud
density and chosing `r` such that is proportional to the mean distance between points.
It can be tweaked by specifying the `scale` parameter in case the resulting radius is not good enough.

The distance and normal approximations are roughly valid for the points within the `2*epsilon`-border.

There is a delicate interplay between radius choice and border width, this function also attempts to return
	a sensible default border width `epsilon` that matches the radius choice.

The actual border points will be those that have distance less than epsilon
#Testing distances


```julia-repl
julia> X = rand(2, 2000); #data

julia> dists, normals, epsilon = boundary_dists(X);

julia> # There is the assumption that the data will **NOT** be reordered;

julia> border_idxs = filter(i -> dists[i] < epsilon, eachindex(dists))

julia> border_points = X[:, border_idxs]
```
The actual border width is not fixed in stone. For better border matching, it might be needed to consider
	slightly bigger border widths by testing against `3/2 epsilon` or `5/4 epsilon` for example.

The function can be specialised for gridded data through the `grid` keyword, which will use a more precise algorithm
	to approximate the density as well as using smaller radius, since we can leverage the regularity of the grid
	to require less points for a good enough estimation.

If already available a nearest neighbour tree data it can be passed to the function through the
	`knndata` keyword. Internally the function uses the `NearestNeighbors` package.

Lastly, by default the function will use a first order approximation due to it being good enough
on simple shapes and much faster than the second order. But there are situations where the first order is not
enough, especially when dealing with thicker borders, negative curvature in the data or
non uniform distribution of the points. In such cases it might be beneficial to switch to the second order (still decently fast don't worry).

In case more precision in the choice of the radius is desired, the lower level methods `_first_order` and
`_second_order` can be used directly, but must be imported explicitly.
"""
function boundary_dists(X::AbstractMatrix, scale=1; grid=false, second_order=false, knndata=nothing)
	d, n = size(X)

	tree = isnothing(knndata) ? KDTree(X) : knndata
	ρ , _ = grid ? grid_density_estimation(X; knndata=tree) : density_estimation(X; knndata=tree)

	# obtain the minimum radius such that we have border width that is twice the mean
	# distance between points in the dataset.
	# in case of a regular grid we can bring it down to the single mean distance between points
	# for more accuracy.
	r = scale*min_r(ρ; dim=d, grid)

	# then we can obtain the maximum epsilon
	ε = eps_from_r(r, dim=d)

	dists, norms = second_order ? _second_order(X, r; knndata=tree) : _first_order(X, r; knndata=tree)
	return dists, norms, ε
end

end # module BoundaryTest
