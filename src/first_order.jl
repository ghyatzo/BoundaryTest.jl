
"""
    _first_order(X::AbstractMatrix, r; knndata=nothing)

First order estimation of distance to border for boundary points in a point cloud.
It returns the estimated distances to the border and (outward) normals of all points, for a specified
radius `r`. The distances are valid only for points close to the border, and there is a delicate
interaction between the choice of radius `r` and `ε-border`.
If an existing NN tree is existing it can be passed to avoid the extra calculations.

!!! Warning
	This is a generic version that assumes familiarity with the paper and choice of parameters.
	Since it assumes that the radius passed will be in a context in which `ε` has been calculated as well.
"""
function _first_order(X::AbstractMatrix, r; knndata=nothing)
	# we want a datasets that has its points along the columns
	d, n = size(X)
	tree = isnothing(knndata) ? KDTree(X) : knndata

	border_dist = zeros(n)
	normals = similar(X)
	p = Progress(n; dt=1.0, showspeed=true)
	@inbounds Threads.@threads for i = 1:n
		# for the ith point we first find the other points within the ball B(xᵢ, r)
		idxs = inrange(tree, X[:, i], r)

		diffs = @views X[:, idxs] .- X[:, i]

		# νᵣ(xᵢ) = 1/n Σⱼ 1_{B(x₀,r)}(xᵢ) (xⱼ - xᵢ), normalised.
		# the minus is added here so we don't have to negate the diffs in the next step.
		normals[:, i] .= -normalize!(sum(diffs, dims=2))

		# dᵣ(xᵢ) = max_{xⱼ ∈ B(xᵢ,r)∩χ} -(xⱼ - xᵢ)⋅νᵣ(xᵢ)
		# dᵢ = mapreduce(d -> dot(d, view(normals, :, i)), max, eachcol(diffs))
		dᵢ = maximum(diffs' * view(normals, :, i))

		# return the test statistic for every point, so only one pass is necessary.
		border_dist[i] = dᵢ
		next!(p)
	end
	return border_dist, normals
end


function _first_order_manifold(X::AbstractMatrix, r, m; knndata=nothing)
	d, n = size(X)
	tree = isnothing(knndata) ? KDTree(X) : knndata

	border_dist = zeros(n)
	normals = similar(X)
	p = Progress(n; dt=1.0, showspeed=true)
	@inbounds Threads.@threads for i = 1:n
		# for the ith point we first find the other points within the ball B(xᵢ, r)
		idxs = inrange(tree, X[:, i], r)

		diffs = @views X[:, idxs] .- X[:, i]

		E = eigvecs(diffs*diffs')
		T = E[:, end:-1:end-(m-1)] # tangent subspace at Xi

		# νᵣ(xᵢ) = 1/n Σⱼ 1_{B(x₀,r)}(xᵢ) (xⱼ - xᵢ), normalised.
		# the minus is added here so we don't have to negate the diffs in the next step.
		# We can either project the diffs, or the normal vectors, the result calculation is the same
		# If Π is the projection onto the subspace spanned by T, then
		# Π(xj - xi) = (TT')(xj - xi)
		# => Π(xj-xi)⋅ν(xi) = (Π(xj-xi))'ν(xi) = (xj-xi)'(TT')ν(xi) = (xj-xi)⋅Π(ν(xi))
		normals[:, i] .= -normalize!(T*T'*sum(diffs, dims=2))

		# dᵣ(xᵢ) = max_{xⱼ ∈ B(xᵢ,r)∩χ} -(xⱼ - xᵢ)⋅νᵣ(xᵢ)
		dᵢ = mapreduce(d -> dot(d, view(normals, :, i)), max, eachcol(diffs))

		# return the test statistic for every point, so only one pass is necessary.
		border_dist[i] = dᵢ
		next!(p)
	end
	return border_dist, normals
end