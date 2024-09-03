

function _average_normal_cutoff(i, idxs, normals::AbstractMatrix{T}) where T
	avg_ν = Matrix{T}(undef, size(normals,1), length(idxs[]))
	νi = view(normals, :, i)
	# we're passing the idxs (a vector of vectors) as a view
	for (j, id) in enumerate(idxs[])
		νj = view(normals, :, id)
		d = ifelse(dot(νj, νi) > zero(T), one(T), zero(T))
		avg_ν[:, j] .= νi .+ ((νj .- νi) .* (0.5 .* d))
	end
	avg_ν
end


function _estimate_distance_cutoff(i, idxs, data::T, normals::T) where {T <: AbstractMatrix{<:Real}}
	avg_ν = _average_normal_cutoff(i, idxs, normals)
	diffs = @views data[:, idxs[]] .- data[:, i]
	return maximum(dot.(eachcol(diffs), eachcol(avg_ν)))
end

function _estimate_distance(i, idxs, data::T, normals::T) where {T <: AbstractMatrix{<:Real}}
	avg_ν = @views (normals[:, idxs[]] .+ normals[:, i]) .* 0.5
	diffs = @views data[:, idxs[]] .- data[:, i]
	return maximum(dot.(eachcol(diffs), eachcol(avg_ν)))
end

"""
    _second_order(X::AbstractMatrix, r ; cutoff=true)

Second order estimation of distance to the border for boundary points in a point cloud.
	It returns the estimated distances and (outward normals) of all points, for a specified
radius `r`. The distances are valid only for points close to the border, and there is a delicate
interaction between the choice of radius `r` and `ε-border`.
If an existing NN tree is existing it can be passed to avoid the extra calculations.

!!! Warning
	This is a generic version that assumes familiarity with the paper and choice of parameters.
	Since it assumes that the radius passed will be in a context in which `ε` has been calculated as well.

A `cutoff` keyword can be specified, it is on by default as it reduces the number of
normals evaluations and makes the algorithm more solid against negative curvature.
But it can also lead to reduces accuracy.
"""
function _second_order(X::AbstractMatrix{T}, r; knndata=nothing, cutoff=true) where T
	@assert(r > 0)

	# we want a datasets that has its points along the columns
	d, n = size(X)
	tree = isnothing(knndata) ? KDTree(X) : knndata = KDTree(X; reorder=true)

	border_dist = zeros(n)
	normals = similar(X)
	rNNs = Vector{Vector{Int}}(undef, n)

	p_norms = Progress(n, dt=1.0, desc="Estimating normals...", showspeed=true)
	# compute all normals
	@inbounds Threads.@threads for i = 1:n
		rNNs[i] = inrange(tree, view(X, :, i), r)
		# θ(xᵢ) = 1/(n*ω_d)(2/r)^d Σⱼ 1_{B(xᵢ, r/2)}(xⱼ)
		# νᵣ(x₀) = 1/n Σᵢ 1_{B(x₀,r)}(xᵢ)/θ(xᵢ) (xᵢ - x₀), normalised.
		diffs = @views X[:, rNNs[i]] .- X[:, i]
		for j in 1:size(diffs, 2)
			@inbounds view(diffs, :, j) ./= @views inrangecount(tree, X[:, rNNs[i][j]], r/2)
		end

		@inbounds normals[:, i] .= -normalize!(sum(diffs, dims=2))
		next!(p_norms)
	end

	p_dist = Progress(n, dt=1.0, desc="Estimating Distances...", showspeed=true)
	# average all normals and compute the distance estimation
	@inbounds Threads.@threads for i = 1:n
		border_dist[i] = ifelse(cutoff,
			_estimate_distance_cutoff,
			_estimate_distance)(i, view(rNNs, i), X, normals)
		next!(p_dist)
	end

	# return the test statistic for every point, so only one pass is necessary.
	return border_dist, normals
end


function _second_order_manifold(X::AbstractMatrix{T}, r, m; knndata=nothing, cutoff=true) where T
	@assert(r > 0)

	# we want a datasets that has its points along the columns
	d, n = size(X)
	tree = isnothing(knndata) ? KDTree(X) : knndata = KDTree(X; reorder=true)

	border_dist = zeros(n)
	normals = similar(X)
	rNNs = Vector{Vector{Int}}(undef, n)
	P = zeros(d, m)

	# compute all normals
	p_norms = Progress(n, dt=1.0, desc="Estimating normals...", showspeed=true)
	@views @inbounds Threads.@threads for i = 1:n
		rNNs[i] = inrange(tree, view(X, :, i), r)
		# θ(xᵢ) = 1/(n*ω_d)(2/r)^d Σⱼ 1_{B(xᵢ, r/2)}(xⱼ)
		# νᵣ(x₀) = 1/n Σᵢ 1_{B(x₀,r)}(xᵢ)/θ(xᵢ) (xᵢ - x₀), normalised.
		diffs = @views X[:, rNNs[i]] .- X[:, i]

		for j in axes(diffs, 2)
			@inbounds diffs[:, j] ./= @views inrangecount(tree, X[:, rNNs[i][j]], r/2)
		end

		E = eigvecs(diffs*diffs')
		P = E[:, end:-1:end-(m-1)]

		normals[:, i] .= -normalize!(P * P' * sum(diffs, dims=2))
		next!(p_norms)
	end

	# average all normals and compute the distance estimation
	p_dist = Progress(n, dt=1.0, desc="Estimating Distances...", showspeed=true)
	@inbounds Threads.@threads for i = 1:n
		border_dist[i] = ifelse(cutoff,
			_estimate_distance_cutoff,
			_estimate_distance)(i, view(rNNs, i), X, normals)
		next!(p_dist)
	end

	# return the test statistic for every point, so only one pass is necessary.
	return border_dist, normals
end