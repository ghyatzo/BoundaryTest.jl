
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
function _second_order(X::AbstractMatrix, r; knndata=nothing, cutoff=true)
	@assert(r > 0)

	# we want a datasets that has its points along the columns
	d, n = size(X)
	tree = isnothing(knndata) ? KDTree(X) : knndata = KDTree(X; reorder=true)

	border_dist = zeros(n)
	normals = similar(X)
	rNNs = Vector{Vector{Int}}(undef, n)
	T = eltype(normals)

	p_norms = Progress(n, dt=1.0, desc="Estimating normals...", showspeed=true)
	# compute all normals
	@inbounds Threads.@threads for i = 1:n
		rNNs[i] = inrange(tree, X[:, i], r)
        # θ(xᵢ) = 1/(n*ω_d)(2/r)^d Σⱼ 1_{B(xᵢ, r/2)}(xⱼ)
        # νᵣ(x₀) = 1/n Σᵢ 1_{B(x₀,r)}(xᵢ)/θ(xᵢ) (xᵢ - x₀), normalised.
        diffs = @views (X[:, rNNs[i]] .- X[:, i])
        @inbounds for j in 1:size(diffs, 2)
            view(diffs, :, j) ./= inrangecount(tree, X[:, rNNs[i][j]], r/2)
        end

        normals[:, i] .= -normalize!(sum(diffs, dims=2))
        next!(p_norms)
	end

	p_dist = Progress(n, dt=1.0, desc="Estimating Distances...", showspeed=true)
	# average all normals and compute the distance estimation
	@inbounds Threads.@threads for i = 1:n
		diffs = @views X[:, rNNs[i]] .- X[:, i]

		if cutoff
			avg_ν = zeros(d, length(rNNs[i]))
			# dᵣ(x₀) = max_{xᵢ ∈ B(x₀,r)∩χ} -(xᵢ - x₀) ⋅ [ νᵣ(x₀) + 0.5*(νᵣ(xᵢ)-νᵣ(x₀))1_{R+}(νᵣ(xᵢ)⋅νᵣ(x₀)) ]
			@inbounds for (j, jj) in enumerate(rNNs[i])
			    νⱼ, νᵢ = @views normals[:, jj], normals[:, i]
			    local d = ifelse(dot(νⱼ, νᵢ) > zero(T), one(T), zero(T))
			    avg_ν[:, j] .= νᵢ .+ ((νⱼ .- νᵢ) .* (0.5 .* d))
			end
		else
			# dᵣ(x₀) = max_{xᵢ ∈ B(x₀,r)∩χ} -(xᵢ - x₀) ⋅ 0.5*(νᵣ(xᵢ)+νᵣ(x₀))
			avg_ν = @views (normals[:, rNNs[i]] .+ normals[:, i]) .* 0.5
		end

		border_dist[i] = maximum(dot.(eachcol(diffs), eachcol(avg_ν)))
		next!(p_dist)
	end

	# return the test statistic for every point, so only one pass is necessary.
	return border_dist, normals
end


function _second_order_manifold(X::AbstractMatrix, r, m; knndata=nothing, cutoff=true)
	@assert(r > 0)

	# we want a datasets that has its points along the columns
	d, n = size(X)
	tree = isnothing(knndata) ? KDTree(X) : knndata = KDTree(X; reorder=true)

	border_dist = zeros(n)
	normals = similar(X)
	rNNs = Vector{Vector{Int}}(undef, n)
	P = zeros(d, m)
	T = eltype(normals)

	# compute all normals
	p_norms = Progress(n, dt=1.0, desc="Estimating normals...", showspeed=true)
	@views @inbounds Threads.@threads for i = 1:n
		rNNs[i] = inrange(tree, X[:, i], r)
        # θ(xᵢ) = 1/(n*ω_d)(2/r)^d Σⱼ 1_{B(xᵢ, r/2)}(xⱼ)
        # νᵣ(x₀) = 1/n Σᵢ 1_{B(x₀,r)}(xᵢ)/θ(xᵢ) (xᵢ - x₀), normalised.
        diffs = X[:, rNNs[i]] .- X[:, i]

        @inbounds for j in 1:size(diffs, 2)
            diffs[:, j] ./= inrangecount(tree, X[:, rNNs[i][j]], r/2)
        end

        E = eigvecs(diffs*diffs')
		P = E[:, end:-1:end-(m-1)]

        normals[:, i] .= -normalize!(P * P' * sum(diffs, dims=2))
        next!(p_norms)
	end

	# average all normals and compute the distance estimation
	p_dist = Progress(n, dt=1.0, desc="Estimating Distances...", showspeed=true)
	@views @inbounds Threads.@threads for i = 1:n
		diffs = X[:, rNNs[i]] .- X[:, i]

		if cutoff
			avg_ν = zeros(d, length(rNNs[i]))
			#[ νᵣ(x₀) + 0.5*(νᵣ(xᵢ)-νᵣ(x₀))1_{R+}(νᵣ(xᵢ)⋅νᵣ(x₀)) ]
			@inbounds for (j, jj) in enumerate(rNNs[i])
			    νⱼ, νᵢ = normals[:, jj], normals[:, i]
			    local d = ifelse(dot(νⱼ, νᵢ) > zero(T), one(T), zero(T))
			    avg_ν[:, j] .= νᵢ .+ ((νⱼ .- νᵢ) .* (0.5 .* d))
			end
		else
			# dᵣ(x₀) = max_{xᵢ ∈ B(x₀,r)∩χ} -(xᵢ - x₀) ⋅ 0.5*(νᵣ(xᵢ)+νᵣ(x₀))
			avg_ν = (normals[:, rNNs[i]] .+ normals[:, i]) .* 0.5
		end
		# dᵣ(x₀) = max_{xᵢ ∈ B(x₀,r)∩χ} -(xᵢ - x₀) ⋅ U(i,0)
		# border_dist[i] = mapreduce(i -> dot(view(diffs, :, i), view(avg_ν, :, i)), max, 1:size(diffs, 2))
		border_dist[i] = maximum(dot.(eachcol(diffs), eachcol(avg_ν)))
		next!(p_dist)
	end

	# return the test statistic for every point, so only one pass is necessary.
	return border_dist, normals
end