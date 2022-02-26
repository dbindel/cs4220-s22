### A Pluto.jl notebook ###
# v0.18.1

using Markdown
using InteractiveUtils

# ╔═╡ 905f63ee-8ae5-11ec-28b6-cb745dc54324
using DelimitedFiles

# ╔═╡ 8f9e4ca5-ad6b-4499-801e-8dc90662cb86
using SparseArrays

# ╔═╡ f54d0437-09b0-46af-bc96-9e1d617f43d1
using SuiteSparse

# ╔═╡ 981a4df1-8b82-4713-b868-9b89afe7e7ad
using LinearAlgebra

# ╔═╡ 5709ae49-9634-41cf-800a-54586eec042e
md"""
# Proj 1: Harmonious learning

There are many problems that involve optimizing some objective
function by making local adjustments to a structure or graph.
For example:

- If we want to reinforce a truss with a limited budget, where
  should we add new beams (or strengthen old ones)?
- After a failure in the power grid, how should lines be either
  taken out of service or put in service to ensure no other lines
  are overloaded?
- In a road network, how will road closures or rate-limiting of
  on-ramps affect congestion (for better or worse)?
- In a social network, which edges are most critical to
  spreading information or influence to a target audience?

For our project, we will consider a simple method for 
*graph interpolation*.  We are given a (possibly weighted) undirected graph on
$n$ nodes, and we wish to determine some real-valued numerical property
at each node.  Given values at a few of the nodes, how should we fill in
the remaining values?  A natural approach that is used in some
semi-supervised machine learning approaches is to fill in the remaining
values by assuming that the value at an unlabeled node $i$ is the
(possibly weighted) average of the values at all neighbors of the node.
In this project, we will see how to quickly solve this problem, and how
to efficiently evaluate the sensitivity with respect to different types
of changes in the setup.  Of course, in the process we also want to
exercise your knowledge of linear systems, norms, and the like!
"""

# ╔═╡ 4fadd267-bc86-437a-9d22-155216538dc6
md"""
## Logistics

*You are encouraged to work in pairs on this project.*  You should
produce short report addressing the analysis tasks, and a few
short codes that address the computational tasks.  You may
use any Julia functions you might want.

Most of the code in this project will be short, but that does not make
it easy.  You should be able to convince both me and your partner that
your code is right.  A good way to do this is to test thoroughly.
Check residuals, compare cheaper or more expensive ways of computing
the same thing, and generally use the computer to make sure you don't
commit silly errors in algebra or coding.  You will also want to make
sure that you satisfy the efficiency constraints stated in the tasks.
"""

# ╔═╡ 05020a68-1819-46af-a6ec-204df17ad041
md"""
## Background

The (combinatorial) *graph Laplacian* matrix occurs often when
using linear algebra to analyze graphs.  For an undirected graph on
vertices $\{1, \ldots, n\}$, the weighted graph
Laplacian $L \in \mathbb{R}^{n \times n}$ has entries

$$l_{ij} = \begin{cases}
    -w_{ij}, & \mbox{ if } (i,j) \mbox{ an edge with weight } w_i \\
    d_{i} = \sum_k w_{ik}, & \mbox{i = j} \\
    0, & \mbox{otherwise}.
  \end{cases}$$

The unweighted case corresponds to $w_{ij} = -1$ and $d$ equal to the
node degree.

In our project, we seek to solve problems of the form

$$\begin{bmatrix}
    L_{11} & L_{12} \\
    L_{21} & L_{22}
  \end{bmatrix}
  \begin{bmatrix} u_1 \\ u_2 \end{bmatrix} =
  \begin{bmatrix} 0 \\ r_2 \end{bmatrix}$$

where the leading indices correspond to nodes in the graph at which
$u$ must be inferred (i.e. $u_1$ is an unknown) and the remaining
indices correspond to nodes in the graph at which $u$ is specified
(i.e. $u_2$ is known, though $r_2$ is not).  Note that if $i$ is an
index in the first block, then the equation at row $i$ specifies that

$$u_i = \frac{1}{d_i} \sum_{(i,j) \in \mathcal{E}} w_{ij} u_j,$$

i.e. the value at $i$ is a weighted average of the neighboring values.
"""

# ╔═╡ 44124714-238a-4fc5-8d1d-30907e14e47f
md"""
## Code setup

We will use the California road network data from the SNAP data set; to
retrieve it, download the `roadNet-CA.txt` file from the class web
page.  The following loader function will read in the topology and form
the graph Laplacian in compressed sparse column format (`SparseMatricCSC` in Julia).
This is a big enough network that you will *not*
want to form the graph Laplacian or related matrices in dense form.
On the other hand, because it is a moderate-sized planar graph, sparse
Cholesky factorization on $L$ will work fine.
"""

# ╔═╡ 9a7a702e-b15a-41bd-8f00-e3a553651a65
function load_network(fname)
	ij = readdlm(fname, '\t', Int)
	ij[:,1], ij[:,2]
end

# ╔═╡ b5b06ab0-46cd-42b9-8b2c-cdbaebae16d4
CA_I, CA_J = load_network("roadNet-CA.txt")

# ╔═╡ 91ece647-31a4-4cde-9eff-b4cb4d8b176e
function form_laplacian(I, J)
	n = maximum(I)
	nnz = length(I)

	# Compute node degree vector
	d = zeros(Int, n)
	for k = 1:nnz
		d[I[k]] += 1
	end
	
	# Form the adjacency
	A = sparse(I, J, ones(nnz))

	# Form the Laplacian
	L = spdiagm(d) - A
end

# ╔═╡ 6f255ee7-56d1-49f4-9220-629305d4bff9
md"""
For the tasks in this assignment, it is useful to carry around more than just the graph Laplacian.  We also want to keep track of which nodes have associated known values and what those values are.  For these purposes, it is helpful to use a Julia structure that we pass around.
"""

# ╔═╡ 4a41abc0-f681-4163-b388-fad4fa608238
begin
	# Julia structure for Laplacian with selective labels
	mutable struct LabeledLaplacian
		L :: SparseMatrixCSC{Float64,Int64}              # Laplacian storage
		LC :: Union{Nothing,SuiteSparse.CHOLMOD.Factor}  # Cholesky factor
		u :: Vector{Float64}                             # Node values
		active :: Vector{Bool}                           # Which dofs are active
		new_values :: Vector{Tuple{Int,Float64}}         # Updates to values
		new_weights :: Vector{Tuple{Int,Int,Float64}}    # Updates to edge weights
	end

	# Construct a LabeledLaplacian with no labels from a Laplacian
	function LabeledLaplacian(L :: SparseMatrixCSC{Float64,Int})
		n = size(L)[1]
		u = zeros(n)
		active = ones(Bool,n)
		LabeledLaplacian(L, nothing, u, active, [], [])
	end

	# Construct a LabeledLaplacian with no labels from coord form
	function LabeledLaplacian(I :: Vector{Int}, J :: Vector{Int})
		LabeledLaplacian(form_laplacian(I,J))
	end
	
	# Construct a LabeledLaplacian with no labels from a file name
	function LabeledLaplacian(fname :: String)
		I, J = load_network(fname)
		LabeledLaplacian(I, J)
	end
end

# ╔═╡ a3ba334a-e672-4c16-9d98-51dec6dfe088
md"""
We will set this up so that we can easily add node values and adjust edge weights.
We do this differently depending on whether or not we have already factored (part of) the Laplacian.  If we have an existing factorization, we will keep track of the updates that we would like to apply separately, and handle them via a bordered system approach described below.
"""

# ╔═╡ 87c324c7-f734-462e-abf7-614b6ca021e3
function new_value!(LL :: LabeledLaplacian, v, i)
	if LL.LC == nothing
		LL.u[i] = v
		LL.active[i] = false
	else
		push!(LL.new_values, (i,v))
	end
end

# ╔═╡ ba5a264d-5163-4d18-abe3-6690c9c6a971
function new_value!(LL :: LabeledLaplacian, V :: Vector, I :: Vector)
	for (v,i) in zip(V,I)
		new_value!(LL, v, i)
	end
end

# ╔═╡ 5694b9c1-1342-40ab-bba4-5a17d3f0f2a6
function update_edge!(LL :: LabeledLaplacian, v, i, j)
	if LL.LC == nothing
		LL.L[i,i] += v
		LL.L[j,j] += v
		LL.L[i,j] -= v
		LL.L[j,i] -= v
	else
		push!(LL.new_weights, (i,j,v))
	end
end

# ╔═╡ ad4d0c9a-5ab7-46cd-97d9-3e2cde6ef66b
function update_edge!(LL :: LabeledLaplacian, V :: Vector, I :: Vector, J :: Vector)
	for (v,i,j) in zip(V,I,J)
		update_edge!(LL, v, i, j)
	end
end

# ╔═╡ 8389cddb-726b-437b-a07d-9313e0f0cc7d
md"""
We also provide a `factor` routine to compute (or re-compute) the Cholesky factorization of the Laplacian matrix.  We only do the computation if there seems to be need; if there is an existing factorization and no updates have been made (new values or edge weight adjustments), we will keep the existing factorization as is.
"""

# ╔═╡ 0f7ef41b-00cf-42b8-8932-af05b7cfd195
function factor!(LL :: LabeledLaplacian)

	# Short-circuit: no need to factor if already done (and no updates)
	if LL.LC != nothing && isempty(LL.new_values) && isempty(LL.new_weights)
		return
	end

	# Clear any existing factorization
	LL.LC = nothing

	# Make sure any updates are merged
	for (i,v) in LL.new_values
		LL.u[i] = v
		LL.active[i] = false
	end
	for (i,j,v) in LL.new_weights
		LL.L[i,i] += v
		LL.L[j,j] += v
		LL.L[i,j] -= v
		LL.L[j,i] -= v
	end
	LL.new_values = []
	LL.new_weights = []

	# Compute the Cholesky factorization
	LL.LC = cholesky(LL.L[LL.active, LL.active])

end

# ╔═╡ e22facfe-b7cb-4d20-89a0-171714995198
md"""
Finally, we provide a residual check routine to provide some reassurance about the correctness of our solutions.
"""

# ╔═╡ c8f16076-429f-4445-b97b-3787b6f0682c
function residual(LL :: LabeledLaplacian)

	# Compute residual with un-adjusted L
	r = LL.L * LL.u

	# Add terms associated with edge weight updates
	for (i,j,v) in LL.new_weights
		dij = v*(LL.u[i]-LL.u[j])
		r[i] += dij
		r[j] -= dij
	end

	# Ignore entries for inactive dofs
	for k = 1:length(LL.active)
		if !LL.active[k]
			r[k] = 0.0
		end
	end

	# And separately track errors for newly-inactive dofs
	for (i,v) in LL.new_values
		r[i] = LL.u[i]-v
	end

	r

end

# ╔═╡ 2d3c924d-5645-4c7b-989b-fd6bb4b7ce0b
md"""
And we provide some helper functions for working with the `LabeledLaplacian` objects.
"""

# ╔═╡ 5fc7bfdf-5899-47fe-a677-16cff7d14007
begin

	# Helpers to get sizes of different pieces
	ntotal(LL :: LabeledLaplacian) = length(LL.active)
	nmod_val(LL :: LabeledLaplacian) = length(LL.new_values)
	nmod_wts(LL :: LabeledLaplacian) = length(LL.new_weights)
	nmod(LL :: LabeledLaplacian) = nmod_val(LL) + nmod_wts(LL)
	nactive(LL :: LabeledLaplacian) = sum(LL.active)

	# Get active/inactive index sets
	active(LL :: LabeledLaplacian) = LL.active
	inactive(LL :: LabeledLaplacian) = .!LL.active

end

# ╔═╡ 09d20ee3-4cc7-433d-9094-6338a208d87e
md"""
### Task 1

For the first part of the assignment, we will improve on a naive `solve!` command (given below) that always forces a re-factorization.
"""

# ╔═╡ 33487051-1c76-49e9-b9b4-af7e1103b030
function solve!(LL :: LabeledLaplacian)

	# Force refactorization
	factor!(LL)

	# Index sets for reference unknown and known pieces
	Iu = active(LL)
	Ik = inactive(LL)

	# Compute RHS and do Cholesky solve
	rhs = LL.L[Iu,Ik] * LL.u[Ik]
	LL.u[Iu] = -(LL.LC \ rhs)
	
end

# ╔═╡ 06cf9074-086e-4008-a2cd-00123e0bbbf0
md"""
Our modified version of `solve!` will let us adapt to new values or edge weight updates *without* recomputing the Cholesky factorization.  We can do this by computing a bordered linear system

$$\begin{bmatrix}
    L_{11} & L_{12} & B_1 \\
    L_{21} & L_{22} & B_2 \\
    B_1^T & B_2^T & C
  \end{bmatrix}
  \begin{bmatrix}
    u_1 \\ u_2 \\ w
  \end{bmatrix} =
  \begin{bmatrix}
    0 \\ r_2 \\ f
  \end{bmatrix}.$$

To enforce additional boundary conditions, we use each column of $B_1$
to indicate a node to constrain, and let the corresponding entry of $f$
be the value at that node.  To adjust the weight of an edge $(i, j)$ by $s$,
note that the Laplacian for the new graph would be

$$L' = L + s (e_i-e_j) (e_i-e_j)^T,$$

and we can write $L'u$ as $Lu + (e_i-e_j) \gamma$ where
$\gamma = s (e_i-e_j)^T u$.  Using this observation, 
we can form a bordered system that incorporates
edge weight modifications as well as additional boundary conditions,
all without re-computing any large sparse factorizations.

I split this code into two pieces: a `compute_bordering` function that
produces `B`, `C`, and `f` in the system above, and the `solve!` function 
hat solves the actual system by block Gaussian elimination.

Your updated code should take $O(k)$ linear solves with the existing factorization
to account for $k$ updates, whether new assignments of node values or adjustments
to graph edge weights.
"""

# ╔═╡ 0dc50473-e72d-434e-987b-9563a3e45eba
md"""
#### Sanity checking

We provide a simple test case to check the correctness of our bordered system
approach.  This will start off correct (giving small residual values) for the naive
"refactor every time" approach; you should ideally keep the residuals small while
improving the speed!
"""

# ╔═╡ 68cd9c6b-0ca2-481b-9636-4e134f125b0f
function test_task1(CA_I, CA_J)
	LLCA = LabeledLaplacian(CA_I, CA_J)

	# Add a first batch of values and solve
	new_value!(LLCA, [1.0, 2.0, 3.0], [1, 10, 20])
	t1 = @elapsed solve!(LLCA)
	r1 = norm(residual(LLCA))

	# Add some more node values and re-solve
	new_value!(LLCA, [4.0, 5.0], [30, 40])
	t2 = @elapsed solve!(LLCA)
	r2 = norm(residual(LLCA))

	# Update an edge weight (delete the edge 1-2) and re-solve
	update_edge!(LLCA, -1.0, 1, 2)
	t3 = @elapsed solve!(LLCA)
	r3 = norm(residual(LLCA))

	md"""
	- Initial solve: $(t1) s, residual norm $(r1)
	- New values:    $(t2) s, residual norm $(r2)
	- Edge update:   $(t3) s, residual norm $(r3)  
	"""
end

# ╔═╡ b2e251d0-a7f7-4dae-959d-761eddc8eaa2
test_task1(CA_I, CA_J)

# ╔═╡ cfa9a5a8-9053-460b-9fec-63a81d8b0127
md"""
#### Additional questions

1. We have to assign some values before we are able to run the solver.  Why can't we safely factor the full Laplacian immediately?
2. The largest and smallest entries of the solution vector $u$ should always be entries where we've specified values.  Why is this?
"""

# ╔═╡ dbcbcdc8-c301-4ccc-87cb-51b9453fb70a
md"""
### Task 2

Again using the bordered system idea from the first part, we now want to consider the problem of *leave-one-out cross-validation* of the assigned values at the nodes.  That is, for a given node $j$ that has an assigned value $u_j$, we would like to compare $u_j$ to the value $u_j^{(-j)}$ we would have inferred if all the data but that value at node $j$ were provided.

Complete the `cross_validate` function below to return the difference
$u_j-u_j^{(-j)}$.  As in the previous task, your code should *not* require
a new matrix factorization.  You should use the sanity check to make sure
you have the right answer.

A useful building block will be a version of the solver code that solves systems $L_{11} x = b$ for general right hand sides via the bordered system

$$\begin{bmatrix} L_{11} & B_1 \\ B_1^T & C \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} b \\ 0 \end{bmatrix}$$

Once we have this building block, it is convenient let $z = u-u^{(-j)}$, 
and think of splitting the boundary nodes into group 2 
(consisting just of node $j$) and group 3 (all the other boundary nodes).
In our Julia framework, that means block 1 is associated with `active`, 
block 2 is `j`, and block 3 is the rest of `.!active`.  We know
that $u$ and $u^{(-j)}$ satisfy

$$\begin{bmatrix} 
  L_{11} & l_{12} & L_{13} \\ 
  l_{21} & l_{22} & l_{23} \\
  L_{31} & L_{32} & l_{33}
\end{bmatrix}
\begin{bmatrix} u_1 \\ u_2 \\ u_3 \end{bmatrix} = 
\begin{bmatrix} 0 \\ r_2 \\ r_3 \end{bmatrix},
\begin{bmatrix} 
  L_{11} & l_{12} & L_{13} \\ 
  l_{21} & l_{22} & l_{23} \\
  L_{31} & L_{32} & l_{33}
\end{bmatrix}
\begin{bmatrix} u_1^{(-j)} \\ u_2^{(-j)} \\ u_3^{(-j)} \end{bmatrix} = 
\begin{bmatrix} 0 \\ 0 \\ \tilde{r}_3 \end{bmatrix}$$

and subtracting the two equations gives

$$\begin{bmatrix} 
  L_{11} & l_{12} & L_{13} \\ 
  l_{21} & l_{22} & l_{23} \\
  L_{31} & L_{32} & l_{33}
\end{bmatrix}
\begin{bmatrix} z_1 \\ z_2 \\ z_3 \end{bmatrix} = 
\begin{bmatrix} 0 \\ r_2 \\ r_3-\tilde{r}_3 \end{bmatrix}$$

where $z_3$ is by definition zero (since both $u$ and $u^{(-j)}$ agree on all the boundary nodes other than node $j$).  Therefore, we have

$$\begin{bmatrix} 
  L_{11} & l_{12} \\ 
  l_{21} & l_{22} 
\end{bmatrix}
\begin{bmatrix} z_1 \\ z_2 \end{bmatrix} = 
\begin{bmatrix} 0 \\ r_2 \end{bmatrix}$$

and eliminating the first block gives the system

$$(l_{22} - l_{21} L_{11}^{-1} l_{12}) z_2 = r_2.$$

Note that $z_2$ is precisely the cross-validation value that we've described.
"""

# ╔═╡ 891bd87d-961c-44a8-96cc-eb67fcd00224
function cross_validate(LL :: LabeledLaplacian, j)
	if !LL.active[j]
		0.0 # Replace this with the cross-validation computation
	else
		0.0 # This is correct if j didn't have an assigned value anyhow
	end
end

# ╔═╡ 3afb9b73-6b5a-4c05-9ee1-09ebfe764bcb
md"""
The cross-validation can be done in about the same time as the fast solves described
before using the bordered solver approach.  We give two tests to compare the fast
approach against a reference computation (the second a little harder than the first). 
The reference version takes a few seconds on my machine (vs a fraction of a second).
"""

# ╔═╡ dfde5ff2-7ff2-4bb0-a100-0046e93c0d0c
function test_cross_validate1()
	LLCA = LabeledLaplacian(CA_I, CA_J)

	# Add a first batch of values and solve
	new_value!(LLCA, [1.0, 2.0, 3.0], [1, 10, 20])
	solve!(LLCA)
	u30_ref = LLCA.u[30]

	# Add one more node and re-solve with forced factorization
	new_value!(LLCA, [4.0], [30])
	factor!(LLCA)
	solve!(LLCA)
	
	# Run cross-validation and compare with the expensive version
	zref = 4.0-u30_ref
	t1 = @elapsed begin zcv = cross_validate(LLCA, 30) end

	md"""
	- Slow computation: $(zref)
	- Fast computation: $(zcv)
	- Relerr: $(abs(zref-zcv)/abs(zref))
	- Time (fast): $(t1)
	"""
end

# ╔═╡ 6d34ed7c-0b58-4bb1-aeff-94f0be7a990d
test_cross_validate1()

# ╔═╡ 4723b7fb-499b-4bf9-85a5-79a978c303eb
function test_cross_validate2()
	LLCA = LabeledLaplacian(CA_I, CA_J)

	# Add a first batch of values and solve
	new_value!(LLCA, [1.0, 2.0, 3.0], [1, 10, 20])
	solve!(LLCA)
	u30_ref = LLCA.u[30]

	# Add one more node and re-solve with forced factorization
	new_value!(LLCA, [4.0], [30])
	factor!(LLCA)
	solve!(LLCA)

	LLCA2 = LabeledLaplacian(CA_I, CA_J)

	# Add values in two batches to sanity check this works with bordered solver
	new_value!(LLCA2, [1.0, 2.0, 4.0], [1, 10, 30])
	solve!(LLCA2)
	new_value!(LLCA2, [3.0], [20])
	solve!(LLCA2)
	
	# Run cross-validation and compare with the expensive version
	zref = 4.0-u30_ref
	t1 = @elapsed begin zcv = cross_validate(LLCA, 30) end

	md"""
	- Slow computation: $(zref)
	- Fast computation: $(zcv)
	- Relerr: $(abs(zref-zcv)/abs(zref))
	- Time (fast): $(t1)
	"""
end

# ╔═╡ 3d58b1e5-6bf3-4917-9a3e-041ca759dd48
test_cross_validate2()

# ╔═╡ 7c5d46eb-e7ce-43b6-8066-011e8619ecb4
md"""
### Task 3

Using bordered systems lets us recompute the solution quickly after we
adjust the edge weights.  But what if we want to compute the sensitivity
of the value at some target node to small changes to {\em any} of the
edges?  That is, for a target node $k$, we think of $u_k$ as a function
of all the edge weights, and compute the sparse sensitivity matrix

$$S_{ij} =
  \begin{cases}
    \frac{\partial u_k}{\partial w_{ij}}, & (i,j) \in \mathcal{E} \\
    0, & \mbox{otherwise}.
  \end{cases}$$

Assuming the $u$ vector has already been computed, the sensitivity
computation requires constant work per edge after one additional
linear solve.  Fill in `edge_sensitivity` to carry out this
computation.  Note that you should not require new factorizations if
you already have one; that is, your code should ideally use the bordered
system formalism to incorporate any new boundary conditions or
edge updates added to the system since the last factorization.

As in task 2, you should also provide a sanity check code.
"""

# ╔═╡ 64c99435-dadd-4528-afc3-a1a178ccaf0f
function edge_sensitivity(LL :: LabeledLaplacian, k)
	# Computes a sparse matrix of sensitivities of u_k to the weight on each edge
	I, J, _ = findnz(LL.L)
	SIJ = I .+ J           # Placeholder --you should change!
	sparse(I, J, SIJ)
end

# ╔═╡ e89c0ab6-b522-4399-9f23-c44c118ab734
function test_edge_sensitivity()
	LLCA = LabeledLaplacian(CA_I, CA_J)

	# Add values in two batches to sanity check this works with bordered solver
	new_value!(LLCA, [1.0, 2.0, 4.0], [1, 10, 30])
	solve!(LLCA)
	new_value!(LLCA, [3.0], [20])
	solve!(LLCA)

	# Do full computation
	t1 = @elapsed begin S = edge_sensitivity(LLCA, 100) end

	# Try adjusting the weight from 13 to 14 and finite difference check
	t2 = @elapsed begin
		h = 1e-4
		u100 = LLCA.u[100]
		update_edge!(LLCA, h, 13, 14)
		solve!(LLCA)
		u100p = LLCA.u[100]
		fd = (u100p-u100)/h
	end

	md"""
	- Fast sensitivity on (13,14): $(S[13,14])
	- Slow edge sensitivity on (13,14): $(fd)
	- Relerr: $(abs(S[13,14]-fd)/abs(fd))
	- Elapsed time: $(t1)
	- Estimated via bordered solves: $(t2*nnz(LLCA.L)/2)
	"""
end

# ╔═╡ 8ff34d78-11e9-452c-bfaa-ec0cf218e5df
test_edge_sensitivity()

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DelimitedFiles = "8bb1440f-4735-579b-a4ab-409b98df4dab"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
SuiteSparse = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.2"
manifest_format = "2.0"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
"""

# ╔═╡ Cell order:
# ╟─5709ae49-9634-41cf-800a-54586eec042e
# ╟─4fadd267-bc86-437a-9d22-155216538dc6
# ╟─05020a68-1819-46af-a6ec-204df17ad041
# ╟─44124714-238a-4fc5-8d1d-30907e14e47f
# ╠═905f63ee-8ae5-11ec-28b6-cb745dc54324
# ╠═8f9e4ca5-ad6b-4499-801e-8dc90662cb86
# ╠═f54d0437-09b0-46af-bc96-9e1d617f43d1
# ╠═981a4df1-8b82-4713-b868-9b89afe7e7ad
# ╠═9a7a702e-b15a-41bd-8f00-e3a553651a65
# ╠═b5b06ab0-46cd-42b9-8b2c-cdbaebae16d4
# ╠═91ece647-31a4-4cde-9eff-b4cb4d8b176e
# ╟─6f255ee7-56d1-49f4-9220-629305d4bff9
# ╠═4a41abc0-f681-4163-b388-fad4fa608238
# ╟─a3ba334a-e672-4c16-9d98-51dec6dfe088
# ╠═87c324c7-f734-462e-abf7-614b6ca021e3
# ╠═ba5a264d-5163-4d18-abe3-6690c9c6a971
# ╠═5694b9c1-1342-40ab-bba4-5a17d3f0f2a6
# ╠═ad4d0c9a-5ab7-46cd-97d9-3e2cde6ef66b
# ╟─8389cddb-726b-437b-a07d-9313e0f0cc7d
# ╠═0f7ef41b-00cf-42b8-8932-af05b7cfd195
# ╟─e22facfe-b7cb-4d20-89a0-171714995198
# ╠═c8f16076-429f-4445-b97b-3787b6f0682c
# ╟─2d3c924d-5645-4c7b-989b-fd6bb4b7ce0b
# ╠═5fc7bfdf-5899-47fe-a677-16cff7d14007
# ╟─09d20ee3-4cc7-433d-9094-6338a208d87e
# ╠═33487051-1c76-49e9-b9b4-af7e1103b030
# ╟─06cf9074-086e-4008-a2cd-00123e0bbbf0
# ╟─0dc50473-e72d-434e-987b-9563a3e45eba
# ╠═68cd9c6b-0ca2-481b-9636-4e134f125b0f
# ╠═b2e251d0-a7f7-4dae-959d-761eddc8eaa2
# ╟─cfa9a5a8-9053-460b-9fec-63a81d8b0127
# ╟─dbcbcdc8-c301-4ccc-87cb-51b9453fb70a
# ╠═891bd87d-961c-44a8-96cc-eb67fcd00224
# ╟─3afb9b73-6b5a-4c05-9ee1-09ebfe764bcb
# ╠═dfde5ff2-7ff2-4bb0-a100-0046e93c0d0c
# ╠═6d34ed7c-0b58-4bb1-aeff-94f0be7a990d
# ╠═4723b7fb-499b-4bf9-85a5-79a978c303eb
# ╠═3d58b1e5-6bf3-4917-9a3e-041ca759dd48
# ╟─7c5d46eb-e7ce-43b6-8066-011e8619ecb4
# ╠═64c99435-dadd-4528-afc3-a1a178ccaf0f
# ╠═e89c0ab6-b522-4399-9f23-c44c118ab734
# ╠═8ff34d78-11e9-452c-bfaa-ec0cf218e5df
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
