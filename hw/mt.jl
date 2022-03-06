### A Pluto.jl notebook ###
# v0.18.1

using Markdown
using InteractiveUtils

# ╔═╡ 1fb95f1d-a351-4ee0-a53b-1c836cf4882e
using LinearAlgebra

# ╔═╡ db885ec4-98e2-11ec-30bb-eb078b76cb06
md"""
# Midterm

You may use whatever reading materials you want -- the course notes should be
sufficient, but you may also look online or at books.  Make sure you cite any
sources beyond the course notes.  For this exam, you may ask questions of the
course staff, but you should not work with others, whether inside or outside the
class.
"""

# ╔═╡ bb1f0b7c-b648-4f2a-8b64-52a65f2f34fa
md"""
#### 1: Efficient operations (5 points)

Rewrite each expression to have the indicated complexity.
"""

# ╔═╡ 3dba96b2-c65e-4299-a935-b665c2191c3f
begin

	# 1 point: Rewrite to be O(1) time
	function p1a(x, k)
		ek = zeros(length(x))
		ek[k] = 1.0
		ek'*x
	end

	# 1 point: Rewrite to be O(n) time
	function p1b(d, u, v, x)
		(Diagonal(d)+u*v')*x
	end

	# 1 point: Rewrite to be O(n^2) time
	function p1c(A, u, v, x)
		A*u*v'*A*x
	end

	# 2 points: Rewrite this to take O(mn^2) time rather than O((m+n)^3)
	function p1d(A, b)
		m, n = size(A)
		M = [I A; A' zeros(n,n)]
		rx = M\[b; zeros(n)]
	end

end

# ╔═╡ 2023ed87-239f-4be4-902d-df34f1a49b7a
md"""
#### 2: Floating point fiddling (5 points)

Rewrite each expression or better floating point error.
"""

# ╔═╡ 4d9a417c-bfd2-4ce0-8ddc-ea8b4f98166c
begin

	# 1 point: Rewrite for good accuracy when z is near zero
	function p2a(z)
		1.0-cos(z)
	end

	# 1 point: Rewrite for good accuracy when z is large
	function p2b(z)
		1.0/(1.0+z)+1.0/(1.0-z)
	end

	# 3 points: Rewrite to avoid overflow and to maintain good accuracy for small U;
	#    you may assume U is upper triangular and all non-negative.  One point here
	#    is for complexity -- you should be able to do this in O(n) time.
	#    Hint: you may want to look up log1p (the inverse of expm1 from HW)
	function p2c(U)
		log(det(I+U))
	end

end

# ╔═╡ f1e11c5e-0c7d-452b-9194-c9752848571a
md"""
#### 3: Normwise nonsense (5 points)

1. (1 point) Argue that $\|Ax\|_\infty \leq \left( \max_{ij} |a_{ij}| \right) \|x\|_1$.
2. (2 points) If $PA = LU$ is computed using the usual partial pivoting strategy, argue that $\|A\|_1 \leq n \|U\|_1$.
3. (2 points) The matrix exponential can be defined formally as $\exp(A) = \sum_{k=0}^\infty \frac{1}{k!} A^k$.  Argue that for any operator norm, $\|\exp(A)\| \leq \exp(\|A\|)$.
"""

# ╔═╡ 5ff3caf7-76b2-4467-bc68-3311ca5f34da
md"""
#### 4: Delightful derivatives (4 points)

Let $A \in \mathbb{R}^{m \times n}$ be well conditioned with $m > n$, and consider
$x(s) = (A+sE)^\dagger b$ for some given $E \in \mathbb{R}^{m \times n}$.
Complete the following code for computing $x(0)$ and $dx(0)/ds$ (you may use the normal equations approach).
"""

# ╔═╡ 0fdf5a9d-3aed-4458-8590-f2494279689b
function deriv_ls(A, E, b)

	# Setup: your code should include these two lines
	m, n = size(A)
	F = cholesky(A'*A)

	# TO DO: Replace the following two lines with your code
	x = A\b        # Rewrite for O(mn) time using F (1 point)
	dx = zeros(n)  # Rewrite to be correct and take O(mn) time (3 points)

	x, dx
end

# ╔═╡ 7b15fcd6-0701-4ae2-9ef7-866392e4cc99
function check_deriv_ls()

	# Finite difference check on derivative
	h = 1e-6
	A = rand(10,3)
	E = rand(10,3)
	b = rand(10)

	xp = (A+h*E)\b
	xm = (A-h*E)\b
	dx_fd = (xp-xm)/(2h)

	x, dx = deriv_ls(A, E, b)
	norm(dx-dx_fd)/norm(dx)

end

# ╔═╡ 2cef793b-586b-4e82-8784-1d48be9e7822
check_deriv_ls()

# ╔═╡ cbb8b2c1-2256-43ca-8e48-f372604088c2
md"""
#### 5: Extending Cholesky (3 points)

Consider the block matrix

$$M = \begin{bmatrix} A & b \\ b^T & d \end{bmatrix}$$

Complete the following code to extend a Cholesky factorization $A = R^T R$ to a Cholesky factor of $M$.  Your code should take $O(n^2)$ time where $A \in \mathbb{R}^{n \times n}$
"""

# ╔═╡ b64b54f0-caf2-4139-934e-cbf7f21349d3
function extend_cholesky(R, b, d)

	# Placeholder: does the job, but costs O(n^3)
	#   Replace with your own code (3 points)
	M = [R'*R b; b' d]
	F = cholesky(M)
	F.U

end

# ╔═╡ cd34f600-18ea-4cff-992d-584fb91e6b09
md"""
You may use the following code to sanity check your solution.
"""

# ╔═╡ e5cef97e-822d-4305-9bcf-35c019e31ea5
function test_extend_cholesky()

	W = rand(10,6)
	M = W'*W
	A = M[1:5,1:5]
	b = M[1:5,end]
	d = M[end,end]
	FA = cholesky(A)
	R = extend_cholesky(FA.U, b, d)
	norm(R'*R-M)/norm(M)

end

# ╔═╡ 8901acef-4c17-4126-bd4a-28a5bd49aba6
test_extend_cholesky()

# ╔═╡ 2ff3b061-fe53-4456-870b-6b136d9b36b4
md"""
#### 6: Least squares limbo (5 points)

Consider the least squares problem of minimizing

$$\phi(x, y) = \|Ax + By - d\|^2$$

and suppose you are given a code `solveAls` such that `solveAls(d)` computes $A^\dagger d$.  In this problem, we will solve the least squares problem for $x$ and $y$ using this solver as a building block.

1. (1 point) Write the normal equations for the least squares problem as a block 2-by-2 system.
2. (1 point) Do block Gaussian elimination to write a Schur complement system that can be solved for $y$
3. (1 point) Rewrite the Schur complement system so that $A$ is not used directly -- only use the combinations $A^T B$ and $A^\dagger$ appear (and anything involving $B$ on its own is fine).
4. (2 points) Complete the following Julia code to solve the least squares problem.
"""

# ╔═╡ 5232b508-fdff-4653-b47b-5be746242478
function solveABls(solveAls, ATB, B, d)
	
	# Placeholder: Replace these lines!
	x = solveAls(d)
	y = B\d

	# Return the solution.
	x, y

end

# ╔═╡ 7a783286-627c-4b21-bee4-86a711ce6030
md"""
You may use the following test to sanity check your solution.
"""

# ╔═╡ c28a61ee-9e76-4d17-9a29-9689d0b6f77e
function test_solveABls()
	A = rand(10,4)
	B = rand(10,3)
	d = rand(10)
	x, y = solveABls((d) -> A\d, A'*B, B, d)
	xyref = [A B]\d
	norm(xyref - [x; y])/norm(xyref)
end

# ╔═╡ c86269d0-c08a-409a-82e2-04928501bd0f
test_solveABls()

# ╔═╡ 85d813d1-45d1-453f-ac2a-60fe59ce7ccb
md"""
#### 7: Your turn

1. (1 point) Share one thing in the class you think is working well.
2. (1 point) Share one thing you think could work better (concrete recommendations are great!).
3. (1 point) What do you consider the most difficult concept from the first part of the course?
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
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

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
"""

# ╔═╡ Cell order:
# ╟─db885ec4-98e2-11ec-30bb-eb078b76cb06
# ╠═1fb95f1d-a351-4ee0-a53b-1c836cf4882e
# ╟─bb1f0b7c-b648-4f2a-8b64-52a65f2f34fa
# ╠═3dba96b2-c65e-4299-a935-b665c2191c3f
# ╟─2023ed87-239f-4be4-902d-df34f1a49b7a
# ╠═4d9a417c-bfd2-4ce0-8ddc-ea8b4f98166c
# ╟─f1e11c5e-0c7d-452b-9194-c9752848571a
# ╟─5ff3caf7-76b2-4467-bc68-3311ca5f34da
# ╠═0fdf5a9d-3aed-4458-8590-f2494279689b
# ╠═7b15fcd6-0701-4ae2-9ef7-866392e4cc99
# ╠═2cef793b-586b-4e82-8784-1d48be9e7822
# ╟─cbb8b2c1-2256-43ca-8e48-f372604088c2
# ╠═b64b54f0-caf2-4139-934e-cbf7f21349d3
# ╟─cd34f600-18ea-4cff-992d-584fb91e6b09
# ╠═e5cef97e-822d-4305-9bcf-35c019e31ea5
# ╠═8901acef-4c17-4126-bd4a-28a5bd49aba6
# ╟─2ff3b061-fe53-4456-870b-6b136d9b36b4
# ╠═5232b508-fdff-4653-b47b-5be746242478
# ╟─7a783286-627c-4b21-bee4-86a711ce6030
# ╠═c28a61ee-9e76-4d17-9a29-9689d0b6f77e
# ╠═c86269d0-c08a-409a-82e2-04928501bd0f
# ╟─85d813d1-45d1-453f-ac2a-60fe59ce7ccb
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
