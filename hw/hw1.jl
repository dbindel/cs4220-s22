### A Pluto.jl notebook ###
# v0.17.7

using Markdown
using InteractiveUtils

# ╔═╡ 67407319-ab93-402b-b281-67afecac152e
using LinearAlgebra

# ╔═╡ dfa9a07e-a7ea-4a38-97f7-9854ad6d6fe9
using BenchmarkTools

# ╔═╡ 487c4b1c-7fd9-11ec-11b8-d9640811f522
md"""
# HW 1 for CS 4220

You may (and probably should) talk about problems with the each other, with the TAs, and with me, providing attribution for any good ideas you might get.  Your final write-up should be your own.
"""

# ╔═╡ 5fd85ce3-d746-4682-b9ce-8980b6692a3c
md"""
#### 1: Placing Parens

Suppose $A, B \in \mathbb{R}^{n \times n}$ and $d, u, v, w \in \mathbb{R}^n$.  Rewrite each of the following Julia functions to compute the same result but with the indicated asymptotic complexity.
"""

# ╔═╡ fcd2b4ff-e6f8-4c94-8f2a-46b0dd239005
# Rewrite to run in O(n)
function hw1p1a(A, d)
	D = diagm(d)
	tr(D*A*D)
end

# ╔═╡ 2fff33f2-d129-41ae-a949-5946e534019a
begin
	function test_hw1p1a()
		A = [1.0 2.0; 3.0 4.0]
		d = [9.0; 11.0]
		hw1p1a(A, d) == 565.0
	end
	
	if test_hw1p1a()
		"P1a code passes correctness test"
	else
		"P1a code fails correctness test"
	end
end

# ╔═╡ 9db8098d-0336-43be-9635-4b1133199dec
begin
	function time_hw1p1a(n)
		A = rand(n, n)
		d = rand(n)
		hw1p1a(A, d)
		@belapsed hw1p1a($A, $d) seconds=0.5
	end
	"Estimated complexity: O(n^$(log2(time_hw1p1a(1000)/time_hw1p1a(500))))"
end

# ╔═╡ d7feb833-2e95-4272-b87d-21b2db67872f
# Rewrite to run in O(n)
function hw1p1b(d, u, v, w)
	A = diagm(d) + u*v'
	A*w
end

# ╔═╡ 3c26307e-71ca-40c5-aa11-4073bfd31fd4
begin
	function test_hw1p1b()
		d = [9.0; 11.0]
		u = [8.0; 12.0]
		v = [7.0; 11.0]
		w = [4.0; 5.0]
		hw1p1b(d, u, v, w) == [700.0; 1051.0]
	end
	
	if test_hw1p1b()
		"P1b code passes correctness test"
	else
		"P1b code fails correctness test"
	end
end

# ╔═╡ 64535fc4-1403-43ee-ba8b-36edae2ad94e
begin
	function time_hw1p1b(n)
		d = rand(n)
		u = rand(n)
		v = rand(n)
		w = rand(n)
		hw1p1b(d, u, v, w)
	    @belapsed hw1p1b($d, $u, $v, $w) seconds=0.5
	end

	"Estimated complexity: O(n^$(log2(time_hw1p1b(4000)/time_hw1p1b(2000))))"
end

# ╔═╡ dc1e6d3d-e205-429c-a47f-0d144fc25a09
# Rewrite to run in O(n^2)
function hw1p1c(A, B, d, w)
	(diagm(d) + A*B)*w
end

# ╔═╡ f3efa1d2-7c2e-4879-b8d1-9e424bc098bf
begin
	function test_hw1p1c()
		A = [1.0 2.0; 3.0 4.0]
		B = [5.0 6.0; 7.0 8.0]
		d = [8.0; 12.0]
		w = [4.0; 5.0]
		hw1p1c(A, B, d, w) == [218.0, 482.0]
	end
	if test_hw1p1c()
		"P1c code passes correctness test"
	else
		"P1c code fails correctness test"
	end
end

# ╔═╡ 3fa8d482-b6fe-4be6-94cd-5b1ac072bafa
begin
	function time_hw1p1c(n)
		A = rand(n, n)
		B = rand(n, n)
		d = rand(n)
		w = rand(n)
		hw1p1c(A, B, d, w)
	    @belapsed hw1p1c($A, $B, $d, $w) seconds=0.5
	end

	"Estimated complexity: O(n^$(log2(time_hw1p1c(4000)/time_hw1p1c(2000))))"
end

# ╔═╡ fb114d09-9e1e-4501-b5d8-051d34d97aa0
md"""
#### 2: Making matrices

In terms of the power basis, write

1. The matrix $A \in \mathbb{R}^{3 \times 3}$ corresponding to the inner product between two quadratics; that is, if $p(x) = c_0 + c_1 x + c_2 x^2$ and $q(x) = d_0 + d_1 x + d_2 x^2$, then $\int_{-1}^1 p(x) q(x) \, dx = d^T A c$.
1. The matrix $B \in \mathbb{R}^{2 \times 3}$ corresponding to the linear map from quadratics to linears associated with differentiation.

The following testers will sanity check your answer, but you should also (briefly) describe how you got the right matrix.
"""

# ╔═╡ a9f6c05e-6be6-4e65-abe8-59f3f1a14d3a
A_quadratic = [1.0   0.0   0.0; 
	           0.0   1.0   0.0; 
	           0.0   0.0   1.0]

# ╔═╡ 298f52ab-3e1c-47b0-b912-fa515794a26a
begin
	# This is a three-point Gauss quadrature rule to approximate
	# integral of f from -1 to 1.  It is exact for polynomials up to degree 5
	function gauss3(f)
		ξ = sqrt(3/5)
		(5*f(-ξ) + 8*f(0.0) + 5*f(ξ))/9
	end
	
	function test_quadratic_form(A)
		c = rand(3)
		d = rand(3)
		p(x) = (c[3]*x + c[2])*x + c[1]
		q(x) = (d[3]*x + d[2])*x + d[1]
		ξ = sqrt(3/5)
		ϕ_A = c'*A*d
		ϕ_int = gauss3((x) -> p(x)*q(x))
		(ϕ_int-ϕ_A)/ϕ_A
	end

	if abs(test_quadratic_form(A_quadratic)) < 1e-8
		"A passes basic check"
	else
		"A fails basic check"
	end
end

# ╔═╡ 39b5e027-4446-4ad1-a5d1-61f2afc62d1b
B_deriv = [0.0 0.0 0.0;
	       0.0 0.0 0.0]

# ╔═╡ 36e0ae4b-567b-43fa-a4df-db4787a3a5ad
begin
	function test_B_deriv(B)
		c = [7.0; 11.0; 3.0]  # p(x) = 3x^2 + 11x + 7
		d = [11.0; 6.0]      # p'(x) = 6x + 11
		B*c == d
	end

	if test_B_deriv(B_deriv)
		"B passes basic check"
	else
		"B fails basic check"
	end
end

# ╔═╡ 02872ae8-e7c4-45b6-a386-a97a5c3ef4dd
md"""
#### 3: Crafty cosines

Suppose $\| \cdot \|$ is a inner product norm in some real vector space and you are given

$$a = \|u\|, \quad b = \|v\|, \quad c = \|u-v\|$$

Express $\langle u, v \rangle$ in terms of $a$, $b$, and $c$.  Be sure to explain where your formula comes from.
"""

# ╔═╡ 0f70e29d-76f4-483b-8ce3-af031eb987ab
function compute_dot(a, b, c)
	return 0.0
end

# ╔═╡ 03d1dfa5-aa1c-4222-8174-abdee1bf1557
begin
	function test_dot_abc()
		u = rand(10)
		v = rand(10)
		a = norm(u)
		b = norm(v)
		c = norm(u-v)
		d1 = compute_dot(a, b, c)
		d2 = v'*u
		(d2-d1)/d2
	end
	
	if abs(test_dot_abc()) < 1e-8
		"Passes sanity check"
	else
		"Fails sanity check"
	end
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[compat]
BenchmarkTools = "~1.2.2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.1"
manifest_format = "2.0"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "940001114a0147b6e4d10624276d56d531dd9b49"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.2.2"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "0b5cfbb704034b5b4c1869e36634438a047df065"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.2.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

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

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
"""

# ╔═╡ Cell order:
# ╟─487c4b1c-7fd9-11ec-11b8-d9640811f522
# ╠═67407319-ab93-402b-b281-67afecac152e
# ╠═dfa9a07e-a7ea-4a38-97f7-9854ad6d6fe9
# ╟─5fd85ce3-d746-4682-b9ce-8980b6692a3c
# ╠═fcd2b4ff-e6f8-4c94-8f2a-46b0dd239005
# ╟─2fff33f2-d129-41ae-a949-5946e534019a
# ╟─9db8098d-0336-43be-9635-4b1133199dec
# ╠═d7feb833-2e95-4272-b87d-21b2db67872f
# ╟─3c26307e-71ca-40c5-aa11-4073bfd31fd4
# ╟─64535fc4-1403-43ee-ba8b-36edae2ad94e
# ╠═dc1e6d3d-e205-429c-a47f-0d144fc25a09
# ╟─f3efa1d2-7c2e-4879-b8d1-9e424bc098bf
# ╟─3fa8d482-b6fe-4be6-94cd-5b1ac072bafa
# ╟─fb114d09-9e1e-4501-b5d8-051d34d97aa0
# ╠═a9f6c05e-6be6-4e65-abe8-59f3f1a14d3a
# ╟─298f52ab-3e1c-47b0-b912-fa515794a26a
# ╠═39b5e027-4446-4ad1-a5d1-61f2afc62d1b
# ╟─36e0ae4b-567b-43fa-a4df-db4787a3a5ad
# ╟─02872ae8-e7c4-45b6-a386-a97a5c3ef4dd
# ╠═0f70e29d-76f4-483b-8ce3-af031eb987ab
# ╟─03d1dfa5-aa1c-4222-8174-abdee1bf1557
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
