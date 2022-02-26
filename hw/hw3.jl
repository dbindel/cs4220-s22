### A Pluto.jl notebook ###
# v0.18.1

using Markdown
using InteractiveUtils

# ╔═╡ bdbcd25e-7d8e-48df-a2d0-41d80b1951e5
using LinearAlgebra

# ╔═╡ 431abcec-1771-4d65-bf77-18c4822f6c11
using QuadGK

# ╔═╡ 8174d616-8626-11ec-2006-c34b85974899
md"""
# HW 3 for CS 4220

You may (and probably should) talk about problems with the each other, with the TAs, and with me, providing attribution for any good ideas you might get.  Your final write-up should be your own.
"""

# ╔═╡ be652c25-25e4-4f8a-8775-8bfc4918bc3d
md"""
#### 1: Artificial inference

In the code block below, we create an artificial data set for a linear function on the unit square in $\mathbb{R}^2$:

$$f(x,y) = c_1 + c_2 x + c_2 y$$

Based on noisy measurements $b_i = f(x_i,y_i) + \epsilon_i$, we want to use least squares to find an estimator $\hat{c}$ for the underlying $c$.  Fill in the `TODO` line to do this least squares computation.
"""

# ╔═╡ 7ddd62e9-46e0-4f76-bab4-7d26bce2e912
begin

	# Set up test problem
	σ = 1e-2
	xy = rand(20,2)
	c = [0.5; 1.0; 2.0]
	b = c[1] .+ c[2]*xy[:,1] .+ c[3]*xy[:,2] + σ*randn(20)

	# TODO: Estimate c via least squares
	cest = [0.6; 1.1; 2.1]   # Placeholder, replace!

	# Compare
	norm(c-cest)

end

# ╔═╡ ac041527-1cbb-4b10-ab5b-536ff7781aba
md"""
#### 2: Retargeting Reflections

Given a nonzero vector $x \in \mathbb{R}^n$ and a target unit vector $u$, find a unit vector $v$ such that

$$(I-2vv^T) x = \sigma u$$
"""

# ╔═╡ 2f69a9f3-3b25-4361-9412-0629844cf00c
function householder(x, u)
	v = u    # TODO: Placeholder!
	σ = 1.0  # TODO: Placeholder!
	return v, σ
end

# ╔═╡ 349985d2-8430-4f7b-9079-9d0dabf43522
function test_householder()
	x = rand(10)
	u = rand(10)
	u /= norm(u)
	v, σ = householder(x, u)
	r = x - 2*v*(v'*x) - σ*u
	norm(r)/norm(x)
end

# ╔═╡ 3764ca8e-8b00-4ab2-8f50-8e00de64f0e9
test_householder()

# ╔═╡ f2c63e53-4af9-4036-964c-3477471abcc5
md"""
#### 3: Continuous least squares

For a general inner product, the normal equations for finding the closest point $p \in \mathcal{V}$ to a target $f$ is

$$\langle q, p-f \rangle = 0, \mbox{ for all } q \in \mathcal{V}.$$

Use this approach to find the cubic polynomial $p(x) = \alpha x + \beta x^3$
that minimizes

$$\|p-\sin(\cdot)\|_2^2 = \int_{-1}^1 (p(x)-\sin(x))^2 \, dx$$

You may use without further note the following integrals:

$$\begin{align*}
\int_{-1}^1 x \sin(x) \, dx &= 2 \sin(1) - 2 \cos(1) \\
\int_{-1}^1 x^3 \sin(x) \, dx &= 10 \cos(1) - 6 \sin(1)
\end{align*}$$
"""

# ╔═╡ 3728baea-0eb0-4cf9-a351-a3da94d890de
begin
	ATp = [2*sin(1)-2*cos(1); 10*cos(1)-6*sin(1)]
	αβ = [1.0, 1.0/6]  # TODO: Placeholder -- coefficients for a Taylor series
end

# ╔═╡ 578f5b07-9174-4ba3-bf0e-88ba7e481671
md"""
We also provide a sanity check -- this should be down around $10^{-16}$ if everything is done correctly.
"""

# ╔═╡ 3bc2ff7b-8562-45d6-9349-3d5f0e914752
begin
	# Sanity check: compute <x, r> and <x^3, r> using numerical quadrature
	r(x) = sin(x) - x*αβ[1] - x.^3*αβ[2]
	[quadgk((x) -> x*r(x), -1.0, 1.0)[1];
	 quadgk((x) -> x^3*r(x), -1.0, 1.0)[1]]
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
QuadGK = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"

[compat]
QuadGK = "~2.4.2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.2"
manifest_format = "2.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "44c37b4636bc54afac5c574d2d02b625349d6582"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.41.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3daef5523dd2e769dad2365274f760ff5f282c7d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.11"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╟─8174d616-8626-11ec-2006-c34b85974899
# ╠═bdbcd25e-7d8e-48df-a2d0-41d80b1951e5
# ╠═431abcec-1771-4d65-bf77-18c4822f6c11
# ╟─be652c25-25e4-4f8a-8775-8bfc4918bc3d
# ╠═7ddd62e9-46e0-4f76-bab4-7d26bce2e912
# ╟─ac041527-1cbb-4b10-ab5b-536ff7781aba
# ╠═2f69a9f3-3b25-4361-9412-0629844cf00c
# ╠═349985d2-8430-4f7b-9079-9d0dabf43522
# ╠═3764ca8e-8b00-4ab2-8f50-8e00de64f0e9
# ╟─f2c63e53-4af9-4036-964c-3477471abcc5
# ╠═3728baea-0eb0-4cf9-a351-a3da94d890de
# ╟─578f5b07-9174-4ba3-bf0e-88ba7e481671
# ╠═3bc2ff7b-8562-45d6-9349-3d5f0e914752
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
