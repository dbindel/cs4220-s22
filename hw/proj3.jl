### A Pluto.jl notebook ###
# v0.18.1

using Markdown
using InteractiveUtils

# ╔═╡ 3d85045e-b5e1-11ec-20a5-192793e5cee1
using LinearAlgebra

# ╔═╡ 371d1d5c-fc18-4850-8f0a-ca4dd8124fe7
using Plots

# ╔═╡ 858bb691-6680-471e-9375-135c19acca8e
md"""
# Project 3: Adaptive Splines
#### Due: 2022-04-22

A [*polyharmonic spline*](https://en.wikipedia.org/wiki/Thin_plate_spline) is a type of function approximator of the form

$$s(x) = \sum_{j=1}^m \phi(\|x-u_j\|) a_j + \gamma_{1:d}^T x + \gamma_{d+1}$$

where

$$\phi(\rho) = \begin{cases}
  r^k \log (r), & k \mbox{ even}, \\
  r^k, & k \mbox{ odd}.
\end{cases}$$

Common examples include cubic splines $(k = 3)$ and thin plate splines ($k = 2)$.  We will consider cubic splines here.

We will touch on two optimization questions in this project.  First, how do we use splines to help optimization?  Second, how do we best choose the coefficients $a$ and $\gamma$ and the centers $\{u_j\}_{j=1}^m$?
"""

# ╔═╡ 0e838745-b813-45fd-8f59-54a3bde304c9
md"""
## Code setup

In the interest of making all our lives easier, we have a fair amount of starter code for this project.  You are not responsible for all of it in detail, but I certainly recommend at least becoming familiar with the interfaces!
"""

# ╔═╡ b3ce38ef-94a1-484e-87ce-9bb5e6b9fa5c
md"""
### Basis functions and evaluation

As a point of comparison, we will want to look at the ordinary spline procedure (where there are as many centers as there are data points, and data is given at the centers).  To start with, we need $\phi$ and its derivatives.
"""

# ╔═╡ 262eecfc-a33e-4c25-b606-15e875d38270
begin
	ϕ(ρ :: Float64)  = ρ^3
	dϕ(ρ :: Float64) = 3*ρ^2
	ϕ(r)   = ϕ(norm(r))
	∇ϕ(r)  = 3*norm(r)*r
	
	function Hϕ(r)
		ρ = norm(r)
		if ρ == 0.0
			zeros(length(r), length(r))
		else
			u = r/ρ
			3*ρ*(I + u*u')
		end
	end
end

# ╔═╡ d0280b13-2792-4c3a-bc5f-93f98558d572
md"""
When evaluating the spline will be convenient to refer to $a$ and $\gamma$ together in one vector, which we will call $c$.
"""

# ╔═╡ c0a2d148-bfc6-4d04-a6c4-bc97e3adce5a
function spline_eval(x :: Vector, u, c)
	d, m = size(u)
	sum(c[j]*ϕ(x-u[:,j]) for j=1:m) + c[m+1:m+d]'*x + c[end]
end

# ╔═╡ bd9dc94d-242b-431d-af77-d145fec1cf32
function spline_eval(x :: Matrix, u, c)
	d, n = size(x)
	d, m = size(u)
	[sum(c[j]*ϕ(xi-u[:,j]) for j=1:m) + c[m+1:m+d]'*xi + c[end]
		for xi in eachcol(x)]
end

# ╔═╡ 81f0fb85-856b-44bf-8413-40feb28f41c3
md"""
We are frequently going to care about the residual on our test function (over many evaluation points), so we also provide small helpers for that computation and for plotting the spline and the residual.
"""

# ╔═╡ 0e80f389-032f-4902-b9b0-1df1380f13e1
md"""
### Fitting the spline

We consider two methods for fitting the spline.  For both, we need to compute two matrices:

- The kernel matrix $K_{XU}$ where $(K_{XU})_{ij} = \phi(\|x_i-u_j\|)$
- The polynomial part $\Pi_{X}$ consisting of rows of the form $\begin{bmatrix} x_i^T & 1 \end{bmatrix}$
"""

# ╔═╡ c5e41a00-bdb2-4e3e-aedc-d64df656035e
begin
	spline_K(x, u) = [ϕ(xi-uj) for xi in eachcol(x), uj in eachcol(u)]
	spline_Π(x) = [x' ones(size(x)[2])]
end

# ╔═╡ 3e00fb77-8606-460c-9185-eec318863bbe
md"""
Later, we will also have need for the Jacobian of $K_{XU}$ with respect to the components of $u$.
"""

# ╔═╡ eeb9a2f8-24c8-434b-9df0-03ca654fcc76
function Dspline_K(x, u)
	d, n = size(x)
	d, m = size(u)
	Jac = zeros(n, d*m)
	for i = 1:n
		for j = 1:m
			J = (j-1)*d+1:j*d
			Jac[i,J] = ∇ϕ(u[:,j]-x[:,i])
		end
	end
	Jac
end

# ╔═╡ e6e563e5-badb-4dbc-b615-3559179cf142
md"""
#### Standard spline fitting

In the standard scheme, we have one center at each data point.  Fitting the spline involves solving the system

$$\begin{bmatrix} K+\sigma^2 I & \Pi \\ \Pi^T & 0 \end{bmatrix}
  \begin{bmatrix} a \\ \gamma \end{bmatrix} = 
  \begin{bmatrix} y \\ 0 \end{bmatrix}.$$

The first $n$ equations are interpolation conditions.  The last $d+1$ equations are sometimes called the "discrete orthogonality" conditions.  When the regularization parameter $\sigma^2$ is nonzero, we have a *smoothing spline*.
"""

# ╔═╡ 9f22c1a7-cd17-4eb7-b044-5e82dd799567
function spline_fit(x, y :: Vector; σ=0.0)
	d, n = size(x)
	K = spline_K(x, x)
	Π = spline_Π(x)
	[K+σ^2*I Π; Π' zeros(d+1,d+1)]\[y; zeros(d+1)]
end

# ╔═╡ 9c19d90b-cb29-4db7-8b2e-16928e0bc275
spline_fit(x, f; σ=0.0) = spline_fit(x, [f(xi) for xi in eachcol(x)], σ=σ)

# ╔═╡ b74de333-35da-4cbf-a760-f2f0540ebcad
md"""
#### Least squares fitting

When the number of data points is large, we may want to use a smaller number of centers as the basis for our approximation.  That is, we consider an approximator of the form

$$s(x) = \sum_{j=1}^n \phi(\|x-u_j\|) a_j + \gamma_{1:d}^T x + \gamma_{d+1}$$

where the locations $u_j$ are decoupled from the $x_i$.  We determine the coefficients $a$ and $\gamma$ by a least squares problem

$$\mbox{minimize } \left\|\begin{bmatrix} K_{XU} &\Pi \end{bmatrix} \begin{bmatrix} a \\ \gamma \end{bmatrix} - y\right\|^2 + \sigma^2 \|a\|^2$$
"""

# ╔═╡ 40041f04-4854-43e9-aa71-2a31666f6bf4
function spline_fit(u, x, y :: Vector; σ=0.0)
	d, m = size(u)
	if σ == 0.0
		[spline_K(x, u) spline_Π(x)]\y
	else
		[spline_K(x, u) spline_Π(x); σ*I zeros(m, d+1)]\[y; zeros(m)]
	end
end

# ╔═╡ a07bc826-f2d3-422a-8fa3-2516268c04fc
spline_fit(u, x, f; σ=0.0) = spline_fit(u, x, [f(xi) for xi in eachcol(x)], σ=σ)

# ╔═╡ 4180dc81-4b4f-49c7-a6bf-a9f98bbe66d9
md"""
### Sampling strategies

Splines are fit to sampled data, so we want a number of ways to draw samples.  Our example is 2D, so we will stick to 2D samplers.

We start with samplers on a regular mesh of points $(x_i, y_j)$ (sometimes called a tensor product mesh).  These get big very fast in high-dimensional spaces, but are fine in 2D.
"""

# ╔═╡ 10823e1b-57ee-43a3-8676-9aa20424c97c
function meshgrid(xx, yy)
	nx = length(xx)
	ny = length(yy)
	result = zeros(2, nx, ny)
	result[1,:,:] = xx*ones(ny)'
	result[2,:,:] = ones(nx)*yy'
	reshape(result, 2, nx*ny)
end

# ╔═╡ d0c9f796-3a53-4f9f-bcef-689e2f5eb135
function meshgrid_uniform(xlo, xhi, ylo, yhi, nx, ny)
	meshgrid(range(xlo, xhi, length=nx),
			 range(ylo, yhi, length=ny))
end

# ╔═╡ a6d5a991-54eb-400c-9201-0a267953de24
function meshgrid_cheb(xlo, xhi, ylo, yhi, nx, ny)
	chebnodes(l,h,n) = (h+l)/2 .+ (h-l)/2*cos.((2*(1:n).-1)/(2n)*π)
	meshgrid(chebnodes(xlo, xhi, nx),
			 chebnodes(ylo, yhi, nx))
end

# ╔═╡ 94fbfb98-fa49-45bc-b714-2ee275a38dc9
md"""
In higher-dimensional spaces, it is tempting to choose random samples.  But taking independent uniform draws is not an especially effective way of covering a space -- random numbers tend to clump up.  For this reason, [*low discrepancy sequences*](https://en.wikipedia.org/wiki/Low-discrepancy_sequence) are often a better basis for sampling than (pseudo)random draws.  There are many such generators; we use a relatively simple one based on an additive recurrence with a multiplier based on the ["generalized golden ratio"](http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/).
"""

# ╔═╡ c30bc9e7-06d8-44d9-9879-3485e90ba44a
function kronecker_quasirand(d, N, start=0)
    
    # Compute the recommended constants ("generalized golden ratio")
    ϕ = 1.0+1.0/d
    for k = 1:10
        gϕ = ϕ^(d+1)-ϕ-1
        dgϕ= (d+1)*ϕ^d-1
        ϕ -= gϕ/dgϕ
    end
    αs = [mod(1.0/ϕ^j, 1.0) for j=1:d]
    
    # Compute the quasi-random sequence
    Z = zeros(d, N)
    for j = 1:N
        for i=1:d
            Z[i,j] = mod(0.5 + (start+j)*αs[i], 1.0)
        end
    end
    
    Z
end

# ╔═╡ 31fdb102-84aa-4598-b922-a1f0b62aa6d0
md"""
We will provide both random and quasi-random samplers.
"""

# ╔═╡ eafec17b-cbef-4c70-898f-f9730cf61250
begin
	rand2d(xlo, xhi, ylo, yhi, n) = xlo .+ (xhi-xlo)*rand(2,n)
	qrand2d(xlo, xhi, ylo, yhi, n) = xlo .+ (xhi-xlo)*kronecker_quasirand(2,n)
end

# ╔═╡ c8455deb-885b-4db2-9029-86300fb25216
md"""
### Optimizers

Most of the time if you are using standard optimizers like Levenberg-Marquardt or Newton, you should use someone else's code and save your ingenuity for problem formulation and initial guesses.  In this spirit, we provide two solvers that we will see in class in a couple weeks: a Levenberg-Marquardt implementation and a Newton implementation.

We will see the Levenberg-Marquardt algorithm for nonlinear least squares as part of our regular lecture material, but we also include it here.  This version uses logic for dynamically scaling the damping parameter (courtesy Nocedal and Wright).
"""

# ╔═╡ 6d343cad-8b34-4e90-aab9-d182644ce22a
function levenberg_marquardt(f, J, x; nsteps=100, rtol=1e-8, τ=1e-3, 
						     monitor=(x, rnorm, μ)->nothing)
    
    # Evaluate everything at the initial point
    x = copy(x)
	xnew = copy(x)
	fx = f(x)
    Jx = J(x)
    Hx = Jx'*Jx

    μ = τ * maximum(diag(Hx))  # Default damping parameter
    ν = 2.0                    # Step re-scaling parameter (default value)
    
    for k = 1:nsteps
        
        # Check for convergence
        g = Jx'*fx
        rnorm = norm(Jx'*fx)
        monitor(x, rnorm, μ)
        if rnorm < rtol
            return x
        end
        
        # Compute a proposed step and re-evaluate residual vector
        p = (Hx + μ*I)\(-g)
        xnew[:] = x[:] + p
        fxnew = f(xnew)
        
        # Compute the gain ratio
        ρ = (norm(fx)^2 - norm(fxnew)^2) / (norm(fx)^2 - norm(fx+Jx*p)^2)
        
        if ρ > 0  # Success!
            
            # Accept new point
            x[:] = xnew
            fx = fxnew
            Jx = J(x)
            Hx = Jx'*Jx
            
            # Reset re-scaling parameter, update damping
            μ *= max(1.0/3.0, 1.0-2.0*(ρ-1.0)^3)
            ν = 2.0
        
        else
                
            # Rescale damping5tb
            μ *= ν
            ν *= 2.0

        end
    end
    #warning("Did not converge in $nsteps iterations")
	x
end

# ╔═╡ 479aecb4-15ce-4ac8-92d2-0369de55bf6f
md"""
The trust-region Newton solver optimizes a quadratic model over some region in which it is deemed trustworthy.  The model does not have to be convex!
"""

# ╔═╡ 5c359ca8-8780-4989-8e8a-74a8201712e4
function solve_tr(g, H, Δ)
    n = length(g)

    # Check interior case
    try
        F = cholesky(H)
        p = -(F\g)
        if norm(p) <= Δ
            return p, false
        end
    catch e
        # Hit this case if Cholesky errors (not pos def)
    end    

    # Compute the relevant eigensolve
    w = g/Δ
    M = [H    -I ;
         -w*w' H ]
    λs, V = eigen(M)
    
    # The right most eigenvalue (always sorted to the end in Julia) is real,
    # and corresponds to the desired λ
    λ = -real(λs[1])
    v = real(V[:,1])
    y2 = v[1:n]
    y1 = v[n+1:end]
    
    # Check if we are in the hard case (to some tolerance)
    gap = real(λs[2])-real(λs[1])
    if norm(y1) <= 1e-8/sqrt(gap)
        # Hard case -- we punt a little and assume only one null vector
        #  Compute min-norm solution plus a multiple of the null vector.
        v = y2/norm(y2)
        q = -(H+norm(H)/n^2*v*v')\g
        return q + v*sqrt(Δ^2-q'*q), true
    else
        # Standard case -- extract solution from eigenvector
        return -sign(g'*y2) * Δ * y1/norm(y1), true
    end
end

# ╔═╡ 0dbe5118-81b7-4299-ba3b-6bb009630d6b
function tr_newton(x0, ϕ, ∇ϕ, Hϕ; nsteps=100, rtol=1e-6, Δmax=Inf, monitor=(x, rnorm, Δ)->nothing)
    
    # Compute an intial step and try trusting it
    x = copy(x0)
	xnew = copy(x0)
    ϕx = ϕ(x)
    gx = ∇ϕ(x)
    Hx = Hϕ(x)
    p = -Hx\gx
    Δ = 1.2 * norm(p)^2
    hit_constraint = false
    
    for k = 1:nsteps

        # Compute gain ratio for new point and decide to accept or reject
        xnew[:] = x[:] + p
        ϕnew = ϕ(xnew)
        μdiff = -( gx'*p + (p'*Hx*p)/2 )
        ρ = (ϕx - ϕnew)/μdiff
        
        # Adjust radius
        if ρ < 0.25
            Δ /= 4.0
        elseif ρ > 0.75 && hit_constraint
            Δ = min(2*Δ, Δmax)
        end
        
        # Accept if enough gain (and check convergence)
        if ρ > 0.1
            x[:] = xnew
            ϕx = ϕnew
            gx = ∇ϕ(x)
            monitor(x, norm(gx), Δ)
            if norm(gx) < rtol
                return x
            end
            Hx = Hϕ(x)
        end

        # Otherwise, solve the trust region subproblem for new step
        p, hit_constraint = solve_tr(gx, Hx, Δ)

    end
    return x
end

# ╔═╡ 07610b62-6daa-4f2a-8f25-d8c773a7f7f3
md"""
### Incremental QR and forward selection

The [*forward selection* algorithm](https://en.wikipedia.org/wiki/Stepwise_regression) is a factor selection process for least squares problems in which factors are added to a model in a greedy fashion in order to minimize the residual at each step.

For fast implementation, it will be convenient for us to be able to extend a QR factorization by adding additional columns to an existing factorization.  This is easy enough to do in practice, but requires a little poking around in the [guts of the Julia QR code](https://github.com/JuliaLang/julia/blob/bf534986350a991e4a1b29126de0342ffd76205e/stdlib/LinearAlgebra/src/qr.jl#L4-L36).
"""

# ╔═╡ 9660a895-0a2e-4756-9750-2b2c0530046e
# Start a QR factorization overwriting the first n columns of A
windowed_qr(A, n) = LinearAlgebra.qrfactUnblocked!(view(A,:,1:n))

# ╔═╡ 68607d72-5cbc-4492-a577-a9cefa2cdc98
# Extend a "windowed" QR factorization up to the first n columns of A
function windowed_qr(F :: QR, A, n)
	m, k = size(F)
	lmul!(F.Q', view(A,:,k+1:n))
	Fnew = LinearAlgebra.qrfactUnblocked!(view(A,k+1:m,k+1:n))
	QR(view(A,:,1:n), [F.τ; Fnew.τ])
end

# ╔═╡ 44fe9368-7a43-457b-8b1a-9b42f63c30a8
md"""
We also will want to compute residuals for many of our algorithms.
"""

# ╔═╡ a944fc59-b27d-4cab-a92a-5e7eb93b17f1
# Overwrite the input with a least squares residual
function resid_ls!(F, y)
	m, n = size(F)
	lmul!(F.Q', y)
	y[1:n] .= 0.0
	lmul!(F.Q, y)
	y
end

# ╔═╡ 0cf1d599-9daa-4208-b969-758281cc9119
function resid_ls!(F, y, r)
	r[:] = y
	resid_ls!(F, r)
end

# ╔═╡ 6d0d2e80-0065-41b7-a18b-640b115b0245
# Solve a least squares problem and overwrite the rhs with the residual
function solve_ls!(F, y)
	m, n = size(F)
	lmul!(F.Q', y)
	c = F.R\y[1:n]
	y[1:n] .= 0.0
	lmul!(F.Q, y)
	c, y
end

# ╔═╡ 050c269d-ae5a-440f-ae34-107e5403cf47
# Starting with kstart factors, greedily choose another kend-kstart factors
# to minimize the residual A[:,I]*x-b.  Returns
#   - The selected index set I
#   - A QR factorization of A[:,I]
#   - The solution vector x
#   - The residual vector r = b-A[:,I]*x
#
function forward_selection(A, b, kstart, kend)
	m, n = size(A)

	# Set up storage
	AQR = zeros(m, kend)
	τ = zeros(m)
	z = zeros(kend)
	idx = zeros(Int, kend)

	# Compute column norms
	Anorm = [norm(aj) for aj = eachcol(A)]

	# Select first k columns (always included)
	AQR[:,1:kstart] = A[:,1:kstart]
	idx[1:kstart] = 1:kstart

	# Start factorization and compute initial residual
	F = windowed_qr(AQR, kstart)
	r = resid_ls!(F, copy(b))

	for k = kstart+1:kend

		# Check vs all candidates, pick the one r projects on most
		rproj = (A'*r) ./ Anorm
		jmax = argmax(abs.(rproj))

		# Fill in column k with selected candidate
		AQR[:,k] = A[:,jmax]
		idx[k] = jmax

		# Extend the factorization and update the residual
		F = windowed_qr(F, AQR, k)
		resid_ls!(F, b, r)

	end

	idx, F, F\b, r
end

# ╔═╡ 145d950e-c721-4d93-bd3b-8bb8f6ca5e62
let
	A = rand(100,20)
	b = rand(100)
	I, F, x, r = forward_selection(A, b, 5, 10)

	r0 = b-A[:,I]*x
	g = A[:,I]'*r
md"""
Some sanity checks on a random test case (keep first 5, choose 5 more):

- $I selected
- Does A[:,I] = QR? Relerr $( norm(A[:,I]-F.Q*F.R)/norm(F.R) )
- Does r = b-A[:,I]*x?  Relerr $( norm(r-r0)/norm(r) )
- Does A[:,I]'*r = 0?  Relerr $( norm(g)/norm(A[:,I])/norm(b) )
"""
end

# ╔═╡ 09fee4d4-f292-451c-a11c-ba9952ee2a51
md"""
## Test function

The [Himmelblau function](https://en.wikipedia.org/wiki/Himmelblau%27s_function) is a standard multi-modal test function used to explore the properties of optimization methods.  We will take the log of the Himmelblau function (shifted up a little in order to avoid log-of-zero issues) as our running example.
"""

# ╔═╡ 74e18286-34b1-408a-a638-938b876390b2
begin

	himmelblau(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
	himmelblau(xy) = himmelblau(xy[1], xy[2])
	
	log_himmelblau(x,y) = log(10 + himmelblau(x,y))
	log_himmelblau(xy) = log(10 + himmelblau(xy))

	xxh = range(-6.0, 6.0, length=100)
	p_logh = plot(xxh, xxh, log_himmelblau, st=:contourf)
end

# ╔═╡ 98bff7fa-5b34-4e7c-874a-4311bd28277e
function plot_spline(u, c)
	plot(xxh, xxh, (x,y) -> spline_eval([x;y], u, c), st=:contourf)
	plot!(u[1,:], u[2,:], st=:scatter, legend=false)
end

# ╔═╡ f5764f3c-c676-4c6e-af10-d622ed8a89ed
md"""
The Himmelblau function has a local maximum at $(-0.270845, -0.923039)$ and four global minimizers:

$$\begin{align*}
g(3.0,2.0)&=0.0, \\
g(-2.805118,3.131312)&=0.0, \\
g(-3.779310,-3.283186)&=0.0, \\ 
g(3.584428,-1.848126)&=0.0.
\end{align*}$$
"""

# ╔═╡ 530de511-8741-4829-9211-b449f6bcfb21
md"""
##### Tasks

1. Express the Himmelblau function as a nonlinear least squares function and use the provided Levenberg-Marquardt solver to find the minimum at $(3, 2)$ from a starting guess at $(2, 2)$.  Use the monitor function to produce a semilog convergence plot.

2. Run ten steps of an (unguarded) Newton solver for the Himmelblau function.  It may save you some algebra to remember the relation between the Hessian of a nonlinear least squares function and the Gauss-Newton approximation.  What does it converge to from the starting guess $(2, 2)$?  Plot convergence (gradient norm vs iteration) again.

3. Run the same experiment with the Newton solver, but this time using the trust region version that is provided.
"""

# ╔═╡ d6f3de3b-c904-4228-a7fa-c60319b1368b
md"""
## Optimizing on a surrogate

Let's now fit a surrogate to the Himmelblau function sampled on 50 samples drawn from our quasi-random number generator.
"""

# ╔═╡ f0b9a7bb-f2dc-4ed1-9b25-a2633d3924b7
begin
	uqrand = qrand2d(-6.0, 6.0, -6.0, 6.0, 50)
	c_qr1 = spline_fit(uqrand, log_himmelblau)
	plot_spline(uqrand, c_qr1)
end

# ╔═╡ 24e50c10-3535-4d78-802a-881c3e310870
md"""
Does the spline have minima close to the minima of the true function?  The answer depends how "nice" the minimizers are and how good the approximation error is. Concretely, suppose $|s(x)-g(x)| < \delta$ over $x \in \Omega$ and $x_*$ is a minimizer of $g$.  If $\min_{\|u\|=1} g(x_*+\rho u) - g(x_*) > 2\delta$ and the ball of radius $\rho$ around $x_*$ lies entirely inside $\Omega$, then $\min_{\|u\|=1} s(x_*+\rho u)-s(x_*) > 0$.  Assuming continuity, this is enough to guarantee that $s$ has a minimizer in the unit ball of radius $\rho$ around $x_*$.  If $g$ has a Lipschitz continuous Hessian with Lipschitz constant $M$ and $\rho < \lambda_{\min}(H_g(x_*))/M$, then a sufficient condition is that $\frac{1}{2} \rho^2 (\lambda_{\min}(H_g(x_*))-M\rho) > 2 \delta$ (still assuming the ball of radius $\rho$ lies within the domain $\Omega$).

(We can get better bounds if we can control the error in the derivative approximation, but we will leave this alone in the current assignment.)
"""

# ╔═╡ 72ea762e-6500-434b-ba77-5792107e9be2
md"""
Putting aside theoretical considerations for a moment, let's try to find the minimizer (or at least a local minimizer) of the spline approximation $s(x)$.  We can make a first pass at this by sampling on a mesh and taking the point where we see the smallest function value.
"""

# ╔═╡ a3faebed-c219-4dcb-896a-0b12e6348e71
let
	xy = meshgrid_uniform(-6.0, 6.0, -6.0, 6.0, 20, 20)
	gxy = spline_eval(xy, uqrand, c_qr1)
	xy_best = xy[:,gxy .<= minimum(gxy)]
	plot_spline(uqrand, c_qr1)
	p = scatter!([xy_best[1,:]], [xy_best[2,:]], marker=:star5, markersize=10)
md"""
$p

Best point found at $(xy_best[1]), $(xy_best[2])
"""
end

# ╔═╡ 3976c748-b2f1-46ed-b757-ba479c5f8481
md"""
##### Tasks

1.  Let $x_*$ be a strong local minimizer of $g$, where $g$ is twice continuously differentiable and the Hessian $H_g$ has a Lipschitz constant $M$ in the operator 2-norm, and that $\rho < \lambda_{\min}(H_g(x_*))/M$.  Using a Taylor expansion about $x_*$, show that $\frac{1}{2} \rho^2(\lambda_{\min}(H_g(x_*))-Mρ) > 2 \delta$ is sufficient to guarantee that $\min_{\|u\|=1} g(x_* + \rho u) - g(x_*) > 2\delta$.

2.  Compute the smallest singular value of the Hessian of the log-transformed Himmelblau function at $(3,2)$.  Also compute the approximation error in the spline fit at $(3,2)$ and use that as an approximate $\delta$ and plug into the estimate

$$\tilde{\rho} = 2\sqrt{\frac{\delta}{\lambda_{\min}(H_g(x_*))}}$$

3. Take ten steps of Newton to find minima the minimum of the spline close to the point found above.  Plot the gradient norm on a semilog scale to verify quadratic convergence.  What happens with a starting guess of $(3,2)$?

For the first task, you may use without proof that $\lambda_{\min}(A+E) \geq \lambda_{\min}(A)-\|E\|_2$ when $A$ and $E$ are symmetric.  For the second question, remember that the log transform we used is $g \mapsto \log(g + 10)$.  You should also not expect to get a tiny value for $\tilde{\rho}$!
"""

# ╔═╡ 14bc2bcb-5195-4820-9212-9fa0956c4b2b
md"""
## Non-adaptive sampling strategies

If we want to use a spline as a surrogate for optimization, we need it to be sufficiently accurate, where what is "sufficient" is determined by the behavior of the Hessian near the desired minima.  For the rest of this assignment, we focus on making more accurate splines.

We start by defining a highly refined "ground truth" mesh of $10^4$ points laid out in a mesh.  Our figure of merit will be the root mean squared approximation error.  We also define a coarser "data mesh" that we will use for least squares spline fits.
"""

# ╔═╡ 465b74a5-6539-4c62-aed8-24ec7cc9b7e4
begin
	xx_truth = meshgrid_uniform(-6.0, 6.0, -6.0, 6.0, 100, 100)
	y_truth = [log_himmelblau(x) for x in eachcol(xx_truth)]

	rms_vs_truth(u, c) = norm(y_truth - spline_eval(xx_truth, u, c)) / sqrt(length(y_truth))

	xx_data = meshgrid_uniform(-6.0, 6.0, -6.0, 6.0, 100, 100)
	y_data = [log_himmelblau(x) for x in eachcol(xx_data)]
end

# ╔═╡ 4ab01573-7a8c-4dd3-a138-8c43f867e562
md"""
We would like to choose a sample set that is going to minimize the RMS error.  We will talk about adapting the sample later; for now, let's consider different ways that we could lay out a set of about 50 sample points:

- We could choose a 7-by-7 mesh (uniform or Chebyshev)
- We could choose 50 points at random
- We could choose 50 points from a low-discrepancy sequence (quasi-random)

Let's set up an experimental framework to compare the RMS error on the ground truth mesh for each of these approaches.
"""

# ╔═╡ e3715c45-600f-4c65-b305-8968be2c31e5
function test_sample(name, u)
	c = spline_fit(u, log_himmelblau)
	p = plot_spline(u, c)
	r = rms_vs_truth(u, c)

md"""
###### $name (standard)

$p

RMS: $r
"""
end

# ╔═╡ cbc12bcc-ffbc-4975-bb46-8e944b0dbfd4
function test_sample(name, u, xdata, ydata)
	c = spline_fit(u, xdata, ydata)
	p = plot_spline(u, c)
	r = rms_vs_truth(u, c)

md"""
###### $name (least squares)

$p

RMS: $r
"""
end

# ╔═╡ aca5795d-11a4-4c96-a73a-634003558338
md"""
##### Tasks

Run experiments with the `test_sample` functions for both the standard spline fitting and least squares fitting against the 4000-point data set.  Comment on anything you observe from your experiments.  Feel free to play with the parameters (e.g. number of points).
"""

# ╔═╡ ce77c338-8eca-48bb-be4f-ba14a46c7f69
md"""
## Forward stepwise regression

Forward stepwise regression is a factor selection method in which we greedily choose the next factor from some set of candidates based on which candidate will most improve the solution.  We have provided you with a reference implementation of the algorithm.  Your task is to use this method to choose centers from a candidate set of points in order to improve your least squares fit.
"""

# ╔═╡ 598435d7-e56c-45d9-ade4-4844b640b530
md"""
##### Tasks

Complete the `spline_forward_regression` function below, then run the code with 500 points chosen from our quasirandom sequence.  Run the test harness above to see how much the RMS error measure improves.
"""

# ╔═╡ 2662eba2-7cb3-4085-87bc-812ed2625046
md"""
## Levenberg-Marquardt refinement

In the lecture notes from 4/11, we discuss the idea of variable projection algorithms for least squares.  We can phrase the problem of optimizing the centers in this way; that is, we let

$$r = (I-AA^\dagger)\bar{y}$$

where 

$$A = \begin{bmatrix} K_{XU} & \Pi_X \\ \sigma I & 0 \end{bmatrix}, \quad
  \bar{y} = \begin{bmatrix} y \\ 0 \end{bmatrix}$$

is treated as a function of the center locations.  The regularization parameter $\sigma$ plays a critical role in this particular problem, so we are careful to keep it throughout!

To save us all some trouble, I will provide the code to compute $r$ and the Jacobian with respect to the center coordinates.
"""

# ╔═╡ 29543f67-8ed6-4012-81c4-5bd78659e8a9
function spline_rproj(u, xx, y; σ=1e-3)
	d, m = size(u)
	d, n = size(xx)
	resid_ls!(qr([spline_K(xx, u) spline_Π(xx); σ*I zeros(m,d+1)]), [y; zeros(m)])
end

# ╔═╡ 9500819d-cae4-48ba-8a0d-bb27808c17cb
function spline_Jproj(u, xx, y; σ=1e-3)
	d, m = size(u)
	d, n = size(xx)

	# Compute c and r
	A = [spline_K(xx, u) spline_Π(xx); σ*I zeros(m,d+1)]
	F = qr(A)
	c, r = solve_ls!(F, [y; zeros(m)])

	# Compute W = -Q'*Jf_proj as in the 4-11 notes
	JA = [Dspline_K(xx, u); zeros(m,m*d)]
	W = F.Q'*JA
	z = JA'*r
	invRT = inv(F.R')

	# Process in groups of d columns (associated with each column of u)
	for j = 1:m
		J = (j-1)*d+1:j*d
		W[:,J] .*= c[j]
		W[1:m+d+1,J] = invRT[:,j] * z[J]'
	end

	# Return f_proj and the Jacobian of f_proj
	-(F.Q * W)
end

# ╔═╡ 70f68c93-1e57-4bd7-ae5c-000fa7ff2b8a
# Finite difference check for variable projection Jacobian computation
let
	h = 1e-5
	δu = randn(size(uqrand))
	rp = spline_rproj(uqrand+h*δu, xx_data, y_data)
	rm = spline_rproj(uqrand-h*δu, xx_data, y_data)
	Jδu_fd = (rp-rm)/(2h)
	Jδu = spline_Jproj(uqrand, xx_data, y_data)*δu[:]

md"""
As usual, we need a finite difference check to prevent programming errors.

- Finite difference relerr in Jacobian computation: $(norm(Jδu_fd-Jδu)/norm(Jδu))
"""
end

# ╔═╡ 8cb07531-4d95-4c80-8b58-171487df1951
md"""
##### Tasks

Use the provided Levenberg-Marquardt solver to refine the center locations picked by the forward selection algorithm.  This is a tricky optimization, and you may need to fiddle with solver parameters to get something you are happy about.  As usual, you should also provide a convergence plot.
"""

# ╔═╡ 7037f111-bd5c-40a6-bc65-6637cc67c0f8
md"""
## Newton refinement

Levenberg-Marquardt is popular for nonlinear least squares problems because it only requires first derivatives and often still gives very robust convergence.  But the convergence behavior for this problem is disappointing, and we have all the derivatives we need if we only have the fortitude to use them.

As in the Levenberg-Marquardt case, I will not require you to write the function evaluation and derivatives.
"""

# ╔═╡ 4c9373f8-f8eb-4ae4-a1af-826839db8389
function spline_ρproj(u, xx, y; σ=1e-3)
	d, m = size(u)
	d, n = size(xx)

	# Compute c and r (and ρ)
	A = [spline_K(xx, u) spline_Π(xx); σ*I zeros(m,d+1)]
	F = qr(A)
	c, r = solve_ls!(F, [y; zeros(m)])
	ρ = (r'*r)/2

	# Compute column derivatives
	JA = [Dspline_K(xx, u); zeros(m,m*d)]

	# Compute γ_i = A_,i c and s_i = A_,i^T r - A^T γ_i
	Γ = copy(JA)
	S = zeros(m+d+1, m*d)
	for j = 1:m
		J = (j-1)*d+1:j*d
		Γ[:,J] .*= c[j]
		S[j,J] = JA[:,J]'*r
	end
	S -= A'*Γ

	# Set up Hessian
	W = F.R'\S
	Hρ = Γ'*Γ - W'*W
	for i=1:n
		for j=1:m
			J = (j-1)*d+1:j*d
			Hρ[J,J] -= r[i]*c[j]*Hϕ(u[:,j]-xx[:,i])
		end
	end
 
	ρ, -Γ'*r, Hρ
end

# ╔═╡ 68a077f0-1280-4f52-87ec-9514df53fb5d
# Finite difference checks for full Newton setup
let
	h = 1e-5
	δu = randn(size(uqrand))
	ρp, ∇ρp, Hρp = spline_ρproj(uqrand+h*δu, xx_data, y_data)
	ρm, ∇ρm, Hρm = spline_ρproj(uqrand-h*δu, xx_data, y_data)
	ρ,  ∇ρ,  Hρ  = spline_ρproj(uqrand, xx_data, y_data)
	∇ρ_δu = ∇ρ'*δu[:]
	∇ρ_δu_fd = (ρp-ρm)/(2h)
	Hρ_δu = Hρ*δu[:]
	Hρ_δu_fd = (∇ρp-∇ρm)/(2h)
md"""
Finite difference checks:

- For the gradient: $(norm(∇ρ_δu-∇ρ_δu_fd)/norm(∇ρ_δu))
- For the Hessian: $(norm(Hρ_δu-Hρ_δu_fd)/norm(Hρ_δu))
"""
end

# ╔═╡ 37009b68-6b70-4204-ac33-19818d8fc348
md"""
##### Tasks

Use the provided Newton solver to refine the center locations picked by one of the earlier algorithms (I recommend using the Levenberg-Marquardt output as a starting point).  Give a convergence plot -- do you see quadratic convergence?
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"

[compat]
Plots = "~1.27.5"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.2"
manifest_format = "2.0"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9950387274246d08af38f6eef8cb5480862a435f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.14.0"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "bf98fa45a0a4cee295de98d4c1462be26345b9a1"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.2"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "12fc73e5e0af68ad3137b886e3f7c1eacfca2640"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.17.1"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "96b0bc6c52df76506efc8a441c6cf1adcb1babc4"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.42.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[deps.DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3daef5523dd2e769dad2365274f760ff5f282c7d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.11"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "51d2dfe8e590fbd74e7a842cf6d13d8a2f45dc01"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.6+0"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "af237c08bda486b74318c8070adb96efa6952530"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.64.2"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "cd6efcf9dc746b06709df14e462f0a3fe0786b1e"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.64.2+0"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "83ea630384a13fc4f002b77690bc0afeb4255ac9"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.2"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "91b5dcf362c5add98049e6c29ee756910b03051d"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.3"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "6f14549f7760d84b2db7a9b10b88cd3cc3025730"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.14"

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

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "c9551dd26e31ab17b86cbd00c2ede019c08758eb"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+1"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "58f25e56b706f95125dcb796f39e1fb01d913a71"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.10"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.NaNMath]]
git-tree-sha1 = "737a5957f387b17e74d4ad2f440eb330b39a62c5"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.0"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ab05aa4cc89736e95915b01e7279e61b1bfe33b8"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.14+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "621f4f3b4977325b9128d5fae7a8b4829a0c2222"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.2.4"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "8162b2f8547bc23876edd0c5181b27702ae58dce"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.0.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "bb16469fd5224100e422f0b027d26c5a25de1200"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.2.0"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "88ee01b02fba3c771ac4dce0dfc4ecf0cb6fb772"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.27.5"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "d3538e7f8a790dc8903519090857ef8e1283eecd"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.5"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "ad368663a5e20dbb8d6dc2fddeefe4dae0781ae8"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+0"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "dc1e451e15d90347a7decc4221842a022b011714"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.5.2"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "cdbd3b1338c72ce29d9584fdbe9e9b70eeb5adca"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.1.3"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "4f6ec5d99a28e1a749559ef7dd518663c5eca3d5"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c3d8ba7f3fa0625b062b82853a7d5229cb728b6b"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.2.1"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8977b17906b0a1cc74ab2e3a05faa16cf08a8291"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.16"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "57617b34fa34f91d536eb265df67c2d4519b8b98"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.5"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "34db80951901073501137bdbc3d5a8e7bbd06670"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.1.2"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╟─858bb691-6680-471e-9375-135c19acca8e
# ╟─0e838745-b813-45fd-8f59-54a3bde304c9
# ╠═3d85045e-b5e1-11ec-20a5-192793e5cee1
# ╠═371d1d5c-fc18-4850-8f0a-ca4dd8124fe7
# ╟─b3ce38ef-94a1-484e-87ce-9bb5e6b9fa5c
# ╠═262eecfc-a33e-4c25-b606-15e875d38270
# ╟─d0280b13-2792-4c3a-bc5f-93f98558d572
# ╠═c0a2d148-bfc6-4d04-a6c4-bc97e3adce5a
# ╠═bd9dc94d-242b-431d-af77-d145fec1cf32
# ╟─81f0fb85-856b-44bf-8413-40feb28f41c3
# ╠═98bff7fa-5b34-4e7c-874a-4311bd28277e
# ╟─0e80f389-032f-4902-b9b0-1df1380f13e1
# ╠═c5e41a00-bdb2-4e3e-aedc-d64df656035e
# ╟─3e00fb77-8606-460c-9185-eec318863bbe
# ╠═eeb9a2f8-24c8-434b-9df0-03ca654fcc76
# ╟─e6e563e5-badb-4dbc-b615-3559179cf142
# ╠═9f22c1a7-cd17-4eb7-b044-5e82dd799567
# ╠═9c19d90b-cb29-4db7-8b2e-16928e0bc275
# ╟─b74de333-35da-4cbf-a760-f2f0540ebcad
# ╠═40041f04-4854-43e9-aa71-2a31666f6bf4
# ╠═a07bc826-f2d3-422a-8fa3-2516268c04fc
# ╟─4180dc81-4b4f-49c7-a6bf-a9f98bbe66d9
# ╠═10823e1b-57ee-43a3-8676-9aa20424c97c
# ╠═d0c9f796-3a53-4f9f-bcef-689e2f5eb135
# ╠═a6d5a991-54eb-400c-9201-0a267953de24
# ╟─94fbfb98-fa49-45bc-b714-2ee275a38dc9
# ╠═c30bc9e7-06d8-44d9-9879-3485e90ba44a
# ╟─31fdb102-84aa-4598-b922-a1f0b62aa6d0
# ╠═eafec17b-cbef-4c70-898f-f9730cf61250
# ╟─c8455deb-885b-4db2-9029-86300fb25216
# ╠═6d343cad-8b34-4e90-aab9-d182644ce22a
# ╟─479aecb4-15ce-4ac8-92d2-0369de55bf6f
# ╟─5c359ca8-8780-4989-8e8a-74a8201712e4
# ╠═0dbe5118-81b7-4299-ba3b-6bb009630d6b
# ╟─07610b62-6daa-4f2a-8f25-d8c773a7f7f3
# ╠═9660a895-0a2e-4756-9750-2b2c0530046e
# ╠═68607d72-5cbc-4492-a577-a9cefa2cdc98
# ╟─44fe9368-7a43-457b-8b1a-9b42f63c30a8
# ╠═a944fc59-b27d-4cab-a92a-5e7eb93b17f1
# ╠═0cf1d599-9daa-4208-b969-758281cc9119
# ╠═6d0d2e80-0065-41b7-a18b-640b115b0245
# ╠═050c269d-ae5a-440f-ae34-107e5403cf47
# ╟─145d950e-c721-4d93-bd3b-8bb8f6ca5e62
# ╟─09fee4d4-f292-451c-a11c-ba9952ee2a51
# ╠═74e18286-34b1-408a-a638-938b876390b2
# ╟─f5764f3c-c676-4c6e-af10-d622ed8a89ed
# ╟─530de511-8741-4829-9211-b449f6bcfb21
# ╟─d6f3de3b-c904-4228-a7fa-c60319b1368b
# ╠═f0b9a7bb-f2dc-4ed1-9b25-a2633d3924b7
# ╟─24e50c10-3535-4d78-802a-881c3e310870
# ╟─72ea762e-6500-434b-ba77-5792107e9be2
# ╠═a3faebed-c219-4dcb-896a-0b12e6348e71
# ╟─3976c748-b2f1-46ed-b757-ba479c5f8481
# ╟─14bc2bcb-5195-4820-9212-9fa0956c4b2b
# ╠═465b74a5-6539-4c62-aed8-24ec7cc9b7e4
# ╟─4ab01573-7a8c-4dd3-a138-8c43f867e562
# ╠═e3715c45-600f-4c65-b305-8968be2c31e5
# ╠═cbc12bcc-ffbc-4975-bb46-8e944b0dbfd4
# ╟─aca5795d-11a4-4c96-a73a-634003558338
# ╟─ce77c338-8eca-48bb-be4f-ba14a46c7f69
# ╟─598435d7-e56c-45d9-ade4-4844b640b530
# ╟─2662eba2-7cb3-4085-87bc-812ed2625046
# ╠═29543f67-8ed6-4012-81c4-5bd78659e8a9
# ╠═9500819d-cae4-48ba-8a0d-bb27808c17cb
# ╟─70f68c93-1e57-4bd7-ae5c-000fa7ff2b8a
# ╟─8cb07531-4d95-4c80-8b58-171487df1951
# ╟─7037f111-bd5c-40a6-bc65-6637cc67c0f8
# ╠═4c9373f8-f8eb-4ae4-a1af-826839db8389
# ╟─68a077f0-1280-4f52-87ec-9514df53fb5d
# ╟─37009b68-6b70-4204-ac33-19818d8fc348
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
