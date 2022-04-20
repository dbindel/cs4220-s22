### A Pluto.jl notebook ###
# v0.18.1

using Markdown
using InteractiveUtils

# ╔═╡ 81527b3c-de3f-474e-8c2a-18a4cb2c694e
using LinearAlgebra

# ╔═╡ 59f5ecf8-78d2-460d-abed-7c08cfd47308
using Plots

# ╔═╡ a70b8507-7669-4b08-a216-7fe944723bbd
using LinearMaps

# ╔═╡ 0f7a0a41-7507-470c-8f8b-5767c5ef26a4
using IterativeSolvers

# ╔═╡ 2621c0ec-bf80-11ec-08c3-e3a7570d19f7
md"""
# Notebook for 2022-04-27
"""

# ╔═╡ 6a876ff8-8a2f-4da8-a0be-87b76027adda
md"""
## Computing with constraints

Recall that our basic problem is

$$\mbox{minimize } \phi(x) \mbox{ s.t. } x \in \Omega$$

where the feasible set $\Omega$ is defined by equality and inequality conditions

$$\Omega = \{ x \in {\mathbb{R}}^n : c_i(x) = 0, i \in \mathcal{E} \mbox{ and }
    c_i(x) \leq 0, i \in \mathcal{I} \}.$$

In the last lecture, we described three different ways to formulate constrained optimization
problem that allow us to build on techniques we previously explored from
unconstrained optimization and equation solving:

1.  *Constraint elimination* (for equality constraints): Find a
    parameterization $g : {\mathbb{R}}^{n-m} \rightarrow \Omega$
    formulations and minimize $\phi(g(y))$ without constraints. This
    requires that the constraints be simple (e.g. affine equality
    constraints).

2.  *Barriers and penalties:* Add a term to the objective function
    depending on some parameter $\mu$. This term penalizes $x$ values
    that violate the constraint (penalty methods) or that come close to
    $\partial \Omega$ from the inside (barrier methods). As
    $\mu \rightarrow 0$, the unconstrained minimum of the modified
    problems converges to the constrained minimum of the original.

3.  *Lagrange multipliers*: Add new variables (multipliers)
    corresponding to “forces” needed to enforce the constraints. The
    *KKT conditions* are a set of nonlinear equations in the
    original unknowns and the multipliers that characterize constrained
    stationary points.

Our goal now is to sketch how modern constrained optimization algorithms
incorporate these different ways of looking at the problem. A full
treatment is well beyond the scope of the class, but we hope to give you
at least the keywords you will need should you encounter them in a
textbook, paper, or a cocktail party. Ideally, knowing something
about what happens in the algorithms will also help you think about
which of various equivalent formulations of an optimization problem will
be more (or less) friendly to solvers. The plan is to first give a "lay
of the land" of different families of algorithms, then to give a more
detailed treatment with the running example of linearly constrained
quadratic programs.

For more details, there are some excellent textbooks in the field;
some texts that I really like from my own shelf include:

- [*Numerical Optimization*, Nocedal and Wright](https://link.springer.com/book/10.1007/978-0-387-40065-5)
- [*Practical Optimization*, Gill, Murray, and Wright](https://doi.org/10.1137/1.9781611975604)
- [*Nonlinear Programming*, Bertsekas](http://www.athenasc.com/nonlinbook.html)
"""

# ╔═╡ 35da672d-7617-4f5e-905b-385b3df233fd
md"""
## Lay of the Land

As we mentioned before, problems with *inequality* constraints tend
to be more difficult than problems with *equality* constraints
alone, because it involves the combinatorial subproblem of figuring out
which constraints are *active* (a constraint $c_i(x) \leq 0$ is
active if $c_i(x) = 0$ at the optimum). Once we have figured out the set
of active constraints, we can reduce an inequality-constrained problem
to an equality-constrained problem. Hence, the purely
equality-constrained case is an important subproblem for
inequality-constrained optimizers, as well as a useful problem class in
its own right.

For problems with only equality constraints, there are several standard
options:

-   *Null space methods* deal with linear equality constraints by
    reducing to an unconstrained problem in a lower-dimensional space.

-   *Projected gradient methods* deal with simple equality
    constraints by combining a (scaled) gradient step and a projection
    onto a constraint set.

-   *Penalty methods* approximately solve an equality-constrained
    problem through an unconstrained problem with an extra term that
    penalizes proposed soutions that violate the constraints. That is,
    we use some constrained minimizer to solve

    $$\mbox{minimize } \phi(x) + \frac{1}{\mu} \sum_{i \in\mathcal{E}} c_i(x)^2.$$

    As $\mu \rightarrow 0$, the minimizers to these approximate problems
    approach the true minimizer, but the Hessians that we encounter
    along the way become increasingly ill-conditioned (with condition
    number proportional to $\mu^{-1}$).

-   *KKT solvers* directly tackle the first-order optimality
    conditions (the KKT conditions), simultaneously computing the
    constrained minimizer and the associated Lagrange multipliers.

-   *Augmented Lagrangian* methods combine the advantages of penalty
    methods and the advantages of the penalty formulation. In an
    augmented Lagrangian solver, one finds critical points for the
    augmented Lagrangian 

    $$\mathcal{L}(x, \lambda; \mu) =
        \phi(x) + \frac{1}{\mu} \sum_{i \in \mathcal{E}} c_i(x)^2 + \lambda^T c(x)$$

    by alternately adjusting the penalty parameter $\mu$ and the
    Lagrange multipliers.

In the inequality-constrained case, we have

-   *Active set methods* solve (or approximately solve) a sequence
    of equality-constrained subproblems, shuffling constraints into and
    out of the proposed working set along the way. These methods are
    particularly attractive when one has a good initial estimate of the
    active set.

-   *Projected gradient methods* deal with simple inequality
    constraints by combining a (scaled) gradient step and a projection
    onto a constraint set.

-   *Barrier methods* and *penalty methods* add a term to the
    objective function in order to penalize constraint violations or
    near-violations; as in the equality-constrained case, a parameter
    $\mu$ governs a tradeoff between solution quality and conditioning
    of the Hessian matrix.

-   *Interior point methods* solve a sequence of barrier subproblems
    using a continuation strategy, where the barrier or penalty
    parameter $\mu$ is the continuation parameter. This is one of the
    most popular modern solver strategies, though active set methods may
    show better performance when one “warm starts” with a good initial
    guess for the solution and the active set of constraints.

As with augmented Lagrangian strategies in the equality-constrained
case, state-of-the art strategies for inequality-constrained problems
often combine approaches, using continuation with respect to a barrier
parameters as a method of determining the active set of constraints in
order to get to an equality-constrained subproblem with a good initial
guess for the solution and the Lagrange multipliers.

The *sequential quadratic programming* (SQP) approach for nonlinear
optimization solves a sequence of linearly-constrained quadratic
optimization problems based on Taylor expansion of the objective and
constraints about each iterate. This generalizes simple Newton iteration
for unconstrained optimization, which similarly solves a sequence of
quadratic optimization problems based on Taylor expansion of the
objective. Linearly-constrained quadratic programming problems are hence
an important subproblem in SQP solvers, as well as being an important
problem class in their own right.
"""

# ╔═╡ 4aeeda56-e034-4902-aa56-62256632b1c3
md"""
## Quadratic programs with equality constraints

We begin with a simple case of a quadratic objective and linear equality
constraints: 

$$\begin{aligned}
  \phi(x) &= \frac{1}{2} x^T H x - x^T d \\
  c(x) &= A^T x-b = 0,
\end{aligned}$$

where $H \in {\mathbb{R}}^{n \times n}$ is symmetric and positive definite
*on the null space of $A^T$* (it may be indefinite or singular
overall), $A \in {\mathbb{R}}^{n \times m}$ is full rank with $m < n$,
and $b \in {\mathbb{R}}^m$. Not only are such problems useful in their
own right, solvers for these problems are also helpful building blocks
for more sophisticated problems — just as minimizing an unconstrained
quadratic can be seen as the starting point for Newton’s method for
unconstrained optimization.
"""

# ╔═╡ 2353c8ad-2008-47e1-81f2-56dc94d6b350
begin
	# Set up a test problem for linearly-constrained QP (2D so that we can plot)
	H = [4.0  1.0 ;
    	 1.0  4.0 ]
	d = [0.5 ; -2.0]
	A = [1.0 ; 1.0]
	b = [1.0]

	ϕ1(xy) = xy'*H*xy/2 - xy'*d
	c1(xy) = A'*x - b
end

# ╔═╡ 19e8bc5b-b5f3-44a4-bc03-b9e4f77be193
let
	xx = range(-3, 3, length=100)
	plot(xx, xx, (x,y) -> ϕ1([x; y]), st=:contour, legend=false)
	plot!(xx, 1.0 .- xx, linewidth=2)
end

# ╔═╡ 0c518780-7f66-4de1-9519-29c62f8089f0
md"""
### Constraint elimination (linear constraints)

As discussed last time, we can write the space of solutions to the
constraint equations in terms of a (non-economy) QR decomposition of
$A$:

$$A =
  \begin{bmatrix} Q_1 & Q_2 \end{bmatrix}
  \begin{bmatrix} R_1 \\ 0 \end{bmatrix}$$

where $Q_2$ is a basis for the null space of $A^T$. The set of solutions satisfying
the constraints $A^T x = b$ is

$$\Omega = \{ u + Q_2 y : y \in {\mathbb{R}}^{(n-m)}, u = Q_1 R_1^{-T} b \};$$

here $u$ is a *particular solution* to the problem. If we substitute
this parameterization of $\Omega$ into the objective, we have the
unconstrained problem 

$$\mbox{minimize } \phi(u + Q_2 y).$$

While we can substitute directly to get a quadratic objective in terms of $y$, 
it is easier (and a good exercise in remembering the chain rule) to compute
the stationary equations

$$\begin{aligned}
  0
  &= \nabla_y \phi(u + Q_2 y) 
  = \left(\frac{\partial x}{\partial y}\right)^T \nabla_x \phi(u+Q_2 y) \\
  &= Q_2^T (H (Q_2 y + u) - d) 
  = (Q_2^T H Q_2) y - Q_2^T (d-Hu).
\end{aligned}$$

In general, even if $A$ is sparse, $Q_2$ may be dense, and so even if $H$ is dense,
we find that $Q_2^T H Q_2$ is dense.
"""

# ╔═╡ da3456a0-d474-4824-ba88-2586dce999f7
begin
	# Solve the 2-by-2 problem via a null-space approach
	F = qr(A)
	Q = F.Q * I
	Q1 = Q[:,[1]]
	Q2 = Q[:,[2]]

	u_ns = Q1*(F.R'\b)
	H22 = Q2'*H*Q2
	r2  = Q2'*(d-H*u_ns)
	y_ns   = H22\r2
	x_ns   = u_ns + Q2*y_ns
end

# ╔═╡ c882f690-9bef-44d1-bd88-93d02aa18e03
let
	xx = range(-3, 3, length=100)
	plot(xx, xx, (x,y) -> ϕ1([x; y]), st=:contour, legend=false)
	plot!(xx, 1.0 .- xx, linewidth=2)
	plot!([u_ns[1]], [u_ns[2]], markercolor=:white, marker=true)
	plot!([x_ns[1]], [x_ns[2]], marker=true)
end

# ╔═╡ 57bc3d3c-6dff-4e18-a7f1-e77e62264cef
md"""
Finding a particular solution and a null space basis via QR is great for numerical
stability, but it may not be ideal when the matrices involved are sparse or structured.
An alternative is to use a sparse LU factorization of $A^T$:

$$P A^T Q = L \begin{bmatrix} U_1 U_2 \end{bmatrix}.$$

where the $U_1$ submatrix is upper triangular.  A particular solution is then

$$x = Q \begin{bmatrix} U_1^{-1} L^{-1} P b \\ 0 \end{bmatrix}$$

and the null space is spanned by

$$Q^T 
  \begin{bmatrix}
    -U_1^{-1} U_2 \\
    I
  \end{bmatrix}$$

This reformulation may be particularly attractive if $A$ is large, sparse, and close
to square.  Note that pivoting on rectangular constraint matrices needs to be done
carefully, e.g. using so-called *rook pivoting* strategies that maintain numerical
stability on full-rank matrices with rank-deficient submatrices.
"""

# ╔═╡ d9beb435-52f9-42e2-b14e-39f24c6b2530
md"""
### Projected gradient and conjugate gradients

The *projected gradient* is a variant of gradient descent for constrained problem.  One assumes that we have
a projection operator $P$ such that $P(x)$ is the closest point to $x$ satisfying the constraint; the iteration
is then

$$x_{k+1} = P\left( x_k - \alpha_k \nabla \phi(x_k) \right).$$

That is, we take an (unconstrained) gradient descent step, then project back to satisfy the constraint.
It's an easy enough method to code, provided you have the projection $P$.

For our linear equality constraints the projection can be computed by a least squares type of solve:

$$\begin{aligned}
  P(x) &= x + A(A^T A)^{-1} (b-A^T x) \\
       &= (A^T)^\dagger b + (I-AA^\dagger) x \\
       &= (A^T)^\dagger b + (I-\Pi) x
\end{aligned}$$

Note that $(A^T)^\dagger b$ is the minimal norm solution to the constraint equation, and the range space of
$I-\Pi = I-AA^\dagger$ is the null space of $A^T$, so this is similar to the picture we saw with the constraint
elimination approach.  And, of course, the gradient in this case is just the residual $r_k = Hx_k - d$.

If we start with a point $x_0$ that is consistent with the constraints, then each successive
point remains on our linear constraint surface; in this case, we can simplify the iteration to

$$x_{k+1} = x_k - \alpha_k (I-\Pi) r_k.$$

This is a stationary iteration for the underdetermined consistent equation

$$(I-\Pi) (Hx_k-d) = 0.$$

Unfortunately, the projected gradient iteration may converge rather slowly.  A tempting thought is to
use a scaled version of the gradient, but the naive version of this iteration will in general converge
to the wrong point unless the projection operator is re-defined in terms of the distance associated with
the same scaling matrix.

If the relevant projection is available, a potentially more attractive route for this problem 
is to write $x = u + z$ for some particular solution $u$ (as in the null space approach) and then
use a method like conjugate gradients on the system

$$(I-\Pi) H (I-\Pi) z = (I-\Pi) (d - Hu).$$

It turns out that the Krylov subspace generated by this iteration remains consistent with the constraint,
and so -- somewhat surprisingly at first glance -- the method continues to work even
though $(I-\Pi) H (I-\Pi)$ is singular.
"""

# ╔═╡ 6ddc1065-6a1a-459b-9263-2c24091b28eb
let
	resid_proj(z) = z-Q1*(Q1'*z)  # Function to apply I-Π
	rhs = resid_proj(d-H*u_ns)    # Compute (I-Π)(d-Hu)

	# Define a 2-by-2 symmetric linear map (I-Π)H(I-Π) from a matvec function
	Afun = LinearMap((z)->resid_proj(H*resid_proj(z)), 2, issymmetric=true)

	# Solve the singular system via CG
	u_ns + cg(Afun, rhs)
end

# ╔═╡ e4375cc8-b75f-4651-8826-fc935c0f8784
md"""
### Penalties and conditioning

Now consider a penalty formulation of the same equality-constrained
optimization function, where the penalty is quadratic:

$$\mbox{minimize } \phi(x) + \frac{1}{2\mu} \|A^T x-b\|^2.$$

In fact, the augmented objective function is again quadratic, and the critical
point equations are

$$(H + \mu^{-1} AA^T) x = d + \mu^{-1} A b.$$

If $\mu$ is small enough and the equality-constrained quadratic program (QP)
has a minimum, then $H+\mu^{-1} AA^T$ is guaranteed to be positive definite.
This means we can solve via Cholesky; or (if the linear system is larger)
we might use conjugate gradients.

We can analyze this more readily by changing to the $Q$ basis from the QR
decomposition of $A$ that we saw in the constraint elimination approach:

$$\begin{bmatrix}
  Q_1^T H Q_1 + \mu^{-1} R_1 R_1^T & Q_1^T H Q_2 \\
  Q_2^T H Q_1 & Q_2^T H Q_2
\end{bmatrix}
(Q^T x) =
\begin{bmatrix}
  Q_1^T d + \mu^{-1} R_1 b \\
  Q_2^T d
\end{bmatrix}$$

Taking a Schur complement, we have

$$(\mu^{-1} R_1 R_1^T + F)(Q_1^T x) = \mu^{-1} R_1 b - g$$

where

$$\begin{aligned}
  F &= Q_1^T H Q_1 - Q_1^T H Q_2 (Q_2^T H Q_2)^{-1} Q_2^T H Q_1 \\
  g &= [I - Q_1^T H Q_2 (Q_2^T H Q_2)^{-1} Q_2^T] d
\end{aligned}$$

As $\mu \rightarrow 0$, the first row of equations is dominated by the
$\mu^{-1}$ terms, and we are left with

$$R_1 R_1^T (Q_1^T x) - R_1 b \rightarrow 0$$

i.e. $Q_1 Q_1^T x$ is converging to $u = Q_1 R_1^{-T} b$, the particular
solution that we saw in the case of constraint elimination. Plugging this
behavior into the second equation gives

$$(Q_2^T H Q_2) (Q_2^T x) - Q_2^T (d-Hu) \rightarrow 0,$$

i.e. $Q_2^T x$ asymptotically behaves like $y$ in the previous example.
We need large $\mu$ to get good results if the constraints are ill-posed or if
$Q_2^T H Q_2$ is close to singular. But in general the condition number
scales like $O(\mu^{-1})$, and so large values of $\mu$ correspond to
problems that are numerically unattractive, as they may lead to large
errors or (for iterative solvers) to slow convergence.
"""

# ╔═╡ 713c1176-8473-41d5-b577-60138650b801
let
	# Demonstrate the solve with a moderate penalty
	μ = 1e-4
	xhat = (H+A*A'/μ)\(d+A*b[1]/μ)

	xx = range(-3, 3, length=100)
	p = plot(xx, xx, (x,y) -> ϕ1([x; y]), st=:contour, legend=false)
	plot!(xx, 1.0 .- xx, linewidth=2)
	plot!([u_ns[1]], [u_ns[2]], markercolor=:white, marker=true)
	plot!([xhat[1]], [xhat[2]], marker=true)

md"""
Error at μ=$μ: $(norm(xhat-x_ns))

$p
"""
end

# ╔═╡ e8385595-ef60-4cf7-bd68-7528a56ccad1
let
	# Vary penalty to illustrate issues -- uniform improvement with smaller penalty until ill-conditioning kills us
	μs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14]
	errs = []
	for μ in μs
	    xhat = (H+A*A'/μ)\(d+A*b[1]/μ)
	    push!(errs, norm(xhat-x_ns))
	end
	plot(μs, errs, xscale=:log10, yscale=:log10, legend=false)
end

# ╔═╡ 4f0b0712-871d-4d1d-a56d-345ae2ce123f
md"""
### Lagrange multipliers and KKT systems

The KKT conditions for our equality-constrained problem say that the
gradient of

$$L(x,\lambda) = \phi(x) + \lambda^T (A^T x-b)$$

should be
zero. In matrix form, the KKT system (saddle point system)

$$\begin{bmatrix}
    H & A \\
    A^T & 0
  \end{bmatrix}
  \begin{bmatrix} x \\ \lambda \end{bmatrix} =
  \begin{bmatrix} d \\ b \end{bmatrix}.$$ 

If $A$ and $H$ are
well-conditioned, then so is this system, so there is no bad numerical
behavior. The system also retains whatever sparsity was present in the
original system matrices $H$ and $A$. However, adding the Lagrange
multipliers not only increases the number of variables, but the extended
system lacks any positive definiteness that $H$ may have.

When there are relatively few constraints and a fast solver with $H$
is available, an attractive way to solve this KKT system is the so-called
range-space method, which we recognize as just block Gaussian elimination:

$$\begin{aligned}
  A^T H^{-1} A \lambda &= A^T H^{-1} d - b \\
  x &= H^{-1} (d - A\lambda)
\end{aligned}$$

Rewritten as we might implement it, we have

$$\begin{aligned}
  H x_0 &= d \\
  H Y   &= A \\
  (A^T Y) \lambda &= A^T x_0 - b \\
  x &= x_0 - Y \lambda
\end{aligned}$$

The KKT system is closely related to the penalty formulation that we saw
in the previous subsection, in that if we use Gaussian elimination to
remove the variable $\lambda$ in 

$$\begin{bmatrix}
    H & A \\
    A^T & -\mu I
  \end{bmatrix}
  \begin{bmatrix} \hat{x} \\ \lambda \end{bmatrix} =
  \begin{bmatrix} d \\ b \end{bmatrix},$$ 

we have the Schur complement
system 

$$(H+\mu^{-1} AA^T) \hat{x} = d + \mu^{-1} A b,$$ 

which is
identical to the stationary point condition for the quadratically
penalized objective.
"""

# ╔═╡ bfde37b4-701b-40f4-9e53-00446e292bfd
xλ = [H A; A' 0.0] \ [d; b]

# ╔═╡ f04462de-8c7d-4b67-95f7-8a762828bc83
x_ns-xλ[1:end-1]  # Error

# ╔═╡ fa8c3589-329f-444c-b46b-c20fe7adbe43
H*xλ[1:end-1]-d  # ∇ϕ(x)

# ╔═╡ ef78e986-f002-4a40-9e78-1ed2995f6c11
xλ[end]*A  # λ ∇c(x)

# ╔═╡ 71f347e3-a9c1-494a-9c97-e5e9a48ae0a1
md"""
Note that the constrained stationarity condition

$$\nabla \phi(x_*) + \lambda \nabla c(x_*) = 0,$$

and we can use this to estimate the Lagrange multipliers from an approximation via a penalty method.
"""

# ╔═╡ b2e4f9e1-fc39-4f9b-a6f0-8ef5e27ce965
let
	μ = 1e-4
	xhat = (H+A*A'/μ)\(d+A*b[1]/μ)
	r = H*xhat-d
	md"λ ≈ $(-norm(r)^2/(A'*r))"
end

# ╔═╡ c3e00aad-57a4-4d98-8587-2e5d74bf1e3a
md"""
### Uzawa iteration

Block Gaussian elimination is an attractive approach when we have a fast solver for $H$ and there are not
too many constraints.  When there are a relatively large number of constraints, we might seek an alternate
method.  One such method is the *Uzawa iteration*

$$\begin{aligned}
  H x_{k+1} &= d - A \lambda_k \\
  \lambda_{k+1} &= \lambda_{k} + \omega (A^T x_{k+1}-b)
\end{aligned}$$

where $\omega > 0$ is a relaxation parameter.  We can eliminate $x_{k+1}$ to get the iteration

$$\lambda_{k+1} 
  = \lambda_k + \omega (A^T H^{-1} (d-A\lambda_k) - b)
  = (I-\omega A^T H^{-1} A) \lambda_k + \omega (A^T H^{-1} d  - b),$$

which is a Richardson iteration on the Schur complement equation $(A^T H^{-1} A) \lambda = A^T H^{-1} d - b$.
We can precondition and accelerate the Uzawa iteration in a variety of ways, as you might guess from our earlier
discussion of iterative methods for solving linear systems.
"""

# ╔═╡ ad04ec17-e2db-403e-99f1-1105c220e69a
md"""
### Augmenting the Lagrangian

From a solver perspective, the block 2-by-2 structure of the KKT system
looks highly attractive. Alas, we do *not* require that $H$ be
positive definite, nor even that it be nonsingular; to have a unique
global minimum, we only need positive definiteness of the projection of
$H$ onto the null space (i.e. $Q_2^T H Q_2$ should be positive
definite). This means we cannot assume that (for example) $H$ will admit
a Cholesky factorization.

The augmented Lagrangian approach can be seen as solving the constrained
system

$$\mbox{minimize } \frac{1}{2} x^T H x - d^T x + \frac{1}{2\mu} \|A^T x-b\|^2
  \mbox{ s.t. } A^T x = b.$$

The term penalizing nonzero $\|A^T x-b\|$ is, of course, irrelevant at points 
satisfying the constraint $A^T x = b$. Hence, the constrained minimum for this
augmented objective is identical to the constrained minimum of the original objective.
However, if the KKT conditions for the modified objective take the form

$$\begin{bmatrix}
    H+\mu^{-1}AA^T & A \\
    A^T & 0
  \end{bmatrix}
  \begin{bmatrix} x \\ \lambda \end{bmatrix} =
  \begin{bmatrix} d + \mu^{-1} A b \\ b \end{bmatrix}.$$

Now we do not necessarily need to drive $\mu$ to zero to obtain a good solution;
but by choosing $\mu$ small enough, we can ensure that $H + \mu^{-1} AA^T$
is positive definite (assuming that the problem is convex subject to the
constraint).  This can be helpful when we want to use a Cholesky factorization
or a method like CG, but the original $H$ matrix is indefinite or singular.
"""

# ╔═╡ 3c21a96f-5f1a-432b-a697-b5e9bdb9c7bc
begin
	# Set up a non-positive-definite problem for linearly-constrained QP (2D so that we can plot)
	H2 = [4.0  1.0 ;
	      1.0  -1.0 ]
	
	ϕ2(xy) = xy'*H2*xy/2 - xy'*d
	c2(xy) = A'*xy - b[1]
	ϕ3(xy) = ϕ2(xy) + 3*norm(c2(xy))^2
	xy_sol = [H2 A; A' 0.0] \ [d; b]
end

# ╔═╡ dafa20fa-1f69-46eb-9c90-7620c6bc62c1
let
	xx = range(-3, 3, length=100)
	plot(xx, xx, (x,y) -> ϕ2([x; y]), st=:contour, legend=false)
	plot!(xx, xx, (x,y) -> ϕ3([x; y]), st=:contour, linestyle=:dash)
	plot!(xx, 1.0 .- xx, linewidth=2)
	plot!([xy_sol[1]], [xy_sol[2]], marker=true)
end

# ╔═╡ 60d75244-3ca0-41aa-8bfa-5d458e436570
let
	# Sanity check: do we have a constrained min (vs max or saddle)?
	xx = range(-3, 3, length=100)
	plot(xx, [ϕ2([x; 1.0-x]) for x in xx], legend=false)
	plot!(xx, [ϕ3([x; 1.0-x]) for x in xx], linestyle=:dash)
end

# ╔═╡ ebc33f64-1009-462d-bb5b-84a1e21ec8bd
# Find the solution via method of Lagrange multipliers
[H2 A; A' 0.0] \ [d; b]

# ╔═╡ 6ec8aefe-4eda-4e7d-b333-5947162e380b
# Augment the Lagrangian so the (1,1) submatrix is positive definite
let
	σ = 2.0
	[H2+A*A'/σ A; A' 0.0] \ [d+A*b[1]/σ; b]
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
IterativeSolvers = "42fd0dbc-a981-5370-80f2-aaf504508153"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LinearMaps = "7a12625a-238d-50fd-b39a-03d52299707e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"

[compat]
IterativeSolvers = "~0.9.2"
LinearMaps = "~3.6.1"
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
git-tree-sha1 = "b153278a25dd42c65abbf4e62344f9d22e59191b"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.43.0"

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

[[deps.IterativeSolvers]]
deps = ["LinearAlgebra", "Printf", "Random", "RecipesBase", "SparseArrays"]
git-tree-sha1 = "1169632f425f79429f245113b775a0e3d121457c"
uuid = "42fd0dbc-a981-5370-80f2-aaf504508153"
version = "0.9.2"

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

[[deps.LinearMaps]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics"]
git-tree-sha1 = "1693d6d0dfefd24ee97ffc5ea91f1cd2cf77ef6e"
uuid = "7a12625a-238d-50fd-b39a-03d52299707e"
version = "3.6.1"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "a970d55c2ad8084ca317a4658ba6ce99b7523571"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.12"

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
git-tree-sha1 = "8d7530a38dbd2c397be7ddd01a424e4f411dcc41"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.2.2"

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
# ╟─2621c0ec-bf80-11ec-08c3-e3a7570d19f7
# ╠═81527b3c-de3f-474e-8c2a-18a4cb2c694e
# ╠═59f5ecf8-78d2-460d-abed-7c08cfd47308
# ╠═a70b8507-7669-4b08-a216-7fe944723bbd
# ╠═0f7a0a41-7507-470c-8f8b-5767c5ef26a4
# ╟─6a876ff8-8a2f-4da8-a0be-87b76027adda
# ╟─35da672d-7617-4f5e-905b-385b3df233fd
# ╟─4aeeda56-e034-4902-aa56-62256632b1c3
# ╠═2353c8ad-2008-47e1-81f2-56dc94d6b350
# ╠═19e8bc5b-b5f3-44a4-bc03-b9e4f77be193
# ╟─0c518780-7f66-4de1-9519-29c62f8089f0
# ╠═da3456a0-d474-4824-ba88-2586dce999f7
# ╠═c882f690-9bef-44d1-bd88-93d02aa18e03
# ╟─57bc3d3c-6dff-4e18-a7f1-e77e62264cef
# ╟─d9beb435-52f9-42e2-b14e-39f24c6b2530
# ╠═6ddc1065-6a1a-459b-9263-2c24091b28eb
# ╟─e4375cc8-b75f-4651-8826-fc935c0f8784
# ╠═713c1176-8473-41d5-b577-60138650b801
# ╠═e8385595-ef60-4cf7-bd68-7528a56ccad1
# ╟─4f0b0712-871d-4d1d-a56d-345ae2ce123f
# ╠═bfde37b4-701b-40f4-9e53-00446e292bfd
# ╠═f04462de-8c7d-4b67-95f7-8a762828bc83
# ╠═fa8c3589-329f-444c-b46b-c20fe7adbe43
# ╠═ef78e986-f002-4a40-9e78-1ed2995f6c11
# ╟─71f347e3-a9c1-494a-9c97-e5e9a48ae0a1
# ╠═b2e4f9e1-fc39-4f9b-a6f0-8ef5e27ce965
# ╟─c3e00aad-57a4-4d98-8587-2e5d74bf1e3a
# ╟─ad04ec17-e2db-403e-99f1-1105c220e69a
# ╠═3c21a96f-5f1a-432b-a697-b5e9bdb9c7bc
# ╠═dafa20fa-1f69-46eb-9c90-7620c6bc62c1
# ╠═60d75244-3ca0-41aa-8bfa-5d458e436570
# ╠═ebc33f64-1009-462d-bb5b-84a1e21ec8bd
# ╠═6ec8aefe-4eda-4e7d-b333-5947162e380b
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
