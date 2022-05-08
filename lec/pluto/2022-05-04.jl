### A Pluto.jl notebook ###
# v0.18.1

using Markdown
using InteractiveUtils

# ╔═╡ 9e2bbd06-cb48-11ec-25d6-7df18d36be05
md"""
# Lecture notes for 2022-05-04
"""

# ╔═╡ 27e9722a-400d-4e7f-a4e8-bf6a363db82a
md"""
## Lay of the Land

In the landscape of continuous optimization problems, there are three ways for things to be hard:

- Lack of smoothness
- Lack of convexity (or other structure guaranteeing unique global minimizers)
- Lack of information about the function

So far, we have only considered difficulties associated with lack of convexity, and those only superficially -- most of our algorithms only give local minimizers, and we haven't talked about finding global optima.  We have assumed everything has as many derivatives as we would like, and that we are at least able to compute gradients of our objective.

Today we discuss optimization in one of the hard cases.  For this class, we will not deal with the case of problems with very hard global structure, other than to say that this is a land where heuristic methods (simulated annealing, genetic algorithms, and company) may make sense.  But there are some useful methods that are available for problems where the global structure is not so hard as to demand heuristics, but the problems are hard in that they are "black box" -- that is, we are limited in what we can compute to looking at function evaluations.

Before describing some methods, I make two pleas.

First, consider these only after having thoughtfully weighed the pros and cons of gradient-based methods.  If the calculus involved in computing the derivatives is too painful, consider a computer algebra system, or look into a tool for automatic differentiation of computer programs.  Alternately, consider whether there are numerical estimates of the gradient (via finite differences) that can be computed more quickly than one might expect by taking advantage of the structure of how the function depends on variables.  But if you really have to work with a black box code, or if the pain of computing derivatives (even with a tool) is too great, a gradient-free approach may be for you.

Second, do not fall into the trap of thinking these methods should be simple to implement.  Nelder-Mead is perhaps simple to implement, which is one reason why it remains so popular; it also fails to converge on many examples, and rarely converges fast.  There are various pattern search methods and model-based methods with more robust convergence or better convergence rates, and there exist good implementations in the world (though Powell himselff has passed, his [PDFO suite](https://www.pdfo.net/) is still going strong!).  And there are much more thorough texts and reviews than this set of lecture notes; a few I like include:

- [Powell, "Direct search algorithms for optimization calculations"](https://doi.org/10.1017/S0962492900002841) -- covers a lot of methods, and particularly Powell's algorithms
- [Powell, "A view of algorithms for optimization without derivatives"](http://www.damtp.cam.ac.uk/user/na/NA_papers/NA2007_03.pdf) -- played a big role in this presentation
- [Kolda, Lewis, and Torczon, "Optimization by direct search: new perspectives on some classical and modern methods"](https://epubs.siam.org/doi/10.1137/S003614450242889) -- deals with one interesting family of methods (very thoroughly)
- [Conn, Scheinberg, and Vicente, "Introduction to Derivative-Free Optimization"](https://epubs-siam-org.proxy.library.cornell.edu/doi/book/10.1137/1.9780898718768) -- available for access via the Cornell library subscription to SIAM ebooks
- [Audet and Warren, "Derivative-Free and Blackbox Optimization"](https://link-springer-com.proxy.library.cornell.edu/book/10.1007/978-3-319-68913-5) -- a Springer textbook, but also available courtesy the Cornell library
- [Larson, Menickelly, and Wild, "Derivative-free optimization methods"](https://arxiv.org/pdf/1904.11585.pdf) -- a recent Acta Numerica survey, again by people who know what they are doing
"""

# ╔═╡ e30e3f29-d551-496e-ac09-51a2c16ba3e4
md"""
## Model-based methods

The idea behind Newton's method is to successively minimize a
quadratic *model* of the function behavior based on a second-order
Taylor expansion about the most recent guess, i.e. $x^{k+1} = x^k + p$
where

$$\operatorname{argmin}_{p} \phi(x) + \phi'(x) p + \frac{1}{2} p^T H(x) p.$$

In some Newton-like methods, we use a more approximate model, usually
replacing the Hessian with something simpler to compute and factor.
In simple gradient-descent methods, we might fall all the way back to
a linear model, though in that case we cannot minimize the model
globally -- we need some other way of controlling step lengths.
We can also explicitly incorporate our understanding of the quality of
the model by specifying a constraint that keeps us from moving outside
a "trust region" where we trust the model to be useful.

In derivative-free methods, we will keep the basic "minimize the
model" approach, but we will use models based on interpolation
(or regression) in place of the Taylor expansions of the Newton
approach.  There are several variants.
"""

# ╔═╡ 6ab21d46-69cb-4e60-acbf-97b2daf30e70
md"""
### Finite difference derivatives

Perhaps the simplest gradient-free approach (though not necessarily the most efficient) takes some existing gradient-based approach and replaces gradients with finite difference approximations.  There are a two difficulties with this approach:

- If $\phi : \mathbb{R}^n \rightarrow \mathbb{R}$, then computing the $\nabla \phi(x)$ by finite differences involves at least $n+1$ function evaluations.  Thus the typical cost per step ends up being $n+1$ function evaluations (or more), where methods that are more explicitly designed to live off samples might only use a single function evaluation per step.

- The finite difference approximations depends on a step size $h$, and their accuracy is a complex function of $h$.  For $h$ too small, the error is dominated by cancellation, revealing roundoff error in the numerical function evaluations.  For $h$ large, the error depends on both the step size and the local smoothness of the function.

The first issue (requiring $n+1$ function evaluations to get the same information content as one function-and-gradient evaluation) is not unique to finite difference
computations, and indeed tends to be a limit to a lot of derivative-free methods.
"""

# ╔═╡ 8ca58440-b7a1-4168-9955-302b02ff6fa2
md"""
### Linear models

A method based on finite difference approximations of gradients might use $n+1$ function evaluations per step: one to compute a value at some new point, and $n$ more in a local neighborhood to compute values to estimate derivatives.  An alternative is to come up with an approximate linear model for the function using $n+1$ function evaluations that may include some ``far away'' function evaluations from previous steps.

We insist that the $n+1$ evaluations form a simplex with nonzero volume; that is, to compute from evaluations at points $x_0, \ldots, x_n$, we want $\{ x_j-x_0 \}_{j=1}^n$ to be linearly independent vectors.  In that case, we can build a model $x \mapsto b^T x + c$ where $b \in \mathbb{R}^n$ and $c \in \mathbb{R}$ are chosen so that the model interpolates the function values.  Then, based on this model, we choose a new point.  A key aspect to these methods is ensuring that the computation of the affine function from the simplex remains well-conditioned.

There are many methods that implicitly use linear approximations based on interpolation over a simplex.  One that uses the concept rather explicitly is Powell's COBYLA (Constrained Optimization BY Linear Approximation), which combines a simplex-based linear approximation with a trust region.
"""

# ╔═╡ 27b739f6-296d-40e7-b93a-85ba5162291b
md"""
### Quadratic models

One can build quadratic models of a function from only function values, but to fit a quadratic model in $n$-dimensional space, we usually need $(n+2)(n+1)/2$ function evaluations -- one for each of the $n(n+1)/2$ distinct second partials, and $n+1$ for the linear part.  Hence, purely function-based methods that use quadratic models tend to be limited to low-dimensional spaces.  However, there are exceptions.  The NEWUOA method (again by Powell) uses $2n+1$ samples to build a quadratic model of the function with a diagonal matrix at second order, and then updates that matrix on successive steps in a Broyden-like way.
"""

# ╔═╡ 54e063cf-ddc3-4af6-ab1e-5bd50939678e
md"""
### Surrogates and response surfaces

Polynomial approximations are useful, but they are far from the only methods for approximating objective functions in high-dimensional spaces.  One popular approach is to use *kernel-based approximations*; for example, we might write a model

$$s(x) = \sum_{j=1}^m c_j \phi(\|x-x_j\|)$$

where the coefficients $c_j$ are chosen to satisfy $m$ interpolation conditions at points $x_1, \ldots, x_m$.  If we add a polynomial term, you'll recognize this as the form of the polyharmonic spline from project 3, but the spline interpretation is only one way of thinking about this class of approximators.  Another option is to interpret this as a Gaussian process model; this is used, for example, in most Bayesian Optimization (BO) methods.  There are a variety of other surfaces one might consider, as well.

In addition to fitting a surface that interpolates known function values, there are also methods that use *regression* to fit some set of known function values in a least squares sense.  This is particularly useful when the function values have noise.
"""

# ╔═╡ e05e48e9-1ac4-4b19-b64e-1399bcd8e18a
md"""
## Pattern search and simplex

So far, the methods we have described are explicit in building a model that approximates the function.  However, there are also methods that use a systematic search procedure in which a model does not explicitly appear.  These sometimes go under the heading of "direct search" methods.
"""

# ╔═╡ 54ee56bf-1f34-44f8-8891-5c7ae6962b20
md"""
### Nelder-Mead

The Nelder-Mead algorithm is one of the most popular derivative-free optimizers around.  For example, this is the default algorithm used for derivative free optimization with `Optim.jl`.  As with methods like COBYLA, the Nelder-Mead approach maintains a simplex of $n+1$ function evaluation points that it updates at each step.  In Nelder-Mead, one updates the simplex based on function values at the simplex corners, the centroid, and one other point; or one contracts the simplex.

Visualizations of Nelder-Mead are often quite striking: the simplex appears to crawl downhill like some sort of mathematical amoeba. But there are examples of functions where Nelder-Mead is not guaranteed to converge to a minimum at all.
"""

# ╔═╡ 20683b43-ea48-4b2e-96d6-e805b0b5e75c
md"""
### Hook-Jeeves and successors

The basic idea of *pattern search* methods is to test points in a
pattern around the current "best" point.  For example, in the
Hook-Jeeves approach (one of the earliest pattern search methods),
one would at each move evaluate $\phi(x^{(k)} \pm \Delta e_j)$ for each
of the $n$ coordinate directions $e_j$.  If one of the new points is
better than $x^{(k)}$, it becomes $x^{(k+1)}$ (and we may increase
$\Delta$ if we already took a step in this direction to get from
$x^{(k-1)}$ to $x^{(k)}$.  Of $x^{(k)}$ is better than any surrounding
point, we decrease $\Delta$ and try again.  More generally, we would
evaluate $\phi(x^{(k)} + d)$ for $d \in \mathcal{G}(\Delta)$, a
*generating set* of directions with some scale factor $\Delta$.
"""

# ╔═╡ 1788d576-59d7-40d3-b075-5d244a419a2f
md"""
## Summarizing thoughts

Direct search methods have been with us for more than half a century: the original Hook-Jeeves paper was from 1961, and the Nelder-Mead paper goes back to 1965.  These methods are attractive in that they require only the ability to compute objective function values, and can be used with "black box" codes -- or even with evaluations based on running a physical experiment!  Computing derivatives requires some effort, even when automatic differentiation and related tools are available, and so gradient-free approaches may also be attractive because of ease-of-use.

Gradient-free methods often work well in practice for solving optimization problems with modest accuracy requirements.  This is true even of methods like Nelder-Mead, for which there are examples of very nice functions (smooth and convex) for which the method is guaranteed to mis-converge.  But though the theoretical foundations for these methods have gradually improved with time, the theory for gradient-free methods is much less clear-cut than the theory for gradient-based methods.  Gradient-based methods also have a clear advantage at higher accuracy requirements.

Gradient-free methods do *not* free a user from the burden of finding a good initial guess.  Methods like Nelder-Mead and pattern search will, at best, converge to local minima.  Many heuristic methods for finding global minimizers are gradient-free; I include among these methods like simulated annealing, genetic algorithms, and Bayesian optimization techniques.  On the other hand, branch-and-bound methods that yield provable global minimizers are often heavily dependent on derivatives (or bounds proved with the help of derivatives).

Just because a method does not explicitly use gradients does not mean it doesn't rely on them implicitly.  Gradient-free methods may have just as much difficulty with functions that are discontinuous, or that have large Lipschitz constants -- particularly those methods that implicitly build a local linear or quadratic model.

In many areas in numerics, an ounce of analysis pays for a pound of computation.  If the computation is to be done repeatedly, or must be done to high accuracy, then it is worthwhile to craft an approach that takes advantage of specific problem structure.  On the other hand, sometimes one just wants to do a cheap exploratory computation to get started, and the effort of using a specialized approach may not
be warranted.  An overview of the options that are available is useful for approaching these tradeoffs intelligently.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.2"
manifest_format = "2.0"

[deps]
"""

# ╔═╡ Cell order:
# ╟─9e2bbd06-cb48-11ec-25d6-7df18d36be05
# ╟─27e9722a-400d-4e7f-a4e8-bf6a363db82a
# ╟─e30e3f29-d551-496e-ac09-51a2c16ba3e4
# ╟─6ab21d46-69cb-4e60-acbf-97b2daf30e70
# ╟─8ca58440-b7a1-4168-9955-302b02ff6fa2
# ╟─27b739f6-296d-40e7-b93a-85ba5162291b
# ╟─54e063cf-ddc3-4af6-ab1e-5bd50939678e
# ╟─e05e48e9-1ac4-4b19-b64e-1399bcd8e18a
# ╟─54ee56bf-1f34-44f8-8891-5c7ae6962b20
# ╟─20683b43-ea48-4b2e-96d6-e805b0b5e75c
# ╟─1788d576-59d7-40d3-b075-5d244a419a2f
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
