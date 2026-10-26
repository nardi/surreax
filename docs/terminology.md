# Terminology

This page defines the terms that the rest of the documentation uses, with the
mathematics behind each one. The other pages link here at the first use of a
term, and each entry links back to the page that shows it in code.

## Notation

A target takes an input $x \in \mathbb{R}^n$. For an explicit target the output
is $y \in \mathbb{R}^k$, and for an implicit target the unknown is
$y \in \mathbb{R}^m$. A hat marks the surrogate, so $\hat f$ approximates $f$.
The Jacobian of a map is written $\partial f / \partial x$. Inputs and outputs
can be pytrees in the code, and the library flattens them to vectors, which the
[flat coordinates](#flat-coordinates) entry describes. The samples
$x_1, \dots, x_N$ are the points where a fit evaluates the target.

## Targets and surrogates

### Target

The function to approximate. `surreax` never needs its source, only the ability
to call it and to differentiate it with JAX. An explicit target is called
directly, and an implicit target is given as a residual.

### Surrogate

A function $\hat f$ that is cheap to evaluate and close to the target over a
[domain](#domain), with a [form](#form) that you chose in advance. The result of
`fit` is a `FittedSurrogate`. It stores every stage that turns a raw input into
a raw output, so it has the signature of the target.

### Explicit target

A function $f : \mathbb{R}^n \to \mathbb{R}^k$ that you can call. A fit of an
explicit target minimizes the squared difference over the samples,

$$
\min_{\hat f \in \mathcal{F}} \sum_{s=1}^{N} \lVert \hat f(x_s) - f(x_s) \rVert^2,
$$

where $\mathcal{F}$ is the set of functions that the [form](#form) allows. The
[basic workflow](getting-started.md) page shows one. The projectile range is an
example, since the range is a formula in the launch speed and angle.

### Implicit target

A function $f : \mathbb{R}^n \to \mathbb{R}^m$ that has no formula and is only
defined by a [residual](#residual). The residual
$g : \mathbb{R}^n \times \mathbb{R}^m \to \mathbb{R}^m$ is zero exactly when $y$
is the answer for the input $x$,

$$
g(x, f(x)) = 0.
$$

Evaluating $f$ at one input means solving $m$ equations in $m$ unknowns. The
temperature along a heated rod is an example. The input is the heater powers,
the unknown is the temperature at every point, and the residual is the heat
balance at every point. Where $\partial g / \partial y$ is nonsingular, the
implicit function theorem says that $f$ exists locally and gives its derivative,

$$
\frac{\partial f}{\partial x}
  = -\left(\frac{\partial g}{\partial y}\right)^{-1} \frac{\partial g}{\partial x}.
$$

The [implicit targets](implicit.md) page describes how to fit one.

### Explicit and implicit surrogates

The words describe how a surrogate is evaluated, and they do not follow the kind
of target. A surrogate is explicit when evaluating it is a fixed feed-forward
computation $\hat y = \hat f(x)$ with no equation to solve. It is implicit, or
solve-at-call, when evaluating it still solves an equation.

Every fit of an explicit target gives an explicit surrogate. For an implicit
target, `InverseMap`, `ExactSolve`, `Objective`, and the distilled form of
`ReducedSolve` and `AutoencoderSolve` also give explicit surrogates, because
they replace the solve by a regression. The class names can mislead here. An
`ImplicitSurrogate` is the explicit result of `InverseMap`, named for the kind
of target that it approximates.

The solve-at-call objects are `reduced_solver`, `autoencoder_solver`,
`hyper_reduce`, `refine`, `root_find`, and a `CompletedSurrogate`. Each one runs
a few Newton or Gauss-Newton steps at every call. They usually leave a smaller
residual than a regression does, and the reduced, learned-manifold and completed
ones differentiate exactly through the implicit function theorem. They cost a
solve per call, and they have no fixed form that an optimization solver can
encode. An explicit surrogate has a form, and a piecewise linear one
[exports](#export) to arrays.

### Domain

The region $\mathcal{D}$ where the surrogate should be accurate, together with
the distribution that samples are drawn from. A `Box` is
$\mathcal{D} = \prod_j [l_j, u_j]$, and other leaves sample on a logarithmic
scale, from a truncated normal law, over integers, or over a set of options. A
fit never promises accuracy outside the domain. See [the
domain](getting-started.md#the-domain).

### Flat coordinates

The vector in $\mathbb{R}^d$ that a domain maps an input to. A box maps an input
to itself, a `LogRange` to its logarithm, and a `Categorical` to a one-hot
vector with one entry per option. The fit, the reductions, and the
[export](#export) all work in flat coordinates.

### Form

The shape that the surrogate may take, such as affine, a polynomial of a given
degree, or a maximum of affine pieces. A form fixes the set $\mathcal{F}$ in the
fit above. Choosing the form is the main decision in a fit, since it sets how
accurate the result can be and what a downstream tool can do with it. See [model
forms](forms.md).

### Linear form

A form that is linear in its parameters through a feature map $\varphi$,

$$
\hat f(x) = W \varphi(x), \qquad W \in \mathbb{R}^{k \times p}.
$$

The affine form, the Chebyshev polynomial form, and the grid form are linear
forms. The fit is one ridge-regularized least squares solve,

$$
W = \arg\min_W \sum_{s} \lVert W \varphi(x_s) - y_s \rVert^2
  + \lambda \lVert W \rVert_F^2,
$$

with $\lambda = 10^{-8}$ by default and no learning rate. The other forms are
not linear in their parameters and are fitted by iteration.

### Method

The strategy that `fit_implicit` uses, such as `InverseMap` or `ExactSolve`. A
method says how to turn a residual into training data and what to train. The
form is separate. It is the linear form that the method fits to its data, for
example the map from the input to the reduced coordinates.

### Structure cost

The report's count of the parts that the model is built from. It is the number
of nonzero basis terms of a linear form, the number of affine pieces of a
max-affine form, and the parameter count of a network. A larger cost buys
accuracy and makes the surrogate harder to reason about.

## Error measures

All errors are measured on held-out samples that did not take part in the fit.
The [report](getting-started.md#the-report) has one mean and one maximum of
each.

### Value error

The distance between the surrogate and the target at a point,

$$
e_v(x) = \lVert \hat f(x) - f(x) \rVert_2.
$$

For an implicit target the true $f$ is not available, so the report measures the
residual that the surrogate leaves instead,
$e_v(x) = \lVert g(x, \hat f(x)) \rVert_2$.

### Derivative error

The distance between the Jacobians,
$e_d(x) = \lVert \partial \hat f / \partial x - \partial f / \partial x \rVert_F$.
A surrogate can match values well and derivatives badly. A downstream optimizer
uses the derivatives, so the report keeps the two errors apart. Discrete
coordinates are left out.

### Projection error

The accuracy that an output reduction gives up before any fitting. With the
encode and decode maps $E$ and $D$ of the reduction, it is
$e_p(x) = \lVert f(x) - D(E(f(x))) \rVert_2$, the best that any model in the
reduced space could reach.

### Worst-case error

The largest value error over the whole domain,
$\max_{x \in \mathcal{D}} e_v(x)$. A held-out sample can miss the region where
the fit is worst, so `fit` climbs $e_v$ with gradient ascent from several starts
and keeps the largest value that it finds. It is a lower bound on the true
maximum, since the climb can stop at a local maximum. See [worst-case
error](techniques.md#worst-case-error).

### Coverage

A number that says whether an input lies where the fit has evidence. For a box
it is zero inside and one plus the relative excess outside. For a domain that
was [inferred from data](getting-started.md#the-domain) it is at most one near
the data and grows with the distance from it, so a value above one flags a
prediction that extrapolates.

## Fitting techniques

### Derivative supervision

Also called Sobolev training. The fit matches the Jacobian as well as the value,

$$
\min \sum_s \lVert \hat f(x_s) - f(x_s) \rVert^2
  + w \sum_s \Bigl\lVert \frac{\partial \hat f}{\partial x}(x_s)
  - \frac{\partial f}{\partial x}(x_s) \Bigr\rVert_F^2,
$$

with the weight $w$ set by `derivative_weight`. For a linear form the Jacobian
is also linear in $W$, so both terms stack into one least squares solve. See
[derivative supervision](techniques.md#derivative-supervision).

### Sparse basis

A linear form with only a budget of $t$ nonzero columns of $W$. Orthogonal
matching pursuit selects them greedily. Each step adds the feature that is most
correlated with the current residual, refits the active features, and stops
after $t$ terms.

### Principal component analysis

Written PCA. For centered data $A \in \mathbb{R}^{N \times d}$ with the singular
value decomposition $A = U \Sigma V^\top$, the first $r$ rows of $V^\top$ are
the directions of largest variance. The reduction keeps the coordinates
$z = V_r^\top (a - \bar a)$ and decodes by $a \approx \bar a + V_r z$. The rank
$r$ is given or is the fewest directions with
$\sum_{i \le r} \sigma_i^2 \big/ \sum_i \sigma_i^2 \ge$ `variance`.

### Active subspace

The directions in input space along which the target changes. From the Jacobians
$J_s = \partial f / \partial x (x_s)$ it takes the matrix
$C = \frac{1}{N} \sum_s J_s^\top J_s \in \mathbb{R}^{n \times n}$ and keeps its
leading eigenvectors. PCA looks at where the data lies, and the active subspace
looks at where the function varies.

### Pruning

An exact input reduction that drops the inputs on which no output depends. Input
$j$ is dropped when its average sensitivity
$\frac{1}{N}\sum_s \lvert J_{s,kj} \rvert$ is below a small fraction of the
largest sensitivity, for every output $k$.

### Coupling and separable blocks

The coupling matrix has the entries
$C_{kj} = \frac{1}{N} \sum_s \lvert J_{s,kj} \rvert$. Draw an edge between
output $k$ and input $j$ when $C_{kj}$ exceeds a fraction of the largest entry.
The connected components of that graph are the blocks, and a target is separable
when it has more than one. `fit_separable` fits each block on its own inputs.

### Equality constraint and completion

A known equation $h(x, y) = 0$ with $c$ components, such as a mass balance.
Split the outputs into free outputs $y_F$ and $c$ dependent outputs $y_D$. The
surrogate predicts $y_F$ and a starting guess for $y_D$. Completion then solves
$h(x, (y_F, y_D)) = 0$ for $y_D$ at call time by Newton steps, so the equation
holds to solver precision instead of to the fit error. The derivative follows
from the implicit function theorem,

$$
\mathrm{d} y_D = -\left(\frac{\partial h}{\partial y_D}\right)^{-1}
  \left(\frac{\partial h}{\partial x}\, \mathrm{d}x
  + \frac{\partial h}{\partial y_F}\, \mathrm{d}y_F\right).
$$

See [equality constraints](techniques.md#equality-constraints).

## Piecewise linear forms

### Max-affine

The maximum of $p$ affine functions,
$\hat f(x) = \max_{j \le p} (a_j^\top x + b_j)$. It is convex and piecewise
linear, so the constraint $\hat f(x) \le u$ is the set of linear inequalities
$a_j^\top x + b_j \le u$, and a linear program can minimize over it.

### Difference of max-affine

The difference $\hat f = \max_j (a_j^\top x + b_j) - \max_j (c_j^\top x + d_j)$
of two max-affine functions. It can represent any continuous piecewise linear
function, and it carries a split into a convex and a concave part.

### Convex and ReLU networks

A ReLU network alternates affine maps with $\max(\cdot, 0)$ and is piecewise
linear. An input-convex network keeps the weights from one hidden layer to the
next nonnegative and lets every hidden layer read the input,

$$
z_{l+1} = \max\bigl(U_l z_l + W_l x + b_l,\, 0\bigr), \qquad U_l \ge 0,
$$

so every output is convex in $x$. The two networks represent the same functions
as the max-affine form and the difference of max-affine form, and they can do so
with far fewer parts. A solver encodes a ReLU network with one binary variable
per hidden unit.

### Export

Turning a fitted piecewise linear surrogate into plain arrays in the original
coordinates, such as the slopes $a_j$ and intercepts $b_j$ of a max-affine form.
The library stops at the arrays and leaves the program of a solver to you. See
[solver export](forms.md#solver-export).

## Implicit targets

### Residual

The function $g(x, y)$ of an implicit target. It has as many components as the
unknown. The residual left by a surrogate at an input is
$\lVert g(x, \hat f(x)) \rVert$, which is how a fit of an implicit target
reports its error.

### Solution manifold

The set of solutions $\{(x, y) : g(x, y) = 0\}$. The input has few coordinates
and the unknown has many, so the solutions for all inputs form a surface whose
dimension is at most $n$ inside a space of dimension $m$. Every reduced method
exploits that low dimension.

### Branch and fold

A residual can have several solutions $y$ for one input, and each connected
family of them is a branch. A fold is a point where
$\det \partial g / \partial y = 0$ and two branches meet. No smooth function $f$
exists through a fold. The `fold` report of the inverse map keeps one branch,
the one with the sign of the determinant at the sample where its magnitude is
largest.

### Newton step

One iteration toward a root of the residual at a fixed input,

$$
y^{+} = y - \left(\frac{\partial g}{\partial y}\right)^{-1} g(x, y).
$$

Near a root with a nonsingular Jacobian the iteration converges quadratically.
Near a fold it converges only linearly, and far from a root of a stiff residual
it can diverge.

### Inverse map

The method of `InverseMap`. It samples pairs $(x, y)$ and evaluates the residual
forward, $r = g(x, y)$. The map $(x, y) \mapsto (x, r)$ is then inverted by a
regression $y = h(x, r)$ over the augmented input $(x, r)$, and the surrogate
reads the slice with zero residual, $\hat f(x) = h(x, 0)$. It needs no root
solve beyond a few Newton steps to bring the samples close to the manifold. See
[the inverse map](implicit.md#the-inverse-map).

### Objective route

The method of `Objective`. It fits $\hat f(x) = W \varphi(x)$ directly by
minimizing the residual that it leaves,

$$
\min_W \frac{1}{N} \sum_s \lVert g(x_s, W \varphi(x_s)) \rVert^2.
$$

It is experimental, since the objective does not prefer one branch over another.

### Exact solve

The method of `ExactSolve`. It solves the full residual at every sample, with no
model reduction inside the solve, and then reduces the solved states for the
fit. The steps are these.

1. For each sample $x_s$, run `newton_steps` Newton steps from a starting state
   drawn from `unknown_domain`. This solves the $m$ equations in $m$ unknowns
   for the full state, which is what the name refers to. A solve has converged
   when $\lVert g(x_s, y_s) \rVert \le \tau \lVert g(x_s, 0) \rVert$ with $\tau$
   = `convergence_tolerance`, and the fit raises an error otherwise.
2. Compute the PCA of the solved states, $y \approx c + V^\top z$ with
   $V \in \mathbb{R}^{r \times m}$.
3. Fit the map from the input to the reduced coordinates,
   $\hat z(x) = W \varphi(x)$, with a linear form.

The surrogate is $\hat y(x) = c + V^\top \hat z(x)$. The name says that the
training states are exact solutions, and it does not say that the surrogate is
exact. Its errors are the truncation of the PCA and the regression error of
$\hat z$. The cost is one full Newton solve per sample, which the sparse
Jacobian makes affordable for a large unknown. The alternative is the [reduced
solve](#reduced-solve), which spends far fewer full solves.

### Snapshot

A solved full state $y_s$ at a training input $x_s$. The snapshot matrix has one
row per snapshot.

### POD basis

The proper orthogonal decomposition of the snapshots is the PCA of the snapshot
matrix. It gives a center $c$ and an orthonormal basis, stored as the rows of
$V \in \mathbb{R}^{r \times m}$. The reduced state is the latent
$z \in \mathbb{R}^r$, and the decoded state is $y = c + V^\top z$.

### Latent

The small vector $z \in \mathbb{R}^r$ of reduced coordinates, whose dimension is
the `rank`. A latent is the coordinates of a state on the reduced basis for a
linear method and on the learned manifold for an autoencoder.

### Reduced solve

The method of `ReducedSolve`. It builds a [POD basis](#pod-basis) from the
snapshots, projects the residual onto the basis, and solves the projected system
for the latent. This is a Galerkin projection,

$$
V\, g\bigl(x,\, c + V^\top z\bigr) = 0 \in \mathbb{R}^r.
$$

Each Newton step is an $r \times r$ solve whatever the size of $m$. The solver
reports where the reduced Jacobian is badly conditioned and whether a second
start finds a different latent root. The method then distills the map from the
input to the latent into a regression, so the final surrogate needs no solve.
The choice of snapshots is the first `snapshots` samples, or a greedy choice
that adds a snapshot where the current basis leaves the most residual.

### Kolmogorov n-width

How well $n$ linear modes can approximate the set $\mathcal{M}$ of all
solutions,

$$
d_n(\mathcal{M}) = \inf_{\dim V_n = n} \; \sup_{u \in \mathcal{M}} \;
  \inf_{v \in V_n} \lVert u - v \rVert.
$$

A linear [reduced solve](#reduced-solve) works well when $d_n$ decays fast. A
sharp feature that moves with the input makes it decay slowly.

### Linear energy

The fraction of the variation of the snapshots that $r$ linear modes capture,
$\sum_{i \le r} \sigma_i^2 \big/ \sum_i \sigma_i^2$ for the singular values of
the centered snapshot matrix. It estimates the n-width from data, and
`recommend_reduced_method` reads it to choose between a linear basis and an
autoencoder.

### Autoencoder solve

The method of `AutoencoderSolve`. A decoder $D$ maps the latent to the full
state onto a learned manifold, $D(z) = c + Wz + N(z)$ with a small network $N$,
and an encoder $E$ maps back. The latent solve minimizes the residual of the
decoded state by Gauss-Newton,

$$
z^\star(x) = \arg\min_z \lVert g(x, D(z)) \rVert^2.
$$

This is the least-squares Petrov-Galerkin projection. Training lowers the
residual at $z^\star(x)$ over sampled inputs. It also trains $E$ so that
$E(D(z)) = z$, and a small network, the `LatentGuess`, that predicts
$z^\star(x)$ to start each solve. With `snapshots` at zero it only evaluates $g$
forward and never solves a full state. A positive count solves that many full
states and pretrains the decoder on them.

### Hyper-reduction

A reduced solve still evaluates the whole residual at each step. Hyper-reduction
evaluates only $s$ chosen entries $P$ of it. It builds a residual basis
$U \in \mathbb{R}^{m \times q}$ from residual snapshots, picks $P$ by the
discrete empirical interpolation method, and approximates the residual by
$g \approx U\, (U_P)^{+} g_P$ with $U_P$ the rows of $U$ in $P$. The latent
solve becomes

$$
\min_z \bigl\lVert (U_P)^{+}\, g_P\bigl(x, D(z)\bigr) \bigr\rVert^2.
$$

It needs a [stencil residual](#stencil-residual), so that evaluating an entry
reads only a few state entries.

### Stencil residual

A residual written entry by entry. Entry $i$ is a function of the input, the
index, and the unknown at the neighbors of $i$,

$$
g_i(x, y) = e\bigl(x,\, i,\, y_{\mathcal{N}(i)}\bigr).
$$

The table $\mathcal{N}$ of neighbor indices is the stencil. It makes the
Jacobian sparse, with entry $(i, j)$ nonzero only for $j \in \mathcal{N}(i)$,
and it lets hyper-reduction evaluate single entries. In the code it is a
`StencilResidual`.

### Linear solver

How a Newton step solves $J\, \delta = g$ with $J = \partial g / \partial y$.
`"dense"` forms $J$ and factors it, which suits a small unknown. `"sparse"`
reads the sparsity pattern of $J$, colors it so that $J$ costs one
Jacobian-vector product per color, and factors it with a sparse direct solver.
`"matrix_free"` runs GMRES on Jacobian-vector products and never forms $J$. The
sparse choice falls back to the matrix-free one when the residual has no usable
pattern.

### Factorization reuse

The `reuse` setting keeps the factorization of an earlier matrix
$J_{\text{old}}$ as the preconditioner of an iterative solve with a new matrix
$J$, and factors $J$ again only when that solve is slow. It saves time when a
factorization costs much more than a back-substitution.

### Refinement

Polishing a prediction with $K$ Newton steps at call time,
$\hat y_K = \mathcal{N}^K(\hat f(x))$ with $\mathcal{N}$ the Newton step. It
turns an explicit surrogate into a solve-at-call one with a smaller residual.

### Continuation

Following one branch through a fold. The samples are ordered along the first
input coordinate and each Newton solve starts from the previous solution.
