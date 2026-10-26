# Test systems

The test suite fits a small set of physical systems, and this page describes
each one. It gives the equations, the shape of the problem, the parts of
`surreax` that the system exercises, and links to the code that builds it and to
the tests that use it. Reading it should tell you which system to pick when you
want to try a feature on a realistic problem.

The builders live in the `tests` directory of the
[repository](https://github.com/nardi/surreax/tree/main/tests), and the package
on PyPI does not ship them. To use one, clone the repository and import it from
the top level, for example `from tests import implicit_systems`. The links on
this page point at the `main` branch.

## The two kinds of system

An explicit target is a function that the library can call, and the builders in
[`projectile_range`](https://github.com/nardi/surreax/blob/main/tests/systems.py#L19)
and its neighbors in `tests/systems.py` are of this kind. Each is a pure JAX
function of an input with a hand-derived form, so a test can compare a surrogate
against the exact function and its exact Jacobian.

An implicit system is a residual instead. The input is $x \in \mathbb{R}^n$, the
unknown is $y \in \mathbb{R}^m$, and the residual
$g : \mathbb{R}^n \times \mathbb{R}^m \to \mathbb{R}^m$ is zero when $y$ solves
the model for that input. The surrogate approximates the map $f$ with
$g(x, f(x)) = 0$. The [implicit targets](implicit.md) page describes the methods
that fit it. Every builder in
[`heated_rod`](https://github.com/nardi/surreax/blob/main/tests/implicit_systems.py#L53)
and the functions around it returns an
[`ImplicitSystem`](https://github.com/nardi/surreax/blob/main/tests/implicit_systems.py#L27),
a named tuple of three fields.

- `residual` is a `StencilResidual`, which writes entry $i$ of the residual as a
  function of the input, the index $i$, and the unknown at a few neighbors of
  node $i$. The table of neighbor indices is the stencil. It makes the Jacobian
  $\partial g / \partial y$ sparse with a known pattern, and it lets
  hyper-reduction evaluate single entries.
- `domain` is the `Box` of inputs that a fit samples.
- `unknown_domain` is the `Box` that the Newton solves draw their starting
  states from.

Every implicit system has a few inputs and many unknowns. The solution lies on a
manifold whose dimension is at most the number of inputs, whatever the size of
the state, and the systems differ in how that manifold looks. It is flat for a
linear residual, curved for a nonlinear one, and it can bend sharply, fold, or
split into several branches.

## Overview

| System | Kind | Inputs | Unknowns or outputs | Main use |
| --- | --- | --- | --- | --- |
| [Projectile range](#projectile-range) | Explicit | 2 | 1 | Basic fit and report |
| [Damped oscillator](#damped-oscillator) | Explicit | 3, a pytree | 1 | Structured inputs |
| [RLC circuit](#rlc-circuit) | Explicit | 3, a pytree | 6 | Vector outputs and output reduction |
| [Few active inputs](#few-active-inputs) | Explicit | 12 | 1 | Pruning, active subspaces, sparse bases |
| [Diffusion with sources](#diffusion-with-sources) | Implicit | 4 | 4096 | Scaling of the solvers |
| [Heated rod](#heated-rod) | Implicit | 4 | any | Linear residual with an exact flat manifold |
| [Radiating rod](#radiating-rod) | Implicit | 4 | any | Curved manifold |
| [Thermostat rod](#thermostat-rod) | Implicit | 2 | any | Response that is neither convex nor concave |
| [Radiating plate and block](#radiating-plate-and-block) | Implicit | 4 | any | Fill-in of sparse factorizations |
| [Rod with hot spots](#rod-with-hot-spots) | Implicit | 4 | any | Hyper-reduction |
| [Hot fluid pipe](#hot-fluid-pipe) | Implicit | 2 | any | Sharp moving front |
| [Buckling beam](#buckling-beam) | Implicit | 2 | 24 | Bifurcation and a singular Jacobian |
| [Resistor network](#resistor-network) | Implicit | 4 | any | Sparse graph, not banded |
| [Diode network](#diode-network) | Implicit | 4 | any | Stiff exponentials |
| [Latch ring](#latch-ring) | Implicit | 2 | any | Several solutions |

The input counts of the rod, plate, block and network systems are the defaults,
since their builders take the number of heaters or sources as an argument.

## Explicit targets

### Projectile range

The horizontal range of a projectile launched at speed $v$ and angle $\theta$ is

$$
R(v, \theta) = \frac{v^2 \sin 2\theta}{g}, \qquad g = 9.81 \ \mathrm{m/s^2}.
$$

The input is a flat array of length two, with $v \in [5, 20]$ m/s and
$\theta \in [0.2, 1.3]$ rad, and the output has one entry. It is the smallest
realistic target, smooth and a product of a quadratic and a sine. The [basic
workflow](getting-started.md) page fits it. The builders are
[`projectile_range`](https://github.com/nardi/surreax/blob/main/tests/systems.py#L19)
and
[`projectile_domain`](https://github.com/nardi/surreax/blob/main/tests/systems.py#L25).

The tests fit it with a degree-five polynomial, compare the surrogate against
the exact function under `jit`, `vmap` and differentiation, and check that
derivative supervision lowers the derivative error. The benchmarks time its fit.

- [`test_integration_on_physical_systems`](https://github.com/nardi/surreax/blob/main/tests/integration/test_fit.py#L60)
- [`test_surrogate_agrees_across_jit_vmap_and_derivatives`](https://github.com/nardi/surreax/blob/main/tests/integration/test_fit.py#L76)
- [`test_derivative_supervision_lowers_derivative_error`](https://github.com/nardi/surreax/blob/main/tests/integration/test_fit.py#L130)

### Damped oscillator

The displacement of a damped harmonic oscillator at the fixed time $t = 1$ is

$$
x = e^{-\delta t} \cos(\omega t), \qquad
\delta = \frac{c}{2m}, \qquad
\omega = \sqrt{\frac{k}{m} - \delta^2},
$$

with mass $m \in [1, 2]$, stiffness $k \in [8, 12]$ and damping
$c \in [0.1, 1]$. The box stays in the underdamped region, so $\omega$ is real
everywhere.

<figure class="schematic">
<svg viewBox="0 0 400 140" width="400" role="img" aria-label="A mass attached to a wall by a spring and a damper in parallel">
<path class="line" d="M30 15V125"/>
<path class="line thin" d="M30 25l-10 10M30 45l-10 10M30 65l-10 10M30 85l-10 10M30 105l-10 10"/>
<path class="line" d="M30 40H60l10-14 16 28 16-28 16 28 16-28 16 28 10-14H200"/>
<path class="line" d="M30 95H80M80 83H130M80 107H130M80 83V107M112 87V103M112 95H200"/>
<rect class="soft" x="200" y="25" width="70" height="95"/>
<rect class="line" x="200" y="25" width="70" height="95"/>
<path class="line thin" d="M270 132H330M322 127l8 5-8 5"/>
<text x="235" y="78" text-anchor="middle">m</text>
<text x="115" y="18" text-anchor="middle">k</text>
<text x="105" y="128" text-anchor="middle">c</text>
<text x="345" y="136">x(t)</text>
</svg>
<figcaption>A mass on a spring and a damper in parallel.</figcaption>
</figure>

The input is not an array. It is an `equinox.Module` with the named fields
`mass`, `stiffness` and `damping`, and the `Box` holds the bounds as the same
kind of module. The system shows that the target, the domain, the surrogate and
its derivatives all work on a pytree input. The builders are
[`OscillatorState`](https://github.com/nardi/surreax/blob/main/tests/systems.py#L30),
[`oscillator_displacement`](https://github.com/nardi/surreax/blob/main/tests/systems.py#L42)
and
[`oscillator_domain`](https://github.com/nardi/surreax/blob/main/tests/systems.py#L50).

- [`test_pytree_input_surrogate_agrees_across_jit_vmap_and_derivatives`](https://github.com/nardi/surreax/blob/main/tests/integration/test_fit.py#L182)
- [`test_box_unravels_a_pytree_bound_round_trip`](https://github.com/nardi/surreax/blob/main/tests/integration/test_fit.py#L172)

### RLC circuit

A series resistor, inductor and capacitor with the output taken across the
capacitor has the low-pass response

$$
|H(\omega)| = \frac{1}{\sqrt{(1 - \omega^2 L C)^2 + (\omega R C)^2}}.
$$

The target evaluates it at six frequencies $\omega_k = 0.5, 0.8, \dots, 2$, so
the output is a vector of length six. The three component values are a pytree of
scalars in $[0.5, 2]$.

<figure class="schematic">
<svg viewBox="0 0 420 150" width="420" role="img" aria-label="A series resistor and inductor driving a capacitor, with the output taken across the capacitor">
<circle class="node" cx="50" cy="70" r="16"/>
<path class="line thin" d="M42 70q4-8 8 0t8 0"/>
<path class="line" d="M50 54V30H90M130 30H160M224 30H300V60M300 76V115H50V86"/>
<rect class="line" x="90" y="22" width="40" height="16"/>
<path class="line" d="M160 30a8 8 0 0 1 16 0a8 8 0 0 1 16 0a8 8 0 0 1 16 0a8 8 0 0 1 16 0"/>
<path class="line" d="M284 60H316M284 76H316"/>
<circle class="solid" cx="300" cy="30" r="3"/>
<circle class="solid" cx="300" cy="115" r="3"/>
<path class="line thin dash" d="M300 30H350M300 115H350"/>
<path class="line thin" d="M345 36V109M340 44l5-8 5 8M340 101l5 8 5-8"/>
<text x="110" y="16" text-anchor="middle">R</text>
<text x="192" y="16" text-anchor="middle">L</text>
<text x="270" y="72" text-anchor="middle">C</text>
<text x="20" y="74" text-anchor="middle">in</text>
<text x="360" y="76">out</text>
</svg>
<figcaption>The series RLC circuit, with the output across the capacitor.</figcaption>
</figure>

The six outputs vary together, since each is a smooth function of the same three
components, which makes the target a case for output reduction. The builders are
[`RlcComponents`](https://github.com/nardi/surreax/blob/main/tests/systems.py#L57),
[`rlc_response`](https://github.com/nardi/surreax/blob/main/tests/systems.py#L68)
and
[`rlc_domain`](https://github.com/nardi/surreax/blob/main/tests/systems.py#L76).

- [`test_output_reduction_fits_a_vector_target`](https://github.com/nardi/surreax/blob/main/tests/integration/test_fit.py#L141)
- [`test_integration_on_physical_systems`](https://github.com/nardi/surreax/blob/main/tests/integration/test_fit.py#L60)

### Few active inputs

A target in twelve inputs that depends on only two of them,

$$
f(x) = x_3^2 + 1.5\, x_8 - x_3 x_8, \qquad x \in [-1, 1]^{12},
$$

with the inputs numbered from one here and from zero in the code. The Jacobian
shows that ten inputs have no effect, and the function varies along only two
directions. It exercises exact pruning, active-subspace reduction, and the
sparse basis selection, where a total-degree basis in twelve inputs has far more
terms than a small sample can support. The builders are
[`few_active_inputs`](https://github.com/nardi/surreax/blob/main/tests/systems.py#L90)
and
[`high_dimensional_domain`](https://github.com/nardi/surreax/blob/main/tests/systems.py#L95).

- [`test_matching_pursuit_recovers_the_active_support`](https://github.com/nardi/surreax/blob/main/tests/integration/test_sparse.py#L11)
- [`test_sparse_fit_is_accurate_with_a_small_structure_cost`](https://github.com/nardi/surreax/blob/main/tests/integration/test_sparse.py#L22)
- [`test_sparse_beats_dense_when_samples_are_fewer_than_the_basis`](https://github.com/nardi/surreax/blob/main/tests/integration/test_sparse.py#L36)

### Diffusion with sources

The one implicit system in `tests/systems.py` is a large one. A field $y$ on
$m = 4096$ grid points is driven by four source amplitudes $x_1, \dots, x_4$,

$$
g_i(x, y) = -(y_{i-1} - 2 y_i + y_{i+1}) + \tfrac{1}{2} y_i
  - \sum_{k=1}^{4} x_k \psi_k(s_i),
$$

with periodic neighbors and the four smooth modes
$\psi = \sin 2\pi s, \cos 2\pi s, \sin 4\pi s, \cos 4\pi s$ on $s \in [0, 1]$.
The Jacobian in $y$ has three nonzeros per row, so a sparse factorization costs
far less than a dense one. The field lies in the span of four modes however fine
the grid is, so the output reduction of `ExactSolve` and the latent space of
`ReducedSolve` both need only four coordinates. This is the case that the sparse
solve and the output reduction are built for.

The builders are
[`large_sparse_residual`](https://github.com/nardi/surreax/blob/main/tests/systems.py#L123),
[`large_sparse_domain`](https://github.com/nardi/surreax/blob/main/tests/systems.py#L139)
and
[`large_sparse_unknown_domain`](https://github.com/nardi/surreax/blob/main/tests/systems.py#L144).

- [`test_exact_solve_scales_to_a_large_field`](https://github.com/nardi/surreax/blob/main/tests/integration/test_scaling.py#L45)
- [`test_reduced_solve_scales_to_a_large_field`](https://github.com/nardi/surreax/blob/main/tests/integration/test_scaling.py#L24)
- [`test_reduced_solver_call_is_cheap_at_scale`](https://github.com/nardi/surreax/blob/main/tests/integration/test_scaling.py#L63)
- [`test_newton_solve_reaches_a_field_a_dense_jacobian_cannot_hold`](https://github.com/nardi/surreax/blob/main/tests/integration/test_scaling.py#L86)

## Implicit systems

The systems below live in `tests/implicit_systems.py`. The equations number the
points and the inputs from one, and the code numbers them from zero. Each
builder takes the number of points or nodes, so a test can run a system small
for correctness and large for scaling. The chain systems use unscaled
differences, so the conduction operator keeps a bounded condition number and the
matrix-free solve converges in a few iterations at any size.

Two tests check every system together. One solves each from zero over the input
domain and checks that the Jacobian reads the unknown only where the stencil
says. The other checks that a stencil entry evaluated on its own equals the
matching entry of the full residual.

- [`test_system_solves_and_stays_within_its_stencil`](https://github.com/nardi/surreax/blob/main/tests/integration/test_implicit_systems.py#L42)
- [`test_stencil_entries_match_the_full_residual`](https://github.com/nardi/surreax/blob/main/tests/integration/test_implicit_systems.py#L56)

### Heated rod

A rod of $m$ points has its ends held at ambient temperature, which is zero
here, and is heated by $k$ heaters. Each heater $j$ deposits heat along the rod
with the profile $\phi_j(i) = \sin(j \pi i / (m + 1))$, and its power
$x_j \in [0, 1]$ is an input. With $y_0 = y_{m+1} = 0$ the residual is

$$
g_i(x, y) = -(y_{i-1} - 2 y_i + y_{i+1}) + \tfrac{1}{2} y_i
  - \sum_{j=1}^{k} x_j \phi_j(i),
$$

where the term $\tfrac12 y_i$ is heat lost to the air.

<figure class="schematic">
<svg viewBox="0 0 580 160" width="580" role="img" aria-label="A chain of nodes between two ambient ends, with a heater profile above it and the three nodes that one residual entry reads">
<path class="soft" d="M40 70Q290 -20 540 70Z"/>
<path class="line thin" d="M40 70Q290 -20 540 70"/>
<path class="line thin" d="M240 36V76M310 28V76M380 36V76M236 68l4 8 4-8M306 68l4 8 4-8M376 68l4 8 4-8"/>
<path class="line" d="M40 100H140M210 100H400M450 100H540"/>
<path class="line thin dash" d="M140 100H210M400 100H450"/>
<rect class="node" x="32" y="92" width="16" height="16"/>
<rect class="node" x="532" y="92" width="16" height="16"/>
<circle class="node" cx="110" cy="100" r="7"/>
<circle class="node" cx="470" cy="100" r="7"/>
<circle class="accent-node" cx="240" cy="100" r="7"/>
<circle class="accent-node" cx="310" cy="100" r="7"/>
<circle class="accent-node" cx="380" cy="100" r="7"/>
<path class="accent" d="M225 120V128H395V120"/>
<text x="40" y="140" text-anchor="middle" class="small">ambient</text>
<text x="540" y="140" text-anchor="middle" class="small">ambient</text>
<text x="110" y="85" text-anchor="middle">y₁</text>
<text x="240" y="85" text-anchor="middle">y<tspan class="small" dy="4">i−1</tspan></text>
<text x="310" y="85" text-anchor="middle">y<tspan class="small" dy="4">i</tspan></text>
<text x="380" y="85" text-anchor="middle">y<tspan class="small" dy="4">i+1</tspan></text>
<text x="470" y="85" text-anchor="middle">y<tspan class="small" dy="4">m</tspan></text>
<text x="310" y="145" text-anchor="middle" class="accent-text">entry i of the residual</text>
<text x="290" y="18" text-anchor="middle" class="small">heat deposited by the heaters</text>
</svg>
<figcaption>A chain of nodes between two ambient ends. Entry i of the residual reads the three highlighted nodes.</figcaption>
</figure>

The residual is affine in $y$ and in $x$, so the temperature is an exact linear
map of the heater powers, $y = A^{-1} \Phi^\top x$, where $A$ is a fixed
tridiagonal matrix and the rows of $\Phi$ hold the heater profiles. The
solutions fill a linear subspace of dimension $k$ and nothing else. This gives
the cleanest checks for the implicit methods. An `Affine` form recovers the map
exactly, and a `ReducedSolve` leaves a residual of zero once its rank reaches
$k$ and a large one just below it.

The builder is
[`heated_rod`](https://github.com/nardi/surreax/blob/main/tests/implicit_systems.py#L53).

- [`test_linear_systems_have_exact_low_rank_manifolds`](https://github.com/nardi/surreax/blob/main/tests/integration/test_implicit_systems.py#L65)
- [`test_heated_rod_residual_drops_at_the_number_of_heaters`](https://github.com/nardi/surreax/blob/main/tests/integration/test_reduced_order_systems.py#L67)
- [`test_recommender_routes_by_the_linear_energy`](https://github.com/nardi/surreax/blob/main/tests/integration/test_autoencoder.py#L148)

### Radiating rod

The same rod also loses heat by radiation along its whole length,

$$
g_i(x, y) = -(y_{i-1} - 2 y_i + y_{i+1}) + \tfrac{1}{2} y_i
  + y_i \lvert y_i \rvert^3 - \sum_j x_j \phi_j(i).
$$

The fourth power is written with a sign, so a point colder than ambient gains
heat. The Jacobian now depends on the state, so the temperature is a curved
function of the heater powers and the manifold bends out of a $k$-dimensional
subspace. With more power the rod radiates more, which makes the temperature
concave in the heater powers. That property is what the optimization test uses.

The system is the standard nonlinear example. It shows that a linear reduced
basis needs more than $k$ modes, it drives the piecewise linear fits through the
latent coefficients of its reduced solver, and at 4096 points it is one of the
systems that the Newton benchmarks time. The builder is
[`radiating_rod`](https://github.com/nardi/surreax/blob/main/tests/implicit_systems.py#L77).

- [`test_curved_and_moving_manifolds_need_more_modes`](https://github.com/nardi/surreax/blob/main/tests/integration/test_implicit_systems.py#L72)
- [`test_a_difference_of_max_affine_approximates_the_system_better_than_an_affine_fit`](https://github.com/nardi/surreax/blob/main/tests/integration/test_piecewise_linear_systems.py#L21)
- [`test_exact_solve_derivative_supervision_lowers_the_jacobian_error`](https://github.com/nardi/surreax/blob/main/tests/integration/test_sparse_solve.py#L186)
- [`test_the_least_heater_power_plan_of_a_convex_surrogate_holds_on_the_rod`](https://github.com/nardi/surreax/blob/main/tests/integration/test_optimization_export.py#L71)

### Thermostat rod

Every point of this rod has a thermostat-controlled heater, and the rod sits in
an ambient temperature $a = x_1$ that varies. A load $q = x_2$ heats a patch of
the rod. With the ends at the ambient temperature the residual is

$$
g_i(x, y) = -(y_{i-1} - 2 y_i + y_{i+1}) + \tfrac12 (y_i - a)
  - P\, s\!\left(\frac{T - y_i}{w}\right) - q\, \varphi_i,
$$

where $s$ is the logistic function, $T = 0.5$ is the setpoint, $w = 0.2$ is the
width over which a heater switches, and $P = 1$ is the heat of a fully on
heater. The patch is $\varphi_i = \exp(-((p_i - 0.7)/0.1)^2)$ at the position
$p_i$ of point $i$. The ambient ranges over $[-2.5, 2]$ and the load over
$[0, 1]$.

<figure class="schematic">
<svg viewBox="0 0 460 250" width="460" role="img" aria-label="Rod temperature against ambient temperature, rising with slope one, then flat near the setpoint, then rising with slope one again">
<path class="line thin" d="M60 220H430M60 220V20"/>
<path class="line thin dash" d="M220 220L420 30"/>
<path class="line thin dash" d="M60 138H430"/>
<path class="accent" d="M60 210L132 145Q140 138 150 138H290Q300 138 308 130L420 30"/>
<text x="430" y="242" text-anchor="end">ambient temperature a</text>
<text x="66" y="16" class="small">rod temperature y</text>
<text x="436" y="134" text-anchor="end" class="small">setpoint</text>
<text x="95" y="212" class="small accent-text">saturated, y = a + 2P</text>
<text x="220" y="128" text-anchor="middle" class="small accent-text">held near the setpoint</text>
<text x="430" y="170" text-anchor="end" class="small accent-text">heaters off, y = a</text>
</svg>
<figcaption>The rod temperature against the ambient temperature.</figcaption>
</figure>

A warm ambient leaves the heaters off, and the rod follows the ambient with
slope one. As the ambient cools, the heaters ramp up and hold the rod near the
setpoint, so the slope drops to about zero. Once the heaters saturate, the rod
follows the ambient again, about $2P$ warmer. The response therefore has slope
one, then zero, then one, which is concave at the first kink and convex at the
second. The Jacobian is positive definite, because the heaters push back against
a temperature change, so the solution is unique.

This is the system for the forms that can represent a function that is neither
convex nor concave. A difference of max-affine fits it, a ReLU network fits it
better, and a convex network cannot follow it. The optimization test builds a
mixed-integer program on a ReLU network fitted to it. The builder is
[`thermostat_rod`](https://github.com/nardi/surreax/blob/main/tests/implicit_systems.py#L111).

- [`test_thermostat_rod_response_is_neither_convex_nor_concave`](https://github.com/nardi/surreax/blob/main/tests/integration/test_implicit_systems.py#L129)
- [`test_a_convex_network_cannot_follow_a_response_that_is_not_convex`](https://github.com/nardi/surreax/blob/main/tests/integration/test_network_forms_comparison.py#L72)
- [`test_the_mixed_integer_plan_of_a_relu_network_holds_on_the_thermostat_rod`](https://github.com/nardi/surreax/blob/main/tests/integration/test_optimization_export.py#L136)

### Radiating plate and block

The radiating rod on a two-dimensional $s \times s$ grid is a plate, and on a
three-dimensional $s \times s \times s$ grid it is a block. Each point exchanges
heat with its four or six grid neighbors. With $\mathcal{N}(i)$ for the
neighbors of point $i$ that lie inside the grid, the residual of the plate is

$$
g_i(x, y) = 4 y_i - \sum_{j \in \mathcal{N}(i)} y_j + \tfrac12 y_i
  + y_i \lvert y_i \rvert^3 - \sum_k x_k \phi_k(i),
$$

and the block has $6 y_i$ in place of $4 y_i$.

<figure class="schematic">
<svg viewBox="0 0 520 190" width="520" role="img" aria-label="The five-point stencil on a square grid and the seven-point stencil on a cubic grid">
<path class="line thin" d="M30 20H142M30 48H142M30 76H142M30 104H142M30 132H142M30 20V132M58 20V132M86 20V132M114 20V132M142 20V132"/>
<path class="accent" d="M58 76H114M86 48V104"/>
<circle class="accent-node" cx="86" cy="76" r="6"/>
<circle class="accent-node" cx="58" cy="76" r="4.5"/>
<circle class="accent-node" cx="114" cy="76" r="4.5"/>
<circle class="accent-node" cx="86" cy="48" r="4.5"/>
<circle class="accent-node" cx="86" cy="104" r="4.5"/>
<text x="86" y="160" text-anchor="middle">plate, 5 per row</text>
<path class="accent" d="M320 90H420M370 40V140M345 112L395 68"/>
<circle class="accent-node" cx="370" cy="90" r="6"/>
<circle class="accent-node" cx="320" cy="90" r="4.5"/>
<circle class="accent-node" cx="420" cy="90" r="4.5"/>
<circle class="accent-node" cx="370" cy="40" r="4.5"/>
<circle class="accent-node" cx="370" cy="140" r="4.5"/>
<circle class="accent-node" cx="345" cy="112" r="4.5"/>
<circle class="accent-node" cx="395" cy="68" r="4.5"/>
<text x="370" y="170" text-anchor="middle">block, 7 per row</text>
</svg>
<figcaption>The stencils of the plate and the block.</figcaption>
</figure>

The Jacobian has five nonzeros per row for the plate and seven for the block. A
chain has a banded Jacobian, but a grid does not. The factorization of a grid
Jacobian fills in between the rows of the grid, and a 3D grid fills in far more
than a 2D one. That makes these two systems the test for the question of when
reusing a stale factorization through `HybridSettings` is cheaper than factoring
again, which the [implicit targets](implicit.md#precision) page reports. The
builders are
[`radiating_plate`](https://github.com/nardi/surreax/blob/main/tests/implicit_systems.py#L146)
and
[`radiating_block`](https://github.com/nardi/surreax/blob/main/tests/implicit_systems.py#L180).

- [`test_stencil_pattern_covers_the_traced_pattern`](https://github.com/nardi/surreax/blob/main/tests/integration/test_sparse_solve.py#L67)
- [`test_benchmark_newton_solve_against_linear_solver`](https://github.com/nardi/surreax/blob/main/tests/benchmarks/test_hot_paths.py#L170)

### Rod with hot spots

An insulated rod radiates only at three bare patches. The radiation term of the
radiating rod gets a mask $m_i$ that is nonzero at only a few points,

$$
g_i(x, y) = -(y_{i-1} - 2 y_i + y_{i+1}) + \tfrac12 y_i
  + 3 m_i\, y_i \lvert y_i \rvert^3 - \sum_j x_j \phi_j(i).
$$

The mask is a sum of three narrow Gaussian bumps at $0.25$, $0.5$ and $0.75$ of
the rod, cut to zero where it falls below $10^{-3}$. Its width is a fixed
fraction of the rod, so the patches cover about the same number of points
however fine the rod is. The nonlinear part of the residual lives on a small set
of entries.

That is the case for hyper-reduction. The entries that the discrete empirical
interpolation method chooses gather on the patches, and the cost of the latent
solve stays flat as the state grows. The builder is
[`rod_with_hot_spots`](https://github.com/nardi/surreax/blob/main/tests/implicit_systems.py#L221).

- [`test_hot_spot_nonlinearity_stays_on_the_patches`](https://github.com/nardi/surreax/blob/main/tests/integration/test_implicit_systems.py#L81)
- [`test_samples_land_on_the_radiating_patches`](https://github.com/nardi/surreax/blob/main/tests/integration/test_hyperreduction.py#L70)
- [`test_hyper_reduced_solve_reads_a_fixed_number_of_rows`](https://github.com/nardi/surreax/blob/main/tests/integration/test_hyperreduction.py#L50)
- [`test_benchmark_latent_solve_against_state_size`](https://github.com/nardi/surreax/blob/main/tests/benchmarks/test_hot_paths.py#L191)

### Hot fluid pipe

Fluid flows through a pipe past a narrow heater at a movable position. With the
heater position $x_1$ and the flow rate $x_2$, the steady balance of advection,
diffusion and a little wall cooling is

$$
g_i(x, y) = v\,(y_i - y_{i-1}) - 0.05\,(y_{i-1} - 2 y_i + y_{i+1})
  + 0.01\, y_i - q_i, \qquad v = 1 + x_2,
$$

with cold inflow, $y_0 = 0$, and no gradient at the outflow. The heat deposit is
$q_i = \exp(-((i - s)/1.5)^2)$ centered at $s = m/8 + x_1 (m - m/4)$.

<figure class="schematic">
<svg viewBox="0 0 480 210" width="480" role="img" aria-label="Pipe temperature against position for two heater positions, a sharp step that moves with the heater">
<path class="line thin" d="M50 180H450M50 180V25"/>
<path class="line thin dash" d="M170 180V60M310 180V60"/>
<path class="line" d="M50 170H160C167 170 167 80 175 78L450 98"/>
<path class="accent" d="M50 170H300C307 170 307 80 315 78L450 88"/>
<path class="line thin" d="M60 30H110M102 25l8 5-8 5"/>
<text x="120" y="34" class="small">flow</text>
<text x="170" y="194" text-anchor="middle" class="small">heater</text>
<text x="310" y="194" text-anchor="middle" class="small">heater</text>
<text x="450" y="208" text-anchor="end" class="small">position along the pipe</text>
<text x="56" y="20" class="small">temperature</text>
</svg>
<figcaption>The pipe temperature for two heater positions.</figcaption>
</figure>

The fluid is cold upstream of the heater and warm after it, so the solution is a
sharp step that slides along the pipe with the heater. The manifold has two
dimensions, but a linear basis needs about one mode per step position. This is
the hard case for a linear reduced basis and the case the learned manifold
exists for. A rank-two autoencoder leaves about a third of the residual that
eight linear modes leave. The builder is
[`hot_fluid_pipe`](https://github.com/nardi/surreax/blob/main/tests/implicit_systems.py#L249).

- [`test_learned_manifold_beats_a_linear_basis_on_a_moving_front`](https://github.com/nardi/surreax/blob/main/tests/integration/test_autoencoder.py#L109)
- [`test_recommender_routes_by_the_linear_energy`](https://github.com/nardi/surreax/blob/main/tests/integration/test_autoencoder.py#L148)
- [`test_curved_and_moving_manifolds_need_more_modes`](https://github.com/nardi/surreax/blob/main/tests/integration/test_implicit_systems.py#L72)

### Buckling beam

A column pressed from its ends buckles past a critical load. The deflection is
$y$, the load $x_1 \in [0, 2]$ is a multiple of the critical load, and
$x_2 \in [-0.05, 0.05]$ is a small sideways push. The beam balances bending
against the load with a cubic term,

$$
g(x, y) = \frac{K y}{\lambda^2} + x_1 \frac{D y}{\lambda} + y^3 - x_2,
$$

where $D$ is the second difference and $K$ the fourth difference on $m = 24$
points, with both ends pinned, and $\lambda$ is the lowest eigenvalue of $-D$.
The lowest bending mode then has stiffness $1 - x_1$.

<figure class="schematic">
<svg viewBox="0 0 520 230" width="520" role="img" aria-label="A column pressed from its ends, and a bifurcation diagram where the straight shape splits into two bowed shapes at the critical load">
<path class="line dash thin" d="M90 40V190"/>
<path class="line" d="M90 40C70 90 70 140 90 190"/>
<path class="line" d="M90 40V32M90 190V198M70 198H110L90 190ZM70 32H110"/>
<path class="line" d="M60 4V28M54 20l6 8 6-8M120 4V28M114 20l6 8 6-8"/>
<path class="line thin" d="M130 115H112M118 110l-6 5 6 5"/>
<text x="135" y="119" class="small">x₁</text>
<text x="140" y="18" class="small">load x₀</text>
<path class="line thin" d="M240 195H490M240 195V25"/>
<path class="line thin" d="M240 108H360"/>
<path class="line thin dash" d="M360 108H490"/>
<path class="accent" d="M240 108H360M360 108C360 88 380 60 490 46M360 108C360 128 380 156 490 170"/>
<path class="line thin" d="M360 195V201M480 195V201M240 195V201"/>
<text x="240" y="216" text-anchor="middle" class="small">0</text>
<text x="360" y="216" text-anchor="middle" class="small">1</text>
<text x="480" y="216" text-anchor="middle" class="small">2</text>
<text x="490" y="190" text-anchor="end" class="small">x₀</text>
<text x="246" y="22" class="small">deflection of the beam</text>
</svg>
<figcaption>A pinned column under load, and the shapes it takes as the load grows.</figcaption>
</figure>

Past $x_1 = 1$ the straight beam loses that stiffness. The Jacobian turns
singular, and two bowed shapes appear, one for each direction of the push. The
system is a bifurcation, so it tests the places where a fit can go wrong. The
reduced solver reports several roots and a poor conditioning near the critical
load, and Newton converges only linearly there, so the convergence check needs
to be switched off for the snapshots. The fourth difference cancels in single
precision on a fine grid, so the beam keeps to a few dozen points. The builder
is
[`buckling_beam`](https://github.com/nardi/surreax/blob/main/tests/implicit_systems.py#L283).

- [`test_beam_buckles_past_the_critical_load`](https://github.com/nardi/surreax/blob/main/tests/integration/test_implicit_systems.py#L94)
- [`test_beam_report_flags_two_shapes_near_the_critical_load`](https://github.com/nardi/surreax/blob/main/tests/integration/test_reduced_order_systems.py#L42)

### Resistor network

A mesh of resistors is driven by current injected at a few nodes. Kirchhoff's
current law at node $i$ gives

$$
g_i(x, y) = \sum_j w_{ij} (y_i - y_j) + 0.1\, y_i - (E x)_i,
$$

where $w_{ij}$ is the conductance of the wire between nodes $i$ and $j$, drawn
uniformly from $[0.5, 2]$, and the term $0.1\, y_i$ is a small leak to ground.
The matrix $E$ places each input as a current at one node. The mesh is a grid
with a few random long wires added.

The residual is the graph Laplacian in place of the chain of the rod. It is
affine in $y$ and in $x$, so the voltages are an exact linear map of the
sources, as for the heated rod. The difference is the sparsity. The long wires
make the Jacobian sparse without being banded, which tests that the pattern
reading, the coloring and the sparse solve do not rely on a chain. The system
also shows that a greedy choice of snapshots reaches an exact solution with six
full solves for four sources. The builder is
[`resistor_network`](https://github.com/nardi/surreax/blob/main/tests/implicit_systems.py#L363).

- [`test_resistor_network_is_exact_on_a_general_graph`](https://github.com/nardi/surreax/blob/main/tests/integration/test_reduced_order_systems.py#L57)
- [`test_linear_systems_have_exact_low_rank_manifolds`](https://github.com/nardi/surreax/blob/main/tests/integration/test_implicit_systems.py#L65)

### Diode network

The resistor mesh with a diode in place of every third wire. A diode from node
$a$ to node $b$ carries the Shockley current

$$
I = I_s \left( e^{(y_a - y_b)/V_T} - 1 \right), \qquad I_s = 0.05,
\quad V_T = 0.5,
$$

and the node equations add it to the resistor currents with the sign that fits
each end of the wire.

<figure class="schematic">
<svg viewBox="0 0 540 200" width="540" role="img" aria-label="A mesh of nodes joined by resistors and diodes, with one long wire and current sources at two nodes">
<path class="line" d="M90 40H360M90 95H360M90 150H360M90 40V150M180 40V150M270 40V150M360 40V150"/>
<path class="line thin dash" d="M90 40C200 -10 300 200 360 150"/>
<path class="accent" d="M180 40H270M90 95V150M360 95H360M270 95V150M180 95H270"/>
<path class="solid" d="M219 34V46L231 40ZM225 150V150"/>
<path class="solid" d="M84 117H96L90 129Z"/>
<path class="solid" d="M264 117H276L270 129Z"/>
<path class="solid" d="M219 89V101L231 95Z"/>
<path class="line" d="M231 34V46M84 129H96M264 129H276M231 89V101"/>
<g class="node">
<circle cx="90" cy="40" r="7"/><circle cx="180" cy="40" r="7"/><circle cx="270" cy="40" r="7"/><circle cx="360" cy="40" r="7"/>
<circle cx="90" cy="95" r="7"/><circle cx="180" cy="95" r="7"/><circle cx="270" cy="95" r="7"/><circle cx="360" cy="95" r="7"/>
<circle cx="90" cy="150" r="7"/><circle cx="180" cy="150" r="7"/><circle cx="270" cy="150" r="7"/><circle cx="360" cy="150" r="7"/>
</g>
<path class="line" d="M30 95H78M70 90l8 5-8 5"/>
<path class="line" d="M450 40H372M380 35l-8 5 8 5"/>
<text x="30" y="86" class="small">x₁</text>
<text x="440" y="30" class="small">x₂</text>
<path class="line thin" d="M450 120H480" /><text x="488" y="124" class="small">resistor</text>
<path class="accent" d="M450 140H480"/><text x="488" y="144" class="small">diode</text>
<path class="line thin dash" d="M450 160H480"/><text x="488" y="164" class="small">long wire</text>
</svg>
<figcaption>A mesh with resistors, diodes, a long wire and two current sources.</figcaption>
</figure>

The exponential makes the voltages a curved function of the sources, with the
nonlinearity only on the diode wires. It is also stiff. Newton diverges from a
start far from a root, which makes it the system for the convergence check, and
the reason the reduced solvers on it use a small box of starting voltages. It is
a four-input target for the piecewise linear forms, where it is only mildly
nonconvex. The learned manifold hyper-reduces on its graph stencil. The builder
is
[`diode_network`](https://github.com/nardi/surreax/blob/main/tests/implicit_systems.py#L395).

- [`test_exact_solve_raises_when_the_newton_solves_diverge`](https://github.com/nardi/surreax/blob/main/tests/integration/test_convergence_check.py#L25)
- [`test_learned_manifold_hyper_reduces_on_a_graph`](https://github.com/nardi/surreax/blob/main/tests/integration/test_hyperreduction.py#L98)
- [`test_a_relu_network_approximates_each_system_far_better_than_an_affine_fit`](https://github.com/nardi/surreax/blob/main/tests/integration/test_network_forms_comparison.py#L63)

### Latch ring

A latch is a pair of cross-coupled inverters. A ring of latches has a weak
coupling $c = 0.05$ between neighbors. Node $2k$ and node $2k+1$ of latch $k$
drive each other,

$$
g_a = y_a - f(y_b) + c\,(2 y_a - y_{a'} - y_{a''}), \qquad
g_b = y_b - f(y_a),
$$

where $a'$ and $a''$ are the corresponding nodes of the neighboring latches and
the inverter is
$f(v) = \tfrac12 \left(1 - \tanh(x_1 (v - \tfrac12 - x_2))\right)$. The loop
gain $x_1$ ranges over $[3, 6]$ and the bias $x_2$ over $[-0.1, 0.1]$.

<figure class="schematic">
<svg viewBox="0 0 520 170" width="520" role="img" aria-label="Two cross-coupled inverters forming a latch, and a ring of four latches with weak coupling">
<path class="line" d="M60 30V60L100 45ZM100 45"/>
<circle class="node" cx="105" cy="45" r="4.5"/>
<path class="line" d="M140 100V130L100 115Z"/>
<circle class="node" cx="95" cy="115" r="4.5"/>
<path class="line" d="M110 45H160V115H140M91 115H30V45H60"/>
<circle class="solid" cx="160" cy="45" r="3.5"/>
<circle class="solid" cx="30" cy="115" r="3.5"/>
<text x="168" y="44" class="small">a</text>
<text x="14" y="130" class="small">b</text>
<text x="80" y="24" text-anchor="middle" class="small">y<tspan dy="3" font-size="9">a</tspan><tspan dy="-3"> = f(y</tspan><tspan dy="3" font-size="9">b</tspan><tspan dy="-3">)</tspan></text>
<text x="80" y="150" text-anchor="middle" class="small">y<tspan dy="3" font-size="9">b</tspan><tspan dy="-3"> = f(y</tspan><tspan dy="3" font-size="9">a</tspan><tspan dy="-3">)</tspan></text>
<g class="line">
<rect x="350" y="12" width="40" height="24" rx="4"/><rect x="415" y="73" width="40" height="24" rx="4"/>
<rect x="350" y="134" width="40" height="24" rx="4"/><rect x="285" y="73" width="40" height="24" rx="4"/>
</g>
<path class="line thin dash" d="M392 30Q420 35 430 70M430 100Q425 135 392 142M348 142Q318 135 308 100M308 70Q318 35 348 30"/>
<text x="370" y="29" text-anchor="middle" class="small">1</text>
<text x="435" y="90" text-anchor="middle" class="small">2</text>
<text x="370" y="151" text-anchor="middle" class="small">3</text>
<text x="305" y="90" text-anchor="middle" class="small">4</text>
<text x="370" y="90" text-anchor="middle" class="small">coupling c</text>
</svg>
<figcaption>A latch of two inverters, and a ring of four latches.</figcaption>
</figure>

With the loop gain above two, each latch holds a stored zero or a stored one, so
the ring has many solutions. They merge at a fold as the bias is swept. No
function of the input passes through all of them, and a solver that starts from
zero reaches only one. The system tests how a fit behaves when the solution is
not unique. The reduced solver reports multiple roots, and the tests start
Newton from a stored zero and from a stored one to confirm that both are
solutions. The builder is
[`latch_ring`](https://github.com/nardi/surreax/blob/main/tests/implicit_systems.py#L443).

- [`test_latch_ring_holds_two_states`](https://github.com/nardi/surreax/blob/main/tests/integration/test_implicit_systems.py#L117)
- [`test_latch_ring_report_flags_two_stored_states`](https://github.com/nardi/surreax/blob/main/tests/integration/test_reduced_order_systems.py#L52)

## Derived test cases

### Reduced solvers as explicit targets

The piecewise linear forms need explicit targets, so
[`Case`](https://github.com/nardi/surreax/blob/main/tests/piecewise_cases.py#L17)
turns an implicit system into one. It builds a `ReducedSolve` with 48 snapshots
and a dense linear solver, and takes the latent coefficients $z(x)$ of the
reduced solver as the target. The decode back to the field is affine, so a
piecewise linear latent is a piecewise linear field. A case scores a fit by the
mean latent error over the mean latent norm on 100 inputs held out from the fit.

[`build`](https://github.com/nardi/surreax/blob/main/tests/piecewise_cases.py#L52)
makes three cases. The `rod` is the radiating rod on 32 points with two heaters
at rank two, the `network` is the diode network on a $12 \times 12$ mesh with
four sources at rank four, and the `thermostat` is the thermostat rod on 32
points at rank three. The [model forms](forms.md#relu-network) page quotes the
comparisons that they produce.

- [`test_a_difference_of_max_affine_approximates_the_system_better_than_an_affine_fit`](https://github.com/nardi/surreax/blob/main/tests/integration/test_piecewise_linear_systems.py#L21)
- [`test_a_max_affine_fit_is_never_worse_than_an_affine_fit`](https://github.com/nardi/surreax/blob/main/tests/integration/test_piecewise_linear_systems.py#L60)
- [`test_a_relu_network_approximates_each_system_far_better_than_an_affine_fit`](https://github.com/nardi/surreax/blob/main/tests/integration/test_network_forms_comparison.py#L63)

### Optimization over exported surrogates

The library exports arrays and leaves the encoding of a solver program to the
user. The functions in
[`minimize_convex`](https://github.com/nardi/surreax/blob/main/tests/optimization_encodings.py#L30)
and
[`minimize_relu`](https://github.com/nardi/surreax/blob/main/tests/optimization_encodings.py#L104)
are the encodings a user would write. Each minimizes a linear cost $c^\top x$
over a box with bounds on the outputs of the surrogate, and returns the
minimizer in the coordinates of the export.

For a convex surrogate, the output is a maximum of affine pieces, and a maximum
is at most a bound $u$ when every piece is. The constraint is then a set of
linear inequalities,

$$
a_j^\top x + b_j \le u \quad \text{for every piece } j,
$$

and a convex network replaces each ReLU by a variable that is at least its
pre-activation. For a ReLU network the program is mixed-integer. A hidden unit
$a = \max(p, 0)$ with pre-activation $p$ gets a binary $s$ and bounds
$l \le p \le u$ from interval arithmetic, and the constraints are

$$
a \ge p, \qquad a \ge 0, \qquad a \le p - l\,(1 - s), \qquad a \le u\, s.
$$

The binary is one exactly when the unit is active. Two integration tests then
check the plans on the full systems. The first fits a convex surrogate to two
sensor temperatures of the radiating rod and finds the least heater power that
keeps both above a target. The second fits a ReLU network to the thermostat rod
and solves a mixed-integer plan with a limit and a floor on two sensors. Each
test solves the full implicit system at the minimizer and compares the cost
against the best cost over a grid of full solves.

- [`test_the_least_heater_power_plan_of_a_convex_surrogate_holds_on_the_rod`](https://github.com/nardi/surreax/blob/main/tests/integration/test_optimization_export.py#L71)
- [`test_the_mixed_integer_plan_of_a_relu_network_holds_on_the_thermostat_rod`](https://github.com/nardi/surreax/blob/main/tests/integration/test_optimization_export.py#L136)
