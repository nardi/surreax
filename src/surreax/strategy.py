"""Choosing a form or a method from the structure of a target.

The structural analysis already reads which inputs drive which outputs. This adds
a reading of what kind of function each output is, affine, polynomial, piecewise
linear, or discontinuous, and turns both readings into a recommendation, so a
form or a method need not be given by hand. The reading is numerical, from how
the Jacobian and the curvature vary over the domain, not from the source, so it
works on any JAX function. It is advisory and always overridable.
"""

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PyTree

from surreax import sampling
from surreax.degree import polynomial_degrees
from surreax.domain import Domain
from surreax.flatten import ravel
from surreax.forms import (
    Affine,
    ChebyshevPolynomial,
    Form,
    MaxAffineForm,
    NetworkForm,
)
from surreax.reduction import (
    InputReducer,
    OutputReducer,
    active_subspace_inputs,
    pca_outputs,
)
from surreax.structure import analyze

Target = Callable[[PyTree[Array]], PyTree[Array]]
Residual = Callable[[Float[Array, " n"], Float[Array, " m"]], Float[Array, " m"]]


def function_kinds(
    target: Target,
    domain: Domain,
    *,
    samples: int = 128,
    tolerance: float = 1e-3,
    jump_ratio: float = 20.0,
    seed: int = 0,
) -> tuple[str, ...]:
    """Read the kind of each flat output of a target over the domain.

    Returns one of `affine`, `polynomial`, `piecewise_linear`, or `discontinuous`
    per output. An affine output has a constant Jacobian, a piecewise linear one
    a varying Jacobian but no curvature, a polynomial one real curvature, and a
    discontinuous one a value jump that the Jacobian does not see.

    Integer and categorical coordinates are held at the values of the first
    sample, so their steps do not read as jumps, and the reading covers the
    continuous coordinates only.
    """
    discrete = domain.discrete_mask
    continuous = np.flatnonzero(~discrete)
    if continuous.size == 0:
        raise ValueError("function_kinds needs at least one continuous coordinate")
    inputs = sampling.sample(domain, samples, seed=seed)
    inputs = jnp.where(jnp.asarray(discrete), inputs[0], inputs)

    def flat_target(point: Float[Array, " d"]) -> Float[Array, " k"]:
        return ravel(target(domain.unravel(point)))

    values = jax.vmap(flat_target)(inputs)
    jacobians = jax.vmap(jax.jacobian(flat_target))(inputs)[:, :, continuous]
    hessians = jax.vmap(jax.hessian(flat_target))(inputs)[:, :, continuous][..., continuous]

    scale = jnp.mean(jnp.linalg.norm(jacobians, axis=2), axis=0) + 1e-12
    slope_variation = jnp.mean(
        jnp.linalg.norm(jacobians - jnp.mean(jacobians, axis=0), axis=2), axis=0
    )
    curvature = jnp.mean(jnp.linalg.norm(hessians, axis=(2, 3)), axis=0)

    # A value jump the gradient misses shows up as a large adjacent secant along
    # the first continuous coordinate, since the samples are ordered along it.
    order = jnp.argsort(inputs[:, continuous[0]])
    gaps = jnp.linalg.norm(jnp.diff(inputs[order], axis=0), axis=1) + 1e-12
    secant = jnp.max(jnp.abs(jnp.diff(values[order], axis=0)) / gaps[:, None], axis=0)

    scale_np = np.asarray(scale)
    kinds = []
    for k in range(values.shape[1]):
        if float(secant[k]) > jump_ratio * scale_np[k]:
            kinds.append("discontinuous")
        elif float(slope_variation[k]) <= tolerance * scale_np[k]:
            kinds.append("affine")
        elif float(curvature[k]) <= tolerance * scale_np[k]:
            kinds.append("piecewise_linear")
        else:
            kinds.append("polynomial")
    return tuple(kinds)


def _form_for_kinds(kinds: frozenset[str], degree: int | None = None) -> Form:
    """Pick the least form that covers every kind in the set.

    A smooth form takes `degree` when it is known from the jaxpr, and degree
    three otherwise.
    """
    if "discontinuous" in kinds:
        warnings.warn(
            "a discontinuous output was detected, falling back to a network form",
            stacklevel=2,
        )
        return NetworkForm(width=32, depth=2)
    smooth = "polynomial" in kinds
    kinked = "piecewise_linear" in kinds
    if smooth and kinked:
        return NetworkForm(width=32, depth=2)
    if kinked:
        return MaxAffineForm(pieces=4)
    if smooth:
        return ChebyshevPolynomial(degree=3 if degree is None else max(degree, 2))
    return Affine()


def _known_degree(kinds: Sequence[str], degrees: Sequence[int | None]) -> int | None:
    """The largest jaxpr degree of the curved outputs, or None if one has none."""
    curved = [degree for kind, degree in zip(kinds, degrees, strict=True) if kind == "polynomial"]
    if not curved or any(degree is None for degree in curved):
        return None
    return max(degree for degree in curved if degree is not None)


def recommend_form(target: Target, domain: Domain, *, samples: int = 128, seed: int = 0) -> Form:
    """Recommend one form for a target, from the kinds of all its outputs.

    A polynomial form takes the degree read from the target's jaxpr, see
    `degree.polynomial_degrees`, when every curved output is a polynomial.
    """
    kinds = function_kinds(target, domain, samples=samples, seed=seed)
    degrees = polynomial_degrees(target, domain)
    return _form_for_kinds(frozenset(kinds), _known_degree(kinds, degrees))


def recommend_block_forms(
    target: Target, domain: Domain, *, samples: int = 128, tolerance: float = 1e-6, seed: int = 0
) -> list[Form]:
    """Recommend a form per output block, in sorted-label order.

    Each block takes the least form that covers the kinds of its outputs, so a
    linear block is fitted exactly and only a nonlinear one pays for a richer
    form.
    """
    kinds = function_kinds(target, domain, samples=samples, seed=seed)
    degrees = polynomial_degrees(target, domain)
    structure = analyze(target, domain, samples=samples, tolerance=tolerance, seed=seed)
    labels = np.asarray(structure.output_block)
    forms: list[Form] = []
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        block_kinds = [kinds[i] for i in members]
        block_degree = _known_degree(block_kinds, [degrees[i] for i in members])
        forms.append(_form_for_kinds(frozenset(block_kinds), block_degree))
    return forms


def residual_is_affine_in_unknown(
    residual: Residual,
    domain: Domain,
    unknown_domain: Domain,
    *,
    samples: int = 128,
    tolerance: float = 1e-3,
    seed: int = 0,
) -> bool:
    """Whether the residual's Jacobian in the unknown is constant over samples."""
    inputs = sampling.sample(domain, samples, seed=seed)
    unknowns = sampling.sample(unknown_domain, samples, seed=seed + 1)
    jacobians = jax.vmap(jax.jacobian(residual, argnums=1))(inputs, unknowns)
    scale = jnp.mean(jnp.linalg.norm(jacobians, axis=(1, 2))) + 1e-12
    variation = jnp.mean(jnp.linalg.norm(jacobians - jnp.mean(jacobians, axis=0), axis=(1, 2)))
    return bool(variation <= tolerance * scale)


def linear_energy(states: Float[Array, "s m"], rank: int) -> float:
    """Fraction of the variation of solved states that `rank` linear modes capture.

    The singular values of the centered states decay fast when the solutions lie
    near a linear subspace and slowly when the manifold bends or carries a moving
    feature, so this reads the Kolmogorov n-width at `rank` from the snapshots.
    """
    singular = jnp.linalg.svd(states - jnp.mean(states, axis=0), compute_uv=False)
    total = jnp.sum(singular**2)
    return float(jnp.sum(singular[:rank] ** 2) / jnp.where(total > 0, total, 1.0))


@dataclass(frozen=True)
class FitSettings:
    """A recommended way to fit a target, which `fit` applies when no form is given."""

    form: Form
    """The model form, with its degree when it is a polynomial."""

    reduce_inputs: InputReducer | None = None
    """An input reduction, recommended when the target varies along few directions."""

    reduce_outputs: OutputReducer | None = None
    """An output reduction, recommended when the outputs move together."""

    sparse_terms: int | None = None
    """A sparse basis budget, recommended when the basis outgrows the samples."""


def _rank_for(strengths: Float[Array, " m"], variance: float) -> int:
    """The fewest directions whose squared strengths reach `variance` of the total.

    All-zero strengths give one direction.
    """
    energy = np.cumsum(np.asarray(strengths) ** 2)
    return int(np.searchsorted(energy, variance * energy[-1]) + 1)


def recommend_fit(
    target: Target,
    domain: Domain,
    *,
    samples: int = 256,
    variance: float = 0.99,
    max_degree: int = 6,
    improvement: float = 2.0,
    seed: int = 0,
) -> FitSettings:
    """Recommend a form, its degree, the reductions and a sparse budget for a target.

    The form comes from `recommend_form`. For a polynomial form, the rest of the
    settings come from three readings of `samples` points.

    - When a curved output is not a polynomial in its jaxpr, the degree is the
      lowest at which raising it no longer cuts the held-out error by the factor
      `improvement`, searched up to `max_degree`.
    - An input reduction is recommended when the active-subspace spectrum of the
      continuous inputs reaches `variance` at a rank below their count, and an
      output reduction when output PCA does at no more than half the outputs.
    - A sparse budget is recommended when the basis has more terms than half the
      training samples, chosen by held-out error over a short sweep.

    The settings are advisory, and `fit` lets each be overridden.
    """
    from surreax.api import fit

    form = recommend_form(target, domain, samples=samples, seed=seed)
    if not isinstance(form, ChebyshevPolynomial):
        return FitSettings(form=form)

    discrete = domain.discrete_mask
    continuous = np.flatnonzero(~discrete)
    inputs = sampling.sample(domain, samples, seed=seed)

    def flat_target(point: Float[Array, " d"]) -> Float[Array, " k"]:
        return ravel(target(domain.unravel(point)))

    outputs = jax.vmap(flat_target)(inputs)
    jacobians = jax.vmap(jax.jacobian(flat_target))(inputs)[:, :, continuous]
    gram = jnp.mean(jnp.einsum("nkd,nke->nde", jacobians, jacobians), axis=0)
    input_strengths = jnp.sqrt(jnp.clip(jnp.linalg.eigvalsh(gram)[::-1], 0.0))
    input_rank = _rank_for(input_strengths, variance)
    reduce_inputs = None
    if continuous.size >= 3 and input_rank < continuous.size:
        reduce_inputs = active_subspace_inputs(rank=input_rank)
    output_strengths = jnp.linalg.svd(outputs - jnp.mean(outputs, axis=0), compute_uv=False)
    output_rank = _rank_for(output_strengths, variance)
    correlated = outputs.shape[1] >= 4 and output_rank <= outputs.shape[1] // 2
    reduce_outputs = pca_outputs(rank=output_rank) if correlated else None

    def held_out_error(candidate: ChebyshevPolynomial, sparse_terms: int | None) -> float:
        fitted = fit(
            target,
            domain,
            candidate,
            samples=samples,
            reduce_inputs=reduce_inputs,
            reduce_outputs=reduce_outputs,
            sparse_terms=sparse_terms,
            worst_case_starts=0,
            seed=seed,
        )
        return float(fitted.report.value_mean)

    kinds = function_kinds(target, domain, samples=samples, seed=seed)
    if _known_degree(kinds, polynomial_degrees(target, domain)) is None:
        degree, error = 2, held_out_error(ChebyshevPolynomial(degree=2), None)
        while degree < max_degree:
            next_error = held_out_error(ChebyshevPolynomial(degree=degree + 1), None)
            if next_error * improvement > error:
                break
            degree, error = degree + 1, next_error
        form = ChebyshevPolynomial(degree=degree)

    kept = input_rank if reduce_inputs is not None else continuous.size
    terms = form.features(jnp.zeros(kept + int(discrete.sum()))).shape[0]
    training = samples - max(1, round(samples * 0.2))
    sparse_terms = None
    if terms > training // 2:
        budgets = sorted({max(1, training // share) for share in (2, 4, 8)})
        errors = [held_out_error(form, budget) for budget in budgets]
        sparse_terms = budgets[int(np.argmin(errors))]
    return FitSettings(
        form=form,
        reduce_inputs=reduce_inputs,
        reduce_outputs=reduce_outputs,
        sparse_terms=sparse_terms,
    )
