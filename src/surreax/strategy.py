"""Choosing a form or a method from the structure of a target.

The structural analysis already reads which inputs drive which outputs. This adds
a reading of what kind of function each output is, affine, polynomial, piecewise
linear, or discontinuous, and turns both readings into a recommendation, so a
form or a method need not be given by hand. The reading is numerical, from how
the Jacobian and the curvature vary over the domain, not from the source, so it
works on any JAX function. It is advisory and always overridable.
"""

import warnings
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PyTree

from surreax import sampling
from surreax.domain import Domain
from surreax.flatten import ravel
from surreax.forms import (
    Affine,
    ChebyshevPolynomial,
    DifferenceOfMaxAffineForm,
    LinearForm,
    MaxAffineForm,
    NetworkForm,
)
from surreax.structure import analyze

Target = Callable[[PyTree[Array]], PyTree[Array]]
Form = LinearForm | MaxAffineForm | DifferenceOfMaxAffineForm | NetworkForm
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
    """
    inputs = sampling.sample(domain, samples, seed=seed)

    def flat_target(point: Float[Array, " d"]) -> Float[Array, " k"]:
        return ravel(target(domain.unravel(point)))

    values = jax.vmap(flat_target)(inputs)
    jacobians = jax.vmap(jax.jacobian(flat_target))(inputs)
    hessians = jax.vmap(jax.hessian(flat_target))(inputs)

    scale = jnp.mean(jnp.linalg.norm(jacobians, axis=2), axis=0) + 1e-12
    slope_variation = jnp.mean(
        jnp.linalg.norm(jacobians - jnp.mean(jacobians, axis=0), axis=2), axis=0
    )
    curvature = jnp.mean(jnp.linalg.norm(hessians, axis=(2, 3)), axis=0)

    # A value jump the gradient misses shows up as a large adjacent secant along
    # the first coordinate, since the samples are ordered along it.
    order = jnp.argsort(inputs[:, 0])
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


def _form_for_kinds(kinds: frozenset[str]) -> Form:
    """Pick the least form that covers every kind in the set."""
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
        return ChebyshevPolynomial(degree=3)
    return Affine()


def recommend_form(target: Target, domain: Domain, *, samples: int = 128, seed: int = 0) -> Form:
    """Recommend one form for a target, from the kinds of all its outputs."""
    kinds = function_kinds(target, domain, samples=samples, seed=seed)
    return _form_for_kinds(frozenset(kinds))


def recommend_block_forms(
    target: Target, domain: Domain, *, samples: int = 128, tolerance: float = 1e-6, seed: int = 0
) -> list[Form]:
    """Recommend a form per output block, in sorted-label order.

    Each block takes the least form that covers the kinds of its outputs, so a
    linear block is fitted exactly and only a nonlinear one pays for a richer
    form.
    """
    kinds = function_kinds(target, domain, samples=samples, seed=seed)
    structure = analyze(target, domain, samples=samples, tolerance=tolerance, seed=seed)
    labels = np.asarray(structure.output_block)
    kinds_array = np.asarray(kinds)
    return [
        _form_for_kinds(frozenset(kinds_array[labels == label].tolist()))
        for label in np.unique(labels)
    ]


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
