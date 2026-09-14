"""Fitting a separable target as independent per-block surrogates.

The structural decomposition finds groups of outputs and inputs that do not
couple. This fits each group on its own inputs and assembles the pieces, so a
target that splits into small blocks is fitted as small blocks, which is faster
and more accurate than one fit over everything. Each block carries its own
fitted surrogate, so blocks can take different forms, which the strategy
selection uses to give a linear block a closed-form fit and a nonlinear block a
richer one.

The composite holds static index maps, so its call scatters each block's inputs
and gathers its outputs without rebuilding anything, and it stays a pytree that
`jit`, `vmap`, and `jacobian` pass through.
"""

from collections.abc import Callable, Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int, PyTree

from surreax.domain import Domain, box
from surreax.fitted import FittedSurrogate
from surreax.flatten import ravel
from surreax.forms import DifferenceOfMaxAffineForm, LinearForm, MaxAffineForm, NetworkForm
from surreax.report import ErrorReport
from surreax.structure import analyze

Target = Callable[[PyTree[Array]], PyTree[Array]]
Form = LinearForm | MaxAffineForm | DifferenceOfMaxAffineForm | NetworkForm


class CompositeSurrogate(eqx.Module):
    """A surrogate assembled from one fitted surrogate per independent block."""

    blocks: tuple[FittedSurrogate, ...]
    input_indices: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    output_indices: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    output_dimension: int = eqx.field(static=True)
    report: ErrorReport

    def __call__(self, x: Float[Array, " d"]) -> Float[Array, " k"]:
        flat = ravel(x)
        output = jnp.zeros(self.output_dimension, dtype=flat.dtype)
        for block, inputs, outputs in zip(
            self.blocks, self.input_indices, self.output_indices, strict=True
        ):
            predicted = block(flat[jnp.asarray(inputs)])
            output = output.at[jnp.asarray(outputs)].set(predicted)
        return output


def _coupling_bound(
    coupling: Float[Array, "k d"],
    output_block: Int[Array, " k"],
    input_block: Int[Array, " d"],
    lower: Float[Array, " d"],
    upper: Float[Array, " d"],
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """First-order error from ignoring the coupling dropped between blocks.

    Each off-block entry moves an output by about its sensitivity times the input
    half-range, so the sum over off-block inputs bounds the error of treating the
    blocks as independent.
    """
    same_block = output_block[:, None] == input_block[None, :]
    half_range = (upper - lower) / 2
    off_block = jnp.where(same_block, 0.0, coupling) * half_range[None, :]
    per_output = jnp.sum(off_block, axis=1)
    return jnp.mean(per_output), jnp.max(per_output)


def fit_separable(
    target: Target,
    domain: Domain,
    form: Form | Sequence[Form],
    *,
    samples: int = 512,
    holdout: float = 0.2,
    ridge: float = 1e-8,
    tolerance: float = 1e-6,
    seed: int = 0,
) -> CompositeSurrogate:
    """Fit a target as independent per-block surrogates and assemble them.

    Reads the block decomposition from the structural analysis, fits each block
    on its own inputs over its sub-box, and reports the worst per-block error and
    the bound on the error from the coupling dropped between blocks.

    `form` is one form for every block, or a sequence of forms, one per block in
    sorted-label order, so a linear block can take a closed-form fit and a
    nonlinear one a richer form.
    """
    from surreax.api import fit

    structure = analyze(target, domain, samples=samples, tolerance=tolerance, seed=seed)
    input_block = np.asarray(structure.input_block)
    output_block = np.asarray(structure.output_block)
    labels = np.unique(output_block)
    block_forms = list(form) if isinstance(form, Sequence) else [form] * labels.shape[0]
    lower, upper = domain.bounds()
    lower_np, upper_np = np.asarray(lower), np.asarray(upper)

    blocks: list[FittedSurrogate] = []
    input_indices: list[tuple[int, ...]] = []
    output_indices: list[tuple[int, ...]] = []
    value_means: list[Float[Array, ""]] = []
    value_maxes: list[Float[Array, ""]] = []
    structure_cost = 0
    output_dimension = output_block.shape[0]
    # Ignored inputs are anchored at the box midpoint, which the block's outputs
    # do not depend on, so their value does not matter.
    midpoint = jnp.asarray((lower_np + upper_np) / 2)

    for index, label in enumerate(labels):
        inputs = tuple(int(j) for j in np.flatnonzero(input_block == label))
        outputs = tuple(int(i) for i in np.flatnonzero(output_block == label))
        chosen = jnp.asarray(inputs)
        gathered = jnp.asarray(outputs)

        def block_target(sub_input, chosen=chosen, gathered=gathered):
            full = midpoint.at[chosen].set(sub_input)
            return ravel(target(domain.unravel(full)))[gathered]

        sub_domain = box(lower_np[list(inputs)].tolist(), upper_np[list(inputs)].tolist())
        fitted = fit(
            block_target,
            sub_domain,
            block_forms[index],
            samples=samples,
            holdout=holdout,
            ridge=ridge,
            worst_case_starts=0,
            seed=seed,
        )
        blocks.append(fitted)
        input_indices.append(inputs)
        output_indices.append(outputs)
        value_means.append(fitted.report.value_mean)
        value_maxes.append(fitted.report.value_max)
        structure_cost += fitted.report.structure_cost

    coupling_mean, coupling_max = _coupling_bound(
        structure.coupling, structure.output_block, structure.input_block, lower, upper
    )
    report = ErrorReport(
        value_mean=jnp.max(jnp.stack(value_means)),
        value_max=jnp.max(jnp.stack(value_maxes)),
        derivative_mean=jnp.asarray(0.0),
        derivative_max=jnp.asarray(0.0),
        projection_mean=coupling_mean,
        projection_max=coupling_max,
        worst_case=jnp.asarray(0.0),
        worst_case_point=jnp.zeros(input_block.shape[0]),
        structure_cost=structure_cost,
    )
    return CompositeSurrogate(
        blocks=tuple(blocks),
        input_indices=tuple(input_indices),
        output_indices=tuple(output_indices),
        output_dimension=output_dimension,
        report=report,
    )
