"""The fitted surrogate object.

It stores every pipeline stage and applies them in order, on a single example.
Because it is an Equinox module it registers as a pytree, so it differentiates,
`vmap`s, and `jit`s like any other JAX function, and it carries the domain and
the error report alongside its parameters.
"""

import equinox as eqx
from jaxtyping import Array, Float, PyTree

from surreax.domain import Domain, coverage
from surreax.fitting import LinearModel
from surreax.flatten import Unraveler
from surreax.forms.difference_max_affine import DifferenceOfMaxAffine
from surreax.forms.max_affine import MaxAffine
from surreax.forms.network import Network
from surreax.forms.relu_network import ConvexNetwork, ReluNetwork
from surreax.normalize import Normalizer
from surreax.reduction import LinearReduction
from surreax.report import ErrorReport


class FittedSurrogate(eqx.Module):
    """An approximation of a target that keeps the chosen model form.

    Applies `flatten -> encode -> normalize -> model -> denormalize -> decode
    -> unflatten`, so from the caller's side it has the target's signature.
    """

    input_reduction: LinearReduction
    input_normalizer: Normalizer
    model: (
        LinearModel | MaxAffine | DifferenceOfMaxAffine | Network | ConvexNetwork | ReluNetwork
    )
    output_normalizer: Normalizer
    output_reduction: LinearReduction
    output_unravel: Unraveler = eqx.field(static=True)
    domain: Domain
    report: ErrorReport

    def __call__(self, x: PyTree[Array]) -> PyTree[Array]:
        encoded = self.input_reduction.forward(self.domain.flatten(x))
        normalized = self.input_normalizer.forward(encoded)
        predicted = self.model(normalized)
        reduced_output = self.output_normalizer.inverse(predicted)
        flat_output = self.output_reduction.inverse(reduced_output)
        return self.output_unravel(flat_output)

    def coverage(self, x: PyTree[Array]) -> Float[Array, ""]:
        """How far an input lies from where the surrogate was fitted, see `domain.coverage`.

        At most one means the input is covered. Above one, the prediction there
        is an extrapolation.
        """
        return coverage(self.domain, self.domain.flatten(x))
