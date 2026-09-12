"""The interface a model form provides.

A model form supplies a feature map. For a form that is linear in its
parameters, the output for a weight matrix `w` is `w @ features(x)`, one row of
`w` per output coordinate. This is the whole interface, so a user can drop in
their own form by writing a single method.
"""

from typing import Protocol, runtime_checkable

from jaxtyping import Array, Float


@runtime_checkable
class LinearForm(Protocol):
    """A form whose output is linear in its parameters through a feature map."""

    def features(self, x: Float[Array, " d"]) -> Float[Array, " f"]:
        """Map one input to its feature vector."""
        ...
