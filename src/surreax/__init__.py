"""Surrogate model training for JAX functions.

`fit` takes a JAX target, a domain to be accurate over, and a model form, and
returns a `FittedSurrogate` that keeps the chosen form.
"""

from surreax.api import Target, fit
from surreax.domain import Box, Domain, box, box_from_data
from surreax.fitted import FittedSurrogate
from surreax.forms import Affine, ChebyshevPolynomial, LinearForm, MaxAffine, MaxAffineForm
from surreax.reduction import active_subspace_inputs, pca_inputs, pca_outputs
from surreax.report import ErrorReport

__all__ = [
    "Affine",
    "Box",
    "ChebyshevPolynomial",
    "Domain",
    "ErrorReport",
    "FittedSurrogate",
    "LinearForm",
    "MaxAffine",
    "MaxAffineForm",
    "Target",
    "active_subspace_inputs",
    "box",
    "box_from_data",
    "fit",
    "pca_inputs",
    "pca_outputs",
]
