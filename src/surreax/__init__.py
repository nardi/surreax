"""Surrogate model training for JAX functions.

`fit` takes a JAX target, a domain to be accurate over, and a model form, and
returns a `FittedSurrogate` that keeps the chosen form.
"""

from surreax.api import Target, fit, fit_adaptive
from surreax.composite import CompositeSurrogate, fit_separable
from surreax.domain import Box, Domain, box, box_from_data
from surreax.export import DifferenceExport, GridExport, MaxAffineExport, export
from surreax.fitted import FittedSurrogate
from surreax.forms import (
    Affine,
    ChebyshevPolynomial,
    DifferenceOfMaxAffine,
    DifferenceOfMaxAffineForm,
    Grid,
    LinearForm,
    MaxAffine,
    MaxAffineForm,
    Network,
    NetworkForm,
)
from surreax.implicit import (
    FoldReport,
    ImplicitSurrogate,
    InverseMap,
    Objective,
    RefinedSurrogate,
    Residual,
    fit_data,
    fit_implicit,
    refine,
    root_find,
)
from surreax.reduction import active_subspace_inputs, pca_inputs, pca_outputs
from surreax.report import ErrorReport
from surreax.structure import Structure, analyze, prune_inputs

__all__ = [
    "Affine",
    "Box",
    "ChebyshevPolynomial",
    "CompositeSurrogate",
    "DifferenceExport",
    "DifferenceOfMaxAffine",
    "DifferenceOfMaxAffineForm",
    "Domain",
    "ErrorReport",
    "FittedSurrogate",
    "FoldReport",
    "Grid",
    "GridExport",
    "ImplicitSurrogate",
    "InverseMap",
    "LinearForm",
    "MaxAffine",
    "MaxAffineExport",
    "MaxAffineForm",
    "Network",
    "NetworkForm",
    "Objective",
    "RefinedSurrogate",
    "Residual",
    "Structure",
    "Target",
    "active_subspace_inputs",
    "analyze",
    "box",
    "box_from_data",
    "export",
    "fit",
    "fit_adaptive",
    "fit_data",
    "fit_implicit",
    "fit_separable",
    "pca_inputs",
    "pca_outputs",
    "prune_inputs",
    "refine",
    "root_find",
]
