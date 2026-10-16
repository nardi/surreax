"""Surrogate model training for JAX functions.

`fit` takes a JAX target, a domain to be accurate over, and a model form, and
returns a `FittedSurrogate` that keeps the chosen form.
"""

from surreax.api import Target, fit, fit_adaptive
from surreax.autoencoder import (
    Autoencoder,
    AutoencoderSolve,
    AutoencoderSolver,
    DecodedSurrogate,
    LatentGuess,
)
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
from surreax.hyperreduction import HyperReducedSolver, LinearDecoder, hyper_reduce
from surreax.implicit import (
    ExactSolve,
    FoldReport,
    ImplicitSurrogate,
    InverseMap,
    LatentReport,
    Objective,
    ReducedSolve,
    ReducedSolver,
    RefinedSurrogate,
    Residual,
    autoencoder_solver,
    fit_data,
    fit_implicit,
    recommend_linear_solver,
    recommend_method,
    recommend_reduced_method,
    reduced_solver,
    refine,
    root_find,
)
from surreax.reduction import active_subspace_inputs, pca_inputs, pca_outputs
from surreax.report import ErrorReport
from surreax.sparse_solve import LinearSolver, residual_pattern
from surreax.stencil import StencilResidual
from surreax.strategy import (
    function_kinds,
    linear_energy,
    recommend_block_forms,
    recommend_form,
)
from surreax.structure import Structure, analyze, prune_inputs

__all__ = [
    "Affine",
    "Autoencoder",
    "AutoencoderSolve",
    "AutoencoderSolver",
    "Box",
    "ChebyshevPolynomial",
    "CompositeSurrogate",
    "DecodedSurrogate",
    "DifferenceExport",
    "DifferenceOfMaxAffine",
    "DifferenceOfMaxAffineForm",
    "Domain",
    "ErrorReport",
    "ExactSolve",
    "FittedSurrogate",
    "FoldReport",
    "Grid",
    "GridExport",
    "HyperReducedSolver",
    "ImplicitSurrogate",
    "InverseMap",
    "LatentGuess",
    "LatentReport",
    "LinearDecoder",
    "LinearForm",
    "LinearSolver",
    "MaxAffine",
    "MaxAffineExport",
    "MaxAffineForm",
    "Network",
    "NetworkForm",
    "Objective",
    "ReducedSolve",
    "ReducedSolver",
    "RefinedSurrogate",
    "Residual",
    "StencilResidual",
    "Structure",
    "Target",
    "active_subspace_inputs",
    "analyze",
    "autoencoder_solver",
    "box",
    "box_from_data",
    "export",
    "fit",
    "fit_adaptive",
    "fit_data",
    "fit_implicit",
    "fit_separable",
    "function_kinds",
    "hyper_reduce",
    "linear_energy",
    "pca_inputs",
    "pca_outputs",
    "prune_inputs",
    "recommend_block_forms",
    "recommend_form",
    "recommend_linear_solver",
    "recommend_method",
    "recommend_reduced_method",
    "reduced_solver",
    "refine",
    "residual_pattern",
    "root_find",
]
