"""Model forms and the protocol they satisfy."""

from surreax.forms.affine import Affine
from surreax.forms.difference_max_affine import DifferenceOfMaxAffine, DifferenceOfMaxAffineForm
from surreax.forms.max_affine import MaxAffine, MaxAffineForm
from surreax.forms.polynomial import ChebyshevPolynomial
from surreax.forms.protocol import LinearForm

__all__ = [
    "Affine",
    "ChebyshevPolynomial",
    "DifferenceOfMaxAffine",
    "DifferenceOfMaxAffineForm",
    "LinearForm",
    "MaxAffine",
    "MaxAffineForm",
]
