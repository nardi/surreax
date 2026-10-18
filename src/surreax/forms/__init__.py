"""Model forms and the protocol they satisfy."""

from surreax.forms.affine import Affine
from surreax.forms.difference_max_affine import DifferenceOfMaxAffine, DifferenceOfMaxAffineForm
from surreax.forms.grid import Grid
from surreax.forms.max_affine import MaxAffine, MaxAffineForm
from surreax.forms.network import Network, NetworkForm
from surreax.forms.polynomial import ChebyshevPolynomial
from surreax.forms.protocol import LinearForm
from surreax.forms.relu_network import (
    ConvexNetwork,
    ConvexNetworkForm,
    ReluNetwork,
    ReluNetworkForm,
)

Form = (
    LinearForm
    | MaxAffineForm
    | DifferenceOfMaxAffineForm
    | NetworkForm
    | ConvexNetworkForm
    | ReluNetworkForm
)
"""Every request `fit` accepts."""

__all__ = [
    "Affine",
    "ChebyshevPolynomial",
    "ConvexNetwork",
    "ConvexNetworkForm",
    "DifferenceOfMaxAffine",
    "DifferenceOfMaxAffineForm",
    "Form",
    "Grid",
    "LinearForm",
    "MaxAffine",
    "MaxAffineForm",
    "Network",
    "NetworkForm",
    "ReluNetwork",
    "ReluNetworkForm",
]
