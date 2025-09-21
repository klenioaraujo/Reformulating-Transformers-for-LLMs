# Make the insect specimens easily importable

from .base_specimen import PsiQRHBase
from .chrysopidae import Chrysopidae_PsiQRH
from .tettigoniidae import Tettigoniidae_PsiQRH
from .camponotus import Camponotus_PsiQRH
from .apis_mellifera import ApisMellifera_PsiQRH

__all__ = [
    "PsiQRHBase",
    "Chrysopidae_PsiQRH",
    "Tettigoniidae_PsiQRH",
    "Camponotus_PsiQRH",
    "ApisMellifera_PsiQRH"
]
