
"""Physicochemical properties and indices for amino-acid sequences.
"""
import array
import collections
import functools
import math
import operator
import random
import statistics
import typing
import numpy

class BLOSUMIndices(typing.NamedTuple):

""" BLOSUM indices were derived of physicochemical properties that have
    been subjected to a VARIMAX analysis and an alignment matrix of the
    20 natural AAs using the BLOSUM62 matrix.

    """
    blosum1: float
    blosum2: float
    blosum3: float
    blosum4: float
    blosum5: float
    blosum6: float
    blosum7: float
    blosum8: float
    blosum9: float
    blosum10: float

class CrucianiProperties(typing.NamedTuple):
    """The Cruciani properties are a collection of scaled principal
    component scores that summarize a broad set of descriptors
    calculated based on the interaction of each amino acid residue with
    several chemical groups (or "probes"), such as charged ions, methyl,
    hydroxyl groups, and so forth."""
    pp1: float
    pp2: float
    pp3: float


class FasgaiVectors(typing.NamedTuple):
    """ The FASGAI vectors (Factor Analysis Scales of Generalized Amino
    Acid Information) are a set of amino acid descriptors, that reflect
    hydrophobicity, alpha and turn propensities, bulky properties,
    compositional characteristics, local flexibility, and electronic
    properties, that can be utilized to represent the sequence
    structural features of peptides or protein motifs.
    """
    f1: float
    f2: float
    f3: float
    f4: float
    f5: float
    f6: float


class KideraFactors(typing.NamedTuple):
    """ The Kidera Factors were originally derived by applying multivariate
    analysis to 188 physical properties of the 20 amino acids and using
    dimension reduction techniques.
    """
    kf1: float
    kf2: float
    kf3: float
    kf4: float
    kf5: float
    kf6: float
    kf7: float
    kf8: float
    kf9: float
    kf10: float


class MSWHIMScores(typing.NamedTuple):
    """MS-WHIM scores were derived from 36 electrostatic potential
    properties derived from the three-dimensional structure of the
    20 natural amino acids.
    """
    mswhim1: float
    mswhim2: float
    mswhim3: float


class PhysicalDescriptors(typing.NamedTuple):
    """The PP descriptors were constructed by improving on existing
    PCA-derived descriptors
    """
    pd1: float
    pd2: float


class PCPDescriptors(typing.NamedTuple):
    """The Physical-Chemical Properties descriptors of a peptide.
    """
    e1: float
    e2: float
    e3: float
    e4: float
    e5: float


class PRINComponents(typing.NamedTuple):
    """The PRIN components of a peptide.
    """
    prin1: float
    prin2: float
    prin3: float


class ProtFPDescriptors(typing.NamedTuple):
    """The ProtFP descriptors of a peptide.
    The ProtFP set was constructed from a large initial selection of
    indices obtained from the `AAindex <https://www.genome.jp/aaindex/>`_
    database for all 20 naturally occurring amino acids.

    """
    protfp1: float
    protfp2: float
    protfp3: float
    protfp4: float
    protfp5: float
    protfp6: float
    protfp7: float
    protfp8: float


class SneathVectors(typing.NamedTuple):
    """The Sneath vectors of a peptide.
    """
    sv1: float
    sv2: float
    sv3: float
    sv4: float


class STScales(typing.NamedTuple):
    """The ST-scales of a peptide.
    """
    st1: float
    st2: float
    st3: float
    st4: float
    st5: float
    st6: float
    st7: float
    st8: float


class SVGERDescriptors(typing.NamedTuple):
    """The SVGER descriptors of a peptide.
    """
    svger1: float
    svger2: float
    svger3: float
    svger4: float
    svger5: float
    svger6: float
    svger7: float
    svger8: float
    svger9: float
    svger10: float
    svger11: float


class TScales(typing.NamedTuple):
    """The T-scales of a peptide.
    """
    t1: float
    t2: float
    t3: float
    t4: float
    t5: float


class VHSEScales(typing.NamedTuple):
    """The VHSE-scales of a peptide.
    """
    vhse1: float
    vhse2: float
    vhse3: float
    vhse4: float
    vhse5: float
    vhse6: float
    vhse7: float
    vhse8: float


class VSTPVDescriptors(typing.NamedTuple):
    """The VSTPV descriptors of a peptide.
    """
    vstpv1: float
    vstpv2: float
    vstpv3: float
    vstpv4: float
    vstpv5: float
    vstpv6: float


class ZScales(typing.NamedTuple):
    """The Z-scales of a peptide.

    """
    z1: float
    z2: float
    z3: float
    z4: float
    z5: float

class Peptide(typing.Sequence[str]):
    """A sequence of amino acids.
    """
    # fmt: off
    _CODE1 = [
        "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
        "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
        "O", "U", "B", "Z", "J", "X"
    ]

    # fmt: off
    _CODE3 = [
        "Ala", "Arg", "Asn", "Asp", "Cys", "Gln", "Glu", "Gly", "His", "Ile",
        "Leu", "Lys", "Met", "Phe", "Pro", "Ser", "Thr", "Trp", "Tyr", "Val",
        "Pyl", "Sec", "Asx", "Glx", "Xle", "Xaa"
    ]
