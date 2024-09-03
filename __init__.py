
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
    @classmethod
    def sample(
        cls,
        length: int,
        frequencies: str = "SwissProt2021",
    ) -> "Peptide":
        """Generate a peptide with the given amino-acid frequencies.
        """
        table = tables.AA_FREQUENCIES.get(frequencies)
        if table is None:
            raise ValueError(f"Invalid amino acid frequencies: {frequencies!r}")

        if length == 0:
            return cls("")

        cumfreq = 0
        cumulative_frequencies = {}
        for k,v in table.items():
            cumfreq += v
            cumulative_frequencies[k] = cumfreq

        residues = ["M"]
        for i in range(1, length):
            x = random.random()
            r = next((k for k,v in cumulative_frequencies.items() if x <= v), "X")
            residues.append(r)

        return cls("".join(residues))

    def __init__(self, sequence: str) -> None:
        """Create a new peptide object with the given sequence.
        """
        # store the sequence in text format
        self.sequence: str = sequence
        # store an encoded version of the sequence as an array of indices
        encoder = {aa:i for i,aa in enumerate(self._CODE1)}
        self.encoded = array.array('B')
        for i, aa in enumerate(sequence):
            self.encoded.append(encoder.get(aa, encoder["X"]))

    def __len__(self) -> int:
        return len(self.sequence)

    @typing.overload
    def __getitem__(self, index: slice) -> "Peptide":
        pass

    @typing.overload
    def __getitem__(self, index: int) -> str:
        pass

    def __getitem__(
        self, index: typing.Union[int, slice]
    ) -> typing.Union[str, "Peptide"]:
        if isinstance(index, slice):
            return Peptide(self.sequence[index])
        return self.sequence[index]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.sequence!r})"

    def descriptors(self) -> typing.Dict[str, float]:
        """Create a dictionary containing every protein descriptor available.
        Example:
            >>> peptide = Peptide("SDKEVDEVDAALSDLEITLE")
            >>> sorted(peptide.descriptors().keys())
            ['BLOSUM1', ..., 'F1', ..., 'KF1', ..., 'MSWHIM1', ..., 'PP1', ...]
        Hint:
            Use this method to create a `~pandas.DataFrame` containing the
            descriptors for several sequences.
        """
        d = {}
        for prefix, method in self.__DESCRIPTORS.items():
            for i, x in enumerate(method(self)):
                d[f"{prefix}{i+1}"] = x
        return d

    def auto_correlation(
        self, table: typing.Dict[str, float], lag: int = 1, center: bool = True
    ) -> float:
        """Compute the auto-correlation index of a peptide sequence.

        Example:
            >>> peptide = Peptide("SDKEVDEVDAALSDLEITLE")
            >>> table = peptides.tables.HYDROPHOBICITY["KyteDoolittle"]
            >>> peptide.auto_correlation(table=table)
            -0.3519908...
            >>> peptide.auto_correlation(table=table, lag=5)
            0.00113355...
        """
        # center the table if requested
        if center:
            mu = statistics.mean(table.values())
            sigma = statistics.stdev(table.values())
            table = {k: (v - mu) / sigma for k, v in table.items()}
        # build look up table
        lut = [table.get(aa, 0.0) for aa in self._CODE1]
        # compute using Cruciani formula
        if numpy is None:
            s1 = s2 = 0.0
            for aa1, aa2 in zip(self.encoded[:-lag], self.encoded[lag:]):
                s1 += lut[aa1] * lut[aa2]
                s2 += lut[aa1] ** 2
        else:
            v1 = numpy.take(lut, self.encoded[:-lag])
            v2 = numpy.take(lut, self.encoded[lag:])
            s1 = numpy.sum(v1*v2)
            s2 = numpy.sum(v1**2)
        return s1 / s2

    def auto_covariance(
        self, table: typing.Dict[str, float], lag: int = 1, center: bool = True
    ) -> float:
        """Compute the auto-covariance index of a peptide sequence.
        """
        # center the table if requested
        if center:
            mu = statistics.mean(table.values())
            sigma = statistics.stdev(table.values())
            table = {k: (v - mu) / sigma for k, v in table.items()}
        # build the lookup table
        lut = [table.get(aa, 0.0) for aa in self._CODE1]
        # compute correlation using Cruciani formula
        if numpy is None:
            s = 0.0
            for aa1, aa2 in zip(self.encoded[:-lag], self.encoded[lag:]):
                s += lut[aa1] * lut[aa2]
        else:
            v1 = numpy.take(lut, self.encoded[:-lag])
            v2 = numpy.take(lut, self.encoded[lag:])
            s = numpy.sum(v1*v2)
        return s / len(self)

    def cross_covariance(
        self,
        table1: typing.Dict[str, float],
        table2: typing.Dict[str, float],
        lag: int = 1,
        center: bool = True,
    ) -> float:
        """Compute the cross-covariance index of a peptide sequence.
        """
        # center the tables if requested
        if center:
            mu1 = statistics.mean(table1.values())
            sigma1 = statistics.stdev(table1.values())
            table1 = {k: (v - mu1) / sigma1 for k, v in table1.items()}
            mu2 = statistics.mean(table2.values())
            sigma2 = statistics.stdev(table2.values())
            table2 = {k: (v - mu2) / sigma2 for k, v in table2.items()}

        # build the lookup table
        lut1 = [table1.get(aa, 0.0) for aa in self._CODE1]
        lut2 = [table2.get(aa, 0.0) for aa in self._CODE1]

        # compute using Cruciani formula
        if numpy is None:
            s = 0.0
            for aa1, aa2 in zip(self.encoded[:-lag], self.encoded[lag:]):
                s += lut1[aa1] * lut2[aa2]
        else:
            v1 = numpy.take(lut1, self.encoded[:-lag])
            v2 = numpy.take(lut2, self.encoded[lag:])
            s = numpy.sum(v1*v2)
        return s / len(self)

    def profile(
        self,
        table: typing.Dict[str, float],
        window: int = 1,
        default: float = 0.0,
    ) -> typing.Sequence[float]:
        """Compute a generic per-residue profile from per-residue indices.
        Arguments:
            table (`dict`): The values per residue to apply to the whole
                protein sequence.
            window (`int`): The window size for computing the profile.
                Leave as *1* to return per-residue values.
            default (`float`): The default value to use for amino-acids
                that are not present in the given table.

        Returns:
            `collections.abc.Sequence` of `float`: The per-residue profile
            values, averaged in the given window size. When ``window`` is
            larger than the available number of resiudes, an empty sequence
            is returned.

        Example:
            >>> peptide = Peptide("PKLVCLKKC")
            >>> peptide.profile(peptides.tables.CHARGE['sign'])
            [0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 1.0, 1.0, -1.0]
            >>> peptide.profile(peptides.tables.MOLECULAR_WEIGHT['expasy'], 5)
            [108..., 111..., 111..., 114..., 115...]

        .. versionadded:: 0.3.0
        """
        if window < 1:
            raise ValueError("Window must be strictly positive")

        # skip computing profile is window is larger than the available
        # number of residues in the peptide sequence
        if len(self) >= window:
            # build a look-up table and index values
            lut = [table.get(aa, default) for aa in self._CODE1]
            if numpy is None:
                values = [lut[i] for i in self.encoded]
            else:
                values = numpy.take(lut, self.encoded)  # type: ignore
            # don't perform window averaging if window is 1
            if window <= 1:
                return list(values)
            elif window > 1:
                p = []
                # use a rolling sum over the window
                s = 0.0
                for i in range(window):
                    s += values[i]
                for j in range(window, len(self)):
                    p.append(s / window)
                    s -= values[j-window]
                    s += values[j]
                p.append(s / window)
        else:
            p = []

        return p
