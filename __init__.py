
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

from . import tables



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

# The first amino-acid will always be a Methionine for biological accuracy

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

    def descriptors(self) -> typing.Dict[str, float]:
        """Create a dictionary containing every protein descriptor available.
        """
        d = {}
        for prefix, method in self.__DESCRIPTORS.items():
            for i, x in enumerate(method(self)):
                d[f"{prefix}{i+1}"] = x
        return d

    def auto_correlation(
        self, table: typing.Dict[str, float], lag: int = 1, center: bool = True
    ) -> float:
        """Compute the auto-correlation index of a peptide sequence with Cruciani Formula 
        """
        # center the table if requested
        if center:
          mu = statistics.mean(table.values())
          sigma = statistics.stdev(table.values())
          table = {k: (v - mu) / sigma for k, v in table.items()}
        # build look up table
        lut = [table.get(aa, 0.0) for aa in self._CODE1]
        # compute using Cruciani formula
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
        # build the lookup table
        lut = [table.get(aa, 0.0) for aa in self._CODE1]
        # compute correlation using Cruciani formula
 
        v1 = numpy.take(lut, self.encoded[:-lag])
        v2 = numpy.take(lut, self.encoded[lag:])
        s = numpy.sum(v1*v2)
        return s / len(self)

    def profile(
        self,
        table: typing.Dict[str, float],
        window: int = 1,
        default: float = 0.0,
    ) -> typing.Sequence[float]:
        """Compute a generic per-residue profile from per-residue indices.
        """
        if window < 1:
            raise ValueError("Window must be strictly positive")

        # skip computing profile is window is larger than the available
        # number of residues in the peptide sequence
        if len(self) >= window:
            # build a look-up table and index values
            lut = [table.get(aa, default) for aa in self._CODE1]
            
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
    
    def counts(self) -> typing.Dict[str, int]:
        """Return a table of amino-acid counts in the peptide.
        """
    
        return {
            aa: self.sequence.count(aa)
            for aa in self._CODE1
        }

    def frequencies(self) -> typing.Dict[str, float]:
        """Return a table of amino-acid frequencies in the peptide.
        """
        return {
            aa:count/len(self)
            for aa,count in self.counts().items()
        }
    
    def aliphatic_index(self) -> float:
        """Compute the aliphatic index of the peptide.
        """
    
        # count aliphatic residues
        ala = self.sequence.count("A") / len(self.sequence)
        val = self.sequence.count("V") / len(self.sequence)
        leu = self.sequence.count("L") / len(self.sequence)
        ile = self.sequence.count("I") / len(self.sequence)
        # support unknown Leu/Ile residues
        xle = self.sequence.count("J") / len(self.sequence)
        # return aliphatic index
        return (ala + 2.9 * val + 3.9 * (leu + ile + xle)) * 100
    
    def boman(self) -> float:
        """Compute the Boman (potential peptide interaction) index.

        """
        return -_sum(self.profile(tables.BOMAN["Boman"])) / len(self)
    
    def charge(self, pH: float = 7, pKscale: str = "Lehninger") -> float:
        """Compute the theoretical net charge of a peptide sequence.
        """
        # get chosen the pKa scale
        scale_pKa = tables.PK.get(pKscale)

        if scale_pKa is None:
            raise ValueError(f"Invalid pK scale: {scale!r}")
        scale_sign = tables.CHARGE["sign"]

        # build a look-up table for the pKa scale and the charge sign
        lut_pKa = [scale_pKa.get(aa, 0.0) for aa in self._CODE1]
        lut_sign = [scale_sign.get(aa, 0.0) for aa in self._CODE1]

        # compute charge of each amino-acid, and sum

        pKa = numpy.take(lut_pKa, self.encoded)
        sign = numpy.take(lut_sign, self.encoded)
        charge = numpy.sum(sign / (1.0 + 10**(sign * (pH - pKa))))

        # add charge for C-terminal and N-terminal ends of the peptide
        if "nTer" in scale_pKa:
            charge += 1.0 / (1.0 + 10 ** (pH - scale_pKa["nTer"]))
        if "cTer" in scale_pKa:
            charge += -1.0 / (1.0 + 10 ** (scale_pKa["cTer"] - pH))

        # return the net protein charge
        return charge
    
    def hydrophobic_moment(self, window: int = 11, angle: int = 100) -> float:

        """Compute the maximal hydrophobic moment of a protein sequence.
        """
        window = min(window, len(self))
        scale = tables.HYDROPHOBICITY["Eisenberg"]
        lut = [scale.get(aa, 0.0) for aa in self._CODE1]
        angles = [(angle * i) % 360 for i in range(window)]

        if numpy is None:
            angsin = [math.sin(math.radians(theta)) for theta in angles]
            angcos = [math.cos(math.radians(theta)) for theta in angles]
        else:
            angsin = numpy.sin(numpy.radians(angles))
            angcos = numpy.cos(numpy.radians(angles))

        maxnorm = 0.0
        for i in range(len(self.sequence) - window + 1):
            # compute sin and cos of angles
            if numpy is None:
                sumsin = sumcos = 0
                for aa, s, c in zip(self.encoded[i:i+window], angsin, angcos):
                    sumsin += lut[aa]*s
                    sumcos += lut[aa]*c
            else:
                hvec = numpy.take(lut, self.encoded[i:i+window])
                sumsin = numpy.sum(hvec * angsin)
                sumcos = numpy.sum(hvec * angcos)
            # compute only the distance component (this way we can avoid
            # computing the square root in each iteration)
            norm = sumsin**2 + sumcos**2
            if norm > maxnorm:
                maxnorm = norm

        # compute the angular moment from the norm
        return math.sqrt(maxnorm) / window

    def hydrophobicity(self, scale: str = "KyteDoolittle") -> float:
        """Compute the hydrophobicity index of a protein sequence.
        """
        table = tables.HYDROPHOBICITY.get(scale)
        if table is None:
            raise ValueError(f"Invalid hydrophobicity scale: {scale!r}")
        return _sum(self.profile(table)) / len(self)
    
    def instability_index(self) -> float:
        
        """Compute the instability index of a protein sequence.
        """
        scale = tables.INSTABILITY["Guruprasad"]
        gp = sum(scale.get(self.sequence[i : i + 2], 1.0) for i in range(len(self.sequence) - 1))
        return gp * 10 / (len(self.sequence))
    
    def isoelectric_point(self, pKscale: str = "EMBOSS") -> float:
        """Compute the isoelectric point of a protein sequence.
        """
        # use a simple bissecting loop to minimize the charge function
        top, bottom, x = 0.0, 14.0, 7.0
        while not math.isclose(top, bottom):
            x = (top + bottom) / 2
            c = self.charge(pH=x, pKscale=pKscale)
            if c >= 0:
                top = x
            if c <= 0:
                bottom = x
        return x
    def mass_shift(
        self,
        aa_shift: typing.Union[str, typing.Dict[str, float], None] = "silac_13c",
        monoisotopic: bool = True,
    ) -> float:
        """Compute the mass difference of modified peptides.
        """
        if isinstance(aa_shift, str):
            table = tables.MASS_SHIFT.get(aa_shift)
            if table is None:
                raise ValueError(f"Invalid mass shift scale: {aa_shift!r}")
            scale = {}
            if aa_shift == "silac_13c":
                scale["K"] = table["K"] - 0.064229 * (not monoisotopic)
                scale["R"] = table["R"] - 0.064229 * (not monoisotopic)
            elif aa_shift == "silac_13c15n":
                scale["K"] = table["K"] - 0.071499 * (not monoisotopic)
                scale["R"] = table["R"] - 0.078669 * (not monoisotopic)
            elif aa_shift == "15n":
                for k, v in table.items():
                    scale[k] = v * 0.997035 - 0.003635 * (not monoisotopic)
        elif isinstance(aa_shift, dict):
            scale = aa_shift
        else:
            raise TypeError(
                f"Expected str or dict, found {aa_shift.__class__.__name__}"
            )

        # sum the mass-shift of each amino acid
        shift = _sum(self.profile(scale))
        # return the shift with C-terminal and N-terminal ends
        return shift + scale.get("nTer", 0.0) + scale.get("cTer", 0.0)

    
    def molecular_weight(
        self,
        average: str = "expasy",
        aa_shift: typing.Union[str, typing.Dict[str, float], None] = None,
    ) -> float:
        
        """Compute the molecular weight of a protein sequence.
        """
        scale = tables.MOLECULAR_WEIGHT.get(average)
        if scale is None:
            raise ValueError(f"Invalid average weight scale: {average!r}")

        # sum the weight of each amino acid and add weight of water molecules
        mass = _sum(self.profile(scale)) + scale["H2O"]
        # add mass shift for labeled proteins
        if aa_shift is not None:
            mass += self.mass_shift(
                aa_shift=aa_shift, monoisotopic=average == "monoisotopic"
            )

        return mass
    
    def mz(
        self,
        charge: int = 2,
        aa_shift: typing.Union[str, typing.Dict[str, float], None] = None,
        cysteins: float = 57.021464,
    ) -> float:
        """Compute the m/z (mass over charge) ratio for a peptide.
        """
        if not isinstance(charge, int):
            raise TypeError(f"Expected int, found {charge.__class__.__name__!r}")

        # compute the mass of the uncharged peptide
        mass = self.molecular_weight(average="monoisotopic", aa_shift=aa_shift)
        # add modification at cysteins
        mass += self.sequence.count("C") * cysteins
        # modify for charged peptides
        if charge >= 0:
            mass += charge * 1.007276  # weights of H+1 ions
            mass /= charge  # divide by charge state

        return mass

    def hydrophobicity_profile(
        self, window: int = 11, scale: str = "KyteDoolittle"
    ) -> typing.Sequence[float]:
        """Build a hydrophobicity profile of a sliding window.
        """
        if scale not in tables.HYDROPHOBICITY:
            raise ValueError(f"Invalid hydrophobicity scale: {scale!r}")

        profile = array.array('f')
        for i in range(len(self.sequence) - window + 1):
            profile.append(self[i : i + window].hydrophobicity(scale=scale))

        return profile

    def hydrophobic_moment_profile(
        self, window: int = 11, angle: int = 100
    ) -> typing.Sequence[float]:
        """Build a hydrophobic moment profile of a sliding window.

        """
        profile = array.array("f")
        for i in range(len(self.sequence) - window + 1):
            profile.append(
                self[i : i + window].hydrophobic_moment(window=window - 1, angle=angle)
            )

        return profile

    def membrane_position_profile(
        self, window: int = 11, angle: int = 100
    ) -> typing.List[str]:
        """Compute the theoretical class of a protein sequence
     
        """
        profile_H = self.hydrophobicity_profile(window=window, scale="Eisenberg")
        profile_uH = self.hydrophobic_moment_profile(window=window, angle=angle)

        profile = []
        for h, uh in zip(profile_H, profile_uH):
            m = h * -0.421 + 0.579
            if uh <= m:
                if h >= 0.5:
                    profile.append("T")
                else:
                    profile.append("G")
            else:
                profile.append("S")

        return profile

   