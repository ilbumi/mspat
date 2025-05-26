"""Information on residues."""

MASK_TOKEN = "[MASK]"  # Mask token for padding  # noqa: S105
DUMMY_TOKEN = "[DUM]"  # Dummy token for padding  # noqa: S105
RESIDUE_3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLU": "E",
    "GLN": "Q",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",  # Selenocysteine (Sec)
    "PYL": "O",  # Pyrrolysine (Pyl)
    "ASX": "B",  # Aspartic acid or Asparagine
    "GLX": "Z",  # Glutamic acid or Glutamine
    "XAA": "X",  # Unknown or 'other' amino acid
}
RESIDUE_1_TO_3 = {v: k for k, v in RESIDUE_3_TO_1.items()}
RESIDUES_VOCAB = [*sorted(RESIDUE_3_TO_1.values()), DUMMY_TOKEN, MASK_TOKEN]
RESIDUE_TO_INDEX = {name: i for i, name in enumerate(RESIDUES_VOCAB)}

BACKBONE_ATOM_NAMES: list[str] = ["N", "CA", "C", "O"]
RESIDUE_TO_SIDECHAIN_ATOM_NAMES: dict[str, list[str]] = {
    "ALA": ["CB"],
    "ARG": ["CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "ASN": ["CB", "CG", "OD1", "ND2"],
    "ASP": ["CB", "CG", "OD1", "OD2"],
    "CYS": ["CB", "SG"],
    "GLU": ["CB", "CG", "CD", "OE1", "OE2"],
    "GLN": ["CB", "CG", "CD", "OE1", "NE2"],
    "GLY": [],
    "HIS": ["CB", "CG", "ND1", "CD2", "CE1", "NE2"],
    "ILE": ["CB", "CG1", "CG2", "CD1"],
    "LEU": ["CB", "CG", "CD1", "CD2"],
    "LYS": ["CB", "CG", "CD", "CE", "NZ"],
    "MET": ["CB", "CG", "SD", "CE"],
    "PHE": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "PRO": ["CB", "CG", "CD"],
    "SER": ["CB", "OG"],
    "THR": ["CB", "OG1", "CG2"],
    "TRP": ["CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "TYR": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "VAL": ["CB", "CG1", "CG2"],
    "SEC": ["CB", "SG"],
    "PYL": ["CB", "CG", "CD", "CE", "NZ"],
    "ASX": ["CB", "CG", "OD1", "ND2"],
    "GLX": ["CB", "CG", "CD", "OE1", "NE2"],
    "XAA": [],
}
RESIDUE_TO_SIDECHAIN_ATOM_NAMES.update({RESIDUE_3_TO_1[k]: v for k, v in RESIDUE_TO_SIDECHAIN_ATOM_NAMES.items()})

ATOM_NAMES_VOCAB = (
    BACKBONE_ATOM_NAMES
    + sorted({an for ans in RESIDUE_TO_SIDECHAIN_ATOM_NAMES.values() for an in ans})
    + ["VCB", "X", DUMMY_TOKEN, MASK_TOKEN]  # virtual CB atom for GLY
)
ATOM_NAMES_TO_INDEX = {name: i for i, name in enumerate(ATOM_NAMES_VOCAB)}
ATOM_NAMES_TO_INDEX["VCB"] = ATOM_NAMES_TO_INDEX["CB"]  # PRETEND GLY HAS CB

ACIDIC_GROUPS = {
    "ASP": ["CG", "OD1", "OD2"],
    "GLU": ["CD", "OE1", "OE2"],
}
BASIC_GROUPS = {
    "ARG": ["CZ"],
    "LYS": ["NZ"],
    "HIS": ["CG", "ND1", "CD2", "CE1", "NE2"],
}
AROMATIC_RESIDUES = {
    "HIS": ["CG", "ND1", "CD2", "CE1", "NE2"],
    "TYR": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "TRP": ["CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
}
HBOND_ACCEPTORS = {
    "ASN": ["OD1"],
    "ASP": ["OD1", "OD2"],
    "GLN": ["OE1"],
    "GLU": ["OE1", "OE2"],
    "HIS": ["ND1"],
    "SER": ["OG"],
    "THR": ["OG1"],
    "TYR": ["OH"],
    "MET": ["SD"],
}
HBOND_DONORS = {
    "ARG": ["NE", "NH1", "NH2"],
    "CYS": ["SG"],
    "GLN": ["NE2"],
    "HIS": ["NE2, ND1"],
    "LYS": ["NZ"],
    "SER": ["OG"],
    "THR": ["OG1"],
    "TRP": ["NE1"],
    "TYR": ["OH"],
}
HYDROPHOBIC_RESIDUES = {
    "ILE",
    "VAL",
    "LEU",
    "PHE",
    "CYS",
    "MET",
    "ALA",
    "TRP",
}
