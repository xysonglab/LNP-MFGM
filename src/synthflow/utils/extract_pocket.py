"""From https://github.com/SeonghwanSeo/PharmacoNet"""

from pathlib import Path

import numpy as np
from Bio.PDB import PDBIO, PDBParser
from Bio.PDB.PDBIO import Select
from numpy.typing import ArrayLike
from rdkit import Chem

# fmt: off
AMINO_ACID = [
    "ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",
    "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR",
]
# fmt: on


def get_mol_coords(mol_path: str | Path) -> np.ndarray:
    format = Path(mol_path).suffix
    if format == ".sdf":
        mol = next(Chem.SDMolSupplier(str(mol_path), sanitize=False))
    elif format == ".mol2":
        mol = Chem.MolFromMol2File(str(mol_path), sanitize=False)
    elif format == ".pdb":
        mol = Chem.MolFromPDBFile(str(mol_path), sanitize=False)
    else:
        raise ValueError(mol_path)
    coords = mol.GetConformer().GetPositions()
    return np.array(coords)


def get_mol_center(mol_path: str | Path) -> tuple[float, float, float]:
    coords = get_mol_coords(mol_path)
    x, y, z = np.mean(coords, 0).tolist()
    return x, y, z


class CenterSelect(Select):
    def __init__(self, center: ArrayLike, cutoff: float):
        self.center = np.array(center)
        self.cutoff = cutoff

    def accept_residue(self, residue):
        if super().accept_residue(residue) == 0:
            return 0
        if residue.get_resname() not in AMINO_ACID:
            return 0
        c_alpha_pos = np.array(residue["CA"].coord)
        dis = np.linalg.norm(c_alpha_pos - self.center, axis=-1)
        return int(dis < self.cutoff)


class LigandSelect(Select):
    def __init__(self, coords: ArrayLike, cutoff: float, use_calpha=False):
        self.coords = np.array(coords)
        self.cutoff = cutoff
        self.use_calpha = False

    def accept_residue(self, residue):
        if super().accept_residue(residue) == 0:
            return 0
        if residue.get_resname() not in AMINO_ACID:
            return 0

        if self.use_calpha:
            c_alpha_pos = np.array(residue["CA"].coord).reshape(1, -1)
            min_dis = np.min(np.linalg.norm(c_alpha_pos - self.coords, axis=-1))
        else:
            # fast cutoff with calpha pos
            c_alpha_pos = np.array(residue["CA"].coord).reshape(1, -1)
            min_dis = np.min(np.linalg.norm(c_alpha_pos - self.coords, axis=-1))
            if min_dis > self.cutoff + 20.0:
                return 0

            # intensive cutoff
            residue_positions = np.array(
                [list(atom.get_vector()) for atom in residue.get_atoms() if "H" not in atom.get_id()]
            )
            pairwise_dis = np.linalg.norm(residue_positions.reshape(1, -1, 3) - self.coords.reshape(-1, 1, 3), axis=-1)
            min_dis = np.min(pairwise_dis)
        return int(min_dis <= self.cutoff)


def extract_pocket_from_center(
    protein_path: str | Path,
    out_pocket_path: str | Path | None = None,
    center: tuple[float, float, float] | None = None,
    ref_ligand_path: str | Path | None = None,
    cutoff: float = 15.0,
    force_pocket_extract: bool = False,
) -> Path:
    protein_path = Path(protein_path)
    if out_pocket_path is None:
        if center is not None:
            name = "pocket_" + protein_path.stem + "_" + f"{center}" + f"_{cutoff}A.pdb"
            out_pocket_path = protein_path.parent / name
        else:
            assert ref_ligand_path is not None
            ref_ligand_path = Path(ref_ligand_path)
            name = "pocket_" + protein_path.stem + "_" + ref_ligand_path.stem + f"_{cutoff}A.pdb"
            out_pocket_path = protein_path.parent / name

    out_pocket_path = Path(out_pocket_path)
    if out_pocket_path.exists() and (not force_pocket_extract):
        return out_pocket_path
    if center is None:
        assert ref_ligand_path is not None
        center = get_mol_center(ref_ligand_path)

    parser = PDBParser()
    structure = parser.get_structure("protein", str(protein_path))
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(out_pocket_path), CenterSelect(center, cutoff))

    return out_pocket_path


def extract_pocket_from_ref_ligand(
    protein_path: str | Path,
    ref_ligand_path: str | Path,
    out_pocket_path: str | Path,
    cutoff: float,
    use_calpha: bool,
    force_pocket_extract: bool = False,
):
    out_pocket_path = Path(out_pocket_path)
    if (not out_pocket_path.exists()) or force_pocket_extract:
        parser = PDBParser()
        structure = parser.get_structure("protein", str(protein_path))
        io = PDBIO()
        io.set_structure(structure)
        ref_lig_coords = get_mol_coords(ref_ligand_path)
        io.save(str(out_pocket_path), LigandSelect(ref_lig_coords, cutoff, use_calpha=use_calpha))
