from __future__ import annotations

import copy
import pickle
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Generic, Self, TypeVar

import numpy as np
import torch
from rdkit import Chem
from scipy.spatial.transform import Rotation

import cgflow.util.functional as smolF
import cgflow.util.rdkit as smolRD
from cgflow.util.rdkit import embed_mol
from cgflow.util.tokeniser import Vocabulary

# Type aliases
_T = torch.Tensor
TDevice = torch.device | str
TCoord = tuple[float, float, float]

# Generics
TSmolMol = TypeVar("TSmolMol", bound="SmolMol")

# Constants
PICKLE_PROTOCOL = 4


# **********************
# *** Util functions ***
# **********************


def _check_type(obj, obj_type, name="object"):
    if not isinstance(obj, obj_type):
        raise TypeError(f"{name} must be an instance of {obj_type} or one of its subclasses, got {type(obj)}")


def _check_shape_len(tensor, allowed, name="object"):
    num_dims = len(tensor.size())
    allowed = [allowed] if isinstance(allowed, int) else allowed
    if num_dims not in allowed:
        raise RuntimeError(f"Number of dimensions of {name} must be in {str(allowed)}, got {num_dims}")


def _check_shapes_equal(t1, t2, dims=None):
    if dims is None:
        if t1.size() != t2.size():
            raise RuntimeError(f"objects must have the same shape, got {t1.shape} and {t2.shape}")
        else:
            return

    if isinstance(dims, int):
        dims = [dims]

    t1_dims = [t1.size(dim) for dim in dims]
    t2_dims = [t2.size(dim) for dim in dims]
    if t1_dims != t2_dims:
        raise RuntimeError(f"Expected dimensions {str(dims)} to match, got {t1.size()} and {t2.size()}")


def _check_dim_shape(tensor, dim, allowed, name="object"):
    shape = tensor.size(dim)
    allowed = [allowed] if isinstance(allowed, int) else allowed
    if shape not in allowed:
        raise RuntimeError(f"Shape of {name} for dim {dim} must be in {allowed}")


def _check_dict_key(map, key, dict_name="dictionary"):
    if key not in map:
        raise RuntimeError(f"{dict_name} must contain key {key}")


# *************************
# *** MolRepr Interface ***
# *************************


class SmolMol(ABC):
    """Interface for molecule representations for the Smol library"""

    def __init__(self, str_id: str):
        self._str_id = str_id

    # *** Properties for molecule objects ***

    @property
    def str_id(self):
        return self.__str__()

    @property
    @abstractmethod
    def device(self) -> torch.device:
        pass

    @property
    @abstractmethod
    def seq_length(self) -> int:
        pass

    # *** Static constructor methods ***

    @staticmethod
    @abstractmethod
    def from_bytes(data: bytes) -> SmolMol:
        pass

    @staticmethod
    @abstractmethod
    def from_rdkit(rdkit, mol: Chem.rdchem.Mol, *args) -> SmolMol:
        pass

    # *** Conversion functions for molecule objects ***

    @abstractmethod
    def to_bytes(self) -> bytes:
        pass

    @abstractmethod
    def to_rdkit(self, *args) -> Chem.rdchem.Mol:
        pass

    # *** Other functionality for molecule objects ***

    @abstractmethod
    def _copy_with(self, *args) -> Self:
        pass

    # *** Interface util functions for all molecule representations ***

    def __len__(self):
        return self.seq_length

    def __str__(self):
        if self._str_id is not None:
            return self._str_id

        return super().__str__()

    # Note: only performs a shallow copy
    def copy(self) -> Self:
        return copy.copy(self)

    def to_device(self, device: TDevice) -> Self:
        obj_copy = self.copy()
        for attr_name in vars(self):
            value = getattr(self, attr_name, None)
            if value is not None and isinstance(value, _T):
                setattr(obj_copy, attr_name, value.to(device))

        return obj_copy


class SmolBatch(Sequence, Generic[TSmolMol]):
    """Abstract class for molecule batch representations for the Smol library"""

    # All subclasses must call super init
    def __init__(self, mols: list[TSmolMol], device: TDevice | None = None):
        if len(mols) == 0:
            raise RuntimeError("Batch must be non-empty")

        if device is None:
            device = mols[0].device

        mols = [mol.to_device(device) for mol in mols]

        self._mols = mols
        self._device = torch.device(device)

    # *** Properties for molecular batches ***

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def seq_length(self) -> list[int]:
        return [mol.seq_length for mol in self._mols]

    @property
    def batch_size(self) -> int:
        return len(self._mols)

    @property
    @abstractmethod
    def mask(self) -> _T:
        pass

    # *** Sequence methods ***

    def __len__(self) -> int:
        return self.batch_size

    def __getitem__(self, item: int) -> TSmolMol:
        return self._mols[item]

    # *** Default methods which may need overriden ***

    def to_bytes(self) -> bytes:
        mol_bytes = [mol.to_bytes() for mol in self._mols]
        return pickle.dumps(mol_bytes)

    def to_list(self) -> list[TSmolMol]:
        return self._mols

    def apply(self, fn: Callable[[TSmolMol, int], TSmolMol]) -> Self:
        applied = [fn(mol, idx) for idx, mol in enumerate(self._mols)]
        [_check_type(mol, SmolMol, "apply result") for mol in applied]
        return self.from_list(applied)

    def copy(self) -> Self:
        # Only performs shallow copy on individual mols
        mol_copies = [mol.copy() for mol in self._mols]
        return self.from_list(mol_copies)

    def to_device(self, device: TDevice) -> Self:
        applied = [mol.to_device(device) for mol in self._mols]
        return self.from_list(applied)

    @classmethod
    def collate(cls, batches: list[SmolBatch]) -> Self:
        all_mols = [mol for batch in batches for mol in batch]
        return cls.from_list(all_mols)

    # *** Abstract methods for batches ***

    @staticmethod
    @abstractmethod
    def from_bytes(data: bytes) -> SmolBatch:
        pass

    @staticmethod
    @abstractmethod
    def from_list(mols: list[TSmolMol]) -> SmolBatch:
        pass

    @staticmethod
    @abstractmethod
    def from_tensors(*tensors: _T) -> SmolBatch:
        pass

    @staticmethod
    @abstractmethod
    def load(save_dir: str, lazy: bool = False) -> SmolBatch:
        pass

    @abstractmethod
    def save(self, save_dir: str, shards: int = 0, threads: int = 0) -> None:
        pass


# *******************************
# *** MolRepr Implementations ***
# *******************************


# TODO remove distributions for atomics
# TODO documentation
class GeometricMol(SmolMol):
    def __init__(
        self,
        coords: _T,
        atomics: _T,
        bond_indices: _T | None = None,
        bond_types: _T | None = None,
        charges: _T | None = None,
        residues: _T | None = None,
        device: TDevice | None = None,
        is_mmap: bool = False,
        str_id: str | None = None,
    ):
        # Check that each tensor has correct number of dimensions
        _check_shape_len(coords, 2, "coords")

        _check_shape_len(atomics, [1, 2], "atomics")
        _check_shapes_equal(coords, atomics, 0)

        if bond_indices is None and bond_types is not None:
            raise ValueError("bond_indices must be provided if bond_types are provided.")

        # Create an empty edge list if bonds are not provided
        # Or assume single bonds if bond_indices is provided but bond_types is not
        bond_indices = torch.tensor([[]] * 2).T if bond_indices is None else bond_indices
        bond_types = torch.tensor([1] * bond_indices.size(0)) if bond_types is None else bond_types

        _check_shape_len(bond_indices, 2, "bond indices")
        _check_dim_shape(bond_indices, 1, 2, "bond indices")

        _check_shape_len(bond_types, [1, 2], "bond types")
        _check_shapes_equal(bond_indices, bond_types, 0)

        charges = torch.zeros(coords.size(0)) if charges is None else charges

        _check_shape_len(charges, 1, "charges")
        _check_shapes_equal(coords, charges, 0)

        residues = torch.zeros(coords.size(0)) if residues is None else residues

        _check_shape_len(residues, 1, "residues")
        _check_shapes_equal(coords, residues, 0)

        device = coords.device if device is None else torch.device(device)

        self._coords = coords
        self._atomics = atomics
        self._bond_indices = bond_indices
        self._bond_types = bond_types
        self._charges = charges
        self._residues = residues
        self._device = device
        self.holo = None

        # If the data are not stored in mmap tensors, then convert to expected type and move to device
        if not is_mmap:
            # Use float if atomics is a distribution over atomic numbers
            atomics = atomics.float() if len(atomics.size()) == 2 else atomics.long()
            bond_types = bond_types.float() if len(bond_types.size()) == 2 else bond_types.long()

            self._atomics = atomics.to(device)
            self._coords = coords.float().to(device)
            self._bond_indices = bond_indices.long().to(device)
            self._charges = charges.long().to(device)
            self._residues = residues.long().to(device)

        super().__init__(str_id)

    # *** General Properties ***

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def seq_length(self) -> int:
        return self._coords.shape[0]

    # *** Geometric Specific Properties ***

    @property
    def coords(self) -> _T:
        return self._coords.float().to(self._device)

    @property
    def atomics(self) -> _T:
        if len(self._atomics.size()) == 2:
            return self._atomics.float().to(self._device)

        return self._atomics.long().to(self._device)

    @property
    def bond_indices(self) -> _T:
        return self._bond_indices.long().to(self._device)

    @property
    def bond_types(self) -> _T:
        if len(self._bond_types.size()) == 2:
            return self._bond_types.float().to(self._device)

        return self._bond_types.long().to(self._device)

    @property
    def bonds(self) -> _T:
        bond_types = self.bond_types
        if len(bond_types.size()) == 2:
            bond_types = torch.argmax(bond_types, dim=-1)

        return torch.cat((self.bond_indices, bond_types.unsqueeze(1)), dim=-1)

    @property
    def charges(self) -> _T:
        return self._charges.long().to(self._device)

    @property
    def residues(self) -> _T:
        return self._residues.long().to(self._device)

    # Note: this will always return a symmetric NxN matrix
    @property
    def adjacency(self) -> _T:
        bond_indices = self.bond_indices
        bond_types = self.bond_types
        return smolF.adj_from_edges(bond_indices, bond_types, self.seq_length, symmetric=True)

    @property
    def com(self):
        return self.coords.sum(dim=0) / self.seq_length

    @property
    def ligand_mask(self) -> _T:
        """Returns a 1D mask, 1 for ligand, 2 for pocket."""
        mask = [1] * self.seq_length
        return torch.tensor(mask, dtype=torch.int)

    @property
    def is_connected(self) -> bool:
        import networkx as nx

        G = nx.Graph()

        G.add_nodes_from(range(self.seq_length))  # Add all nodes
        G.add_edges_from(self.bond_indices.tolist())  # Add edges

        return nx.is_connected(G)

    # *** Interface Methods ***

    # TODO allow data to not have all dict keys
    @staticmethod
    def from_bytes(data: bytes) -> GeometricMol:
        obj = pickle.loads(data)

        _check_type(obj, dict, "unpickled object")
        _check_dict_key(obj, "coords")
        _check_dict_key(obj, "atomics")
        # _check_dict_key(obj, "bonds")
        _check_dict_key(obj, "charges")
        _check_dict_key(obj, "device")
        _check_dict_key(obj, "id")

        if obj.get("bond_types") is not None:
            bond_indices = obj["bond_indices"]
            bond_types = obj["bond_types"]
        else:
            # Support for an older representation of the bonds
            bonds = obj["bonds"]
            bond_indices = bonds[:, :2]
            bond_types = bonds[:, 2]

        mol = GeometricMol(
            obj["coords"],
            obj["atomics"],
            bond_indices=bond_indices,
            bond_types=bond_types,
            charges=obj["charges"],
            residues=obj.get("residues"),
            device=obj["device"],
            is_mmap=False,
            str_id=obj["id"],
        )
        return mol

    # Note: currently only uses the default conformer for mol
    @staticmethod
    def from_rdkit(mol: Chem.rdchem.Mol) -> GeometricMol:
        # TODO handle this better - maybe create 3D info if not provided, with a warning
        if mol.GetNumConformers() == 0 or not mol.GetConformer().Is3D():
            mol = embed_mol(mol)
            warnings.warn("Molecule does not have 3D coordinates. Using default conformer.")

        conf = mol.GetConformer()
        smiles = smolRD.smiles_from_mol(mol)

        coords = np.array(conf.GetPositions())
        atomics = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]

        bonds = []
        for bond in mol.GetBonds():
            bond_start = bond.GetBeginAtomIdx()
            bond_end = bond.GetEndAtomIdx()

            # TODO perhaps print a warning but just don't add the bond?
            bond_type = smolRD.BOND_IDX_MAP.get(bond.GetBondType())
            if bond_type is None:
                raise NotImplementedError(f"Unsupported bond type {bond.GetBondType()}")

            bonds.append([bond_start, bond_end, bond_type])

        coords = torch.tensor(coords)
        atomics = torch.tensor(atomics)
        charges = torch.tensor(charges)
        if len(bonds) > 0:
            bonds = torch.tensor(bonds).view(-1, 3)
            bond_indices = bonds[:, :2]
            bond_types = bonds[:, 2]
        else:
            bond_indices = torch.zeros((0, 2), dtype=torch.long)
            bond_types = torch.zeros((0,), dtype=torch.long)
        return GeometricMol(coords, atomics, bond_indices, bond_types, charges=charges, str_id=smiles)

    @staticmethod
    def from_sdf(sdf_file: str | Path) -> GeometricMol:
        with open(sdf_file) as f:
            sdf_str = f.read()
        mol = Chem.MolFromMolBlock(sdf_str)
        if mol is None:
            raise ValueError("Could not parse SDF string.")

        return GeometricMol.from_rdkit(mol)

    def to_bytes(self) -> bytes:
        dict_repr = {
            "coords": self.coords,
            "atomics": self.atomics,
            "bond_indices": self.bond_indices,
            "bond_types": self.bond_types,
            "charges": self.charges,
            "residues": self.residues,
            "device": str(self.device),
            "id": self._str_id,
        }
        byte_obj = pickle.dumps(dict_repr, protocol=PICKLE_PROTOCOL)
        return byte_obj

    def to_rdkit(self, vocab: Vocabulary, sanitise: bool = False) -> Chem.rdchem.Mol:
        if len(self.atomics.size()) == 2:
            vocab_indices = torch.argmax(self.atomics, dim=1).tolist()
            tokens = vocab.tokens_from_indices(vocab_indices)

        else:
            atomics = self.atomics.tolist()
            tokens = [smolRD.PT.symbol_from_atomic(a) for a in atomics]

        coords = self.coords.numpy()
        bonds = self.bonds.numpy()
        charges = self.charges.numpy()

        mol = smolRD.mol_from_atoms(coords, tokens, bonds, charges, sanitise=sanitise)
        return mol

    def _copy_with(
        self,
        coords: _T | None = None,
        atomics: _T | None = None,
        bond_indices: _T | None = None,
        bond_types: _T | None = None,
        charges: _T | None = None,
        residues: _T | None = None,
    ) -> GeometricMol:

        coords = self.coords if coords is None else coords
        atomics = self.atomics if atomics is None else atomics
        bond_indices = self.bond_indices if bond_indices is None else bond_indices
        bond_types = self.bond_types if bond_types is None else bond_types
        charges = self.charges if charges is None else charges
        residues = self.residues if residues is None else residues

        obj = GeometricMol(
            coords,
            atomics,
            bond_indices=bond_indices,
            bond_types=bond_types,
            charges=charges,
            residues=residues,
            device=self.device,
            is_mmap=False,
            str_id=self._str_id,
        )
        return obj

    # TODO add tests
    def permute(self, indices: list[int]) -> GeometricMol:
        """Used for permuting atom order. Can also be used for taking a subset of atoms but not duplicating."""

        if len(set(indices)) != len(indices):
            raise ValueError("Indices list cannot contain duplicates.")

        if max(indices) >= self.seq_length:
            raise ValueError(f"Index {max(indices)} is out of bounds for molecule with {self.seq_length} atoms.")

        indices = torch.tensor(indices)

        coords = self.coords[indices]
        atomics = self.atomics[indices]
        charges = self.charges[indices]
        residues = self.residues[indices]

        # Relabel bond from and to indices with new indices
        from_idxs = self.bond_indices[:, 0].clone()
        to_idxs = self.bond_indices[:, 1].clone()
        curr_indices = torch.arange(indices.size(0))

        old_from, new_from = torch.nonzero(from_idxs.unsqueeze(1) == curr_indices, as_tuple=True)
        old_to, new_to = torch.nonzero(to_idxs.unsqueeze(1) == curr_indices, as_tuple=True)

        from_idxs[old_from] = indices[new_from]
        to_idxs[old_to] = indices[new_to]

        # Remove bonds whose indices do not appear in new indices list
        bond_idxs = torch.cat((from_idxs.unsqueeze(-1), to_idxs.unsqueeze(-1)), dim=-1)
        mask = bond_idxs.unsqueeze(-1) == indices.view(1, 1, -1)
        mask = ~(~mask.any(dim=-1)).any(dim=-1)
        bond_indices = bond_idxs[mask]
        bond_types = self.bond_types[mask]

        mol_copy = self._copy_with(
            coords=coords,
            atomics=atomics,
            bond_indices=bond_indices,
            bond_types=bond_types,
            charges=charges,
            residues=residues,
        )
        return mol_copy

    # *** Geometric Specific Methods ***

    def zero_com(self) -> GeometricMol:
        shifted = self.coords - self.com.unsqueeze(0)
        return self._copy_with(coords=shifted)

    def rotate(self, rotation: Rotation | TCoord) -> GeometricMol:
        rotated = smolF.rotate(self.coords, rotation)
        return self._copy_with(coords=rotated)

    def shift(self, shift: float | np.ndarray | TCoord) -> GeometricMol:
        shift_tensor = torch.tensor(shift).view(1, -1)
        shifted = self.coords + shift_tensor
        return self._copy_with(coords=shifted)

    def scale(self, scale: float) -> GeometricMol:
        scaled = self.coords * scale
        return self._copy_with(coords=scaled)


class GeometricMolBatch(SmolBatch[GeometricMol]):
    def __init__(self, mols: list[GeometricMol], device: TDevice | None = None):
        for mol in mols:
            _check_type(mol, GeometricMol, "molecule object")

        super().__init__(mols, device)

        # Cache for batched tensors
        self._coords = None
        self._mask = None
        self._atomics = None
        self._bond_indices = None
        self._bond_types = None
        self._bonds = None
        self._charges = None
        self._residues = None

    # *** General Properties ***

    @property
    def mask(self) -> _T:
        if self._mask is None:
            masks = [torch.ones(mol.seq_length) for mol in self._mols]
            self._mask = smolF.pad_tensors(masks)

        return self._mask

    # *** Geometric Specific Properties ***

    @property
    def coords(self) -> _T:
        if self._coords is None:
            coords = [mol.coords for mol in self._mols]
            self._coords = smolF.pad_tensors(coords)

        return self._coords

    @property
    def atomics(self) -> _T:
        if self._atomics is None:
            atomics = [mol.atomics for mol in self._mols]
            self._atomics = smolF.pad_tensors(atomics)

        return self._atomics

    @property
    def bond_indices(self) -> _T:
        if self._bond_indices is None:
            bond_indices = [mol.bond_indices for mol in self._mols]
            self._bond_indices = smolF.pad_tensors(bond_indices)

        return self._bond_indices

    @property
    def bond_types(self) -> _T:
        if self._bond_types is None:
            bond_types = [mol.bond_types for mol in self._mols]
            self._bond_types = smolF.pad_tensors(bond_types)

        return self._bond_types

    @property
    def bonds(self) -> _T:
        if self._bonds is None:
            bonds = [mol.bonds for mol in self._mols]
            self._bonds = smolF.pad_tensors(bonds)

        return self._bonds

    @property
    def charges(self) -> _T:
        if self._charges is None:
            charges = [mol.charges for mol in self._mols]
            self._charges = smolF.pad_tensors(charges)

        return self._charges

    @property
    def residues(self) -> _T:
        if self._residues is None:
            residues = [mol.residues for mol in self._mols]
            self._residues = smolF.pad_tensors(residues)

        return self._residues

    @property
    def adjacency(self) -> _T:
        n_atoms = max(self.seq_length)
        adjs = [smolF.adj_from_edges(mol.bond_indices, mol.bond_types, n_atoms, symmetric=True) for mol in self._mols]
        return torch.stack(adjs)

    @property
    def com(self) -> _T:
        return smolF.calc_com(self.coords, node_mask=self.mask)

    # *** Interface Methods ***

    @staticmethod
    def from_bytes(data: bytes) -> GeometricMolBatch:
        mols = [GeometricMol.from_bytes(mol_bytes) for mol_bytes in pickle.loads(data)]
        return GeometricMolBatch.from_list(mols)

    @staticmethod
    def from_list(mols: list[GeometricMol]) -> GeometricMolBatch:
        return GeometricMolBatch(mols)

    # TODO add bonds and charges
    @staticmethod
    def from_tensors(
        coords: _T,
        atomics: _T | None = None,
        num_atoms: _T | None = None,
        is_mmap: bool = False,
    ) -> GeometricMolBatch:

        _check_shape_len(coords, 3, "coords")

        if atomics is not None:
            _check_shape_len(atomics, [2, 3], "atomics")
            _check_shapes_equal(coords, atomics, [0, 1])

        if num_atoms is not None:
            _check_shape_len(num_atoms, 1, "num_atoms")
            _check_shapes_equal(coords, num_atoms, 0)

        device = coords.device
        batch_size, max_atoms = coords.size()[:2]

        num_atoms = torch.tensor([max_atoms] * batch_size) if num_atoms is None else num_atoms
        seq_lens = num_atoms.int().tolist()

        mols = []
        for idx in range(coords.size(0)):
            mol_coords = coords[idx, : seq_lens[idx]]
            mol_types = atomics[idx, : seq_lens[idx]] if atomics is not None else None
            mol = GeometricMol(mol_coords, mol_types, device=device, is_mmap=is_mmap)
            mols.append(mol)

        batch = GeometricMolBatch(mols, device)

        # Put all tensors on same device and set batched tensor cache if they are not mem mapped
        if not is_mmap:

            # Use float if types is a distribution over atom types
            if atomics is not None:
                if len(atomics.size()) == 3:
                    atomics = atomics.float().to(device)
                else:
                    atomics = atomics.long().to(device)

            batch._atomics = atomics
            batch._coords = coords.float().to(device)

        return batch

    @staticmethod
    def load(save_dir: str, lazy: bool = False) -> GeometricMolBatch:
        save_path = Path(save_dir)

        if not save_path.exists() or not save_path.is_dir():
            raise RuntimeError(f"Folder {save_dir} does not exist.")

        batches = []
        curr_folders = [save_path]

        while len(curr_folders) != 0:
            curr_path = curr_folders[0]
            if (curr_path / "atoms.npy").exists():
                batch = GeometricMolBatch._load_batch(curr_path, lazy=lazy)
                batches.append(batch)

            children = [path for path in curr_path.iterdir() if path.is_dir()]
            curr_folders = curr_folders[1:]
            curr_folders.extend(children)

        collated = GeometricMolBatch.collate(batches)
        return collated

    def save(self, save_dir: str | Path, shards: int = 0, threads: int = 0) -> None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        if shards is None or shards <= 0:
            return self._save_batch(self, save_path)

        items_per_shard = (len(self) // shards) + 1
        start_idxs = [idx * items_per_shard for idx in range(shards)]
        end_idxs = [(idx + 1) * items_per_shard for idx in range(shards)]
        end_idxs[-1] = len(self)

        batches = [self._mols[s_idx:e_idx] for s_idx, e_idx in zip(start_idxs, end_idxs, strict=False)]
        batches = [GeometricMolBatch.from_list(batch_list) for batch_list in batches]

        f_len = len(str(shards - 1))
        dir_names = [f"{str(b_idx):0{f_len}}_n{str(b.batch_size)}" for b_idx, b in enumerate(batches)]
        save_paths = [save_path / name for name in dir_names]

        if threads is not None and threads > 0:
            executor = ThreadPoolExecutor(threads)
            futures = [executor.submit(self._save_batch, b, path) for b, path in zip(batches, save_paths, strict=False)]
            [future.result() for future in futures]

        else:
            [self._save_batch(batch, path) for batch, path in zip(batches, save_paths, strict=False)]

    # *** Geometric Specific Methods ***

    def zero_com(self) -> GeometricMolBatch:
        shifted = self.coords - self.com
        shifted = shifted * self.mask.unsqueeze(2)
        return self._from_coords(shifted)

    def rotate(self, rotation: Rotation | TCoord) -> GeometricMolBatch:
        return self.apply(lambda mol, idx: mol.rotate(rotation))

    def shift(self, shift: TCoord) -> GeometricMolBatch:
        shift_tensor = torch.tensor(shift).view(1, 1, -1)
        shifted = (self.coords + shift_tensor) * self.mask.unsqueeze(2)
        return self._from_coords(shifted)

    def scale(self, scale: float) -> GeometricMolBatch:
        scaled = (self.coords * scale) * self.mask.unsqueeze(2)
        return self._from_coords(scaled)

    # *** Util Methods ***

    def _from_coords(self, coords: _T) -> GeometricMolBatch:
        _check_shape_len(coords, 3, "coords")
        _check_shapes_equal(coords, self.coords, [0, 1, 2])

        if coords.size(0) != self.batch_size:
            raise RuntimeError("coords batch size must be the same as self batch size")

        if coords.size(1) != max(self.seq_length):
            raise RuntimeError("coords num atoms must be the same as largest molecule")

        coords = coords.float().to(self.device)

        mol_coords = [cs[:num_atoms, :] for cs, num_atoms in zip(list(coords), self.seq_length, strict=False)]
        mols = [mol._copy_with(coords=cs) for mol, cs in zip(self._mols, mol_coords, strict=False)]
        batch = GeometricMolBatch(mols)

        # Set the cache for the tensors that have already been created
        batch._coords = coords
        batch._mask = self.mask if self._mask is not None else None
        batch._atomics = self.atomics if self._atomics is not None else None
        batch._bonds = self.bonds if self._bonds is not None else None

        return batch

    # TODO add bonds and charges
    @staticmethod
    def _load_batch(batch_dir: Path, lazy: bool) -> GeometricMolBatch:
        mmap_mode = "r+" if lazy else None

        num_atoms_arr = np.load(batch_dir / "atoms.npy")
        num_atoms = torch.tensor(num_atoms_arr)

        # torch now supports loading mmap tensors but np mmap seems a lot more mature and creating a tensor from a
        # mmap array using from_numpy() preserves the mmap array without reading in the data until required
        coords_arr = np.load(batch_dir / "coords.npy", mmap_mode=mmap_mode)
        coords = torch.from_numpy(coords_arr)

        atomics_arr = np.load(batch_dir / "atomics.npy", mmap_mode=mmap_mode)
        atomics = torch.from_numpy(atomics_arr)

        # bonds = None
        # bonds_path = batch_dir / "bonds.npy"
        # if edges_path.exists():
        #     bonds_arr = np.load(bonds_path, mmap_mode=mmap_mode)
        #     bonds = torch.from_numpy(bonds_arr)

        batch = GeometricMolBatch.from_tensors(coords, atomics, num_atoms, is_mmap=lazy)
        return batch

    @staticmethod
    def _save_batch(batch, save_path: Path) -> None:
        save_path.mkdir(exist_ok=True, parents=True)

        coords = batch.coords.cpu().numpy()
        np.save(save_path / "coords.npy", coords)

        num_atoms = np.array(batch.seq_length).astype(np.int16)
        np.save(save_path / "atoms.npy", num_atoms)

        atomics = batch.atomics.cpu().numpy()
        np.save(save_path / "atomics.npy", atomics)

        bonds = batch.bonds.cpu().numpy()
        if bonds.shape[1] != 0:
            np.save(save_path / "bonds.npy", bonds)
