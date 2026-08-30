"""Compute molecular features from SMILES strings (RDKit / Mordred backends)."""

import warnings

import numpy as np
import pandas as pd

#: Curated physicochemical descriptors for ``method="descriptors"``.
CURATED_DESCRIPTORS = [
    "MolWt", "MolLogP", "MolMR", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "RingCount", "NumAromaticRings",
    "NumSaturatedRings", "NumAliphaticRings", "HeavyAtomCount",
    "NumHeteroatoms", "FractionCSP3", "NumValenceElectrons",
]

_FINGERPRINT_METHODS = ("morgan", "maccs", "rdkit", "atompair", "torsion")
METHODS = (*_FINGERPRINT_METHODS, "descriptors", "mordred")


def _require_rdkit():
    try:
        from rdkit import Chem
        return Chem
    except ImportError as exc:
        raise ImportError(
            "featurize requires RDKit; install it with: "
            "pip install ikn-library[chem]"
        ) from exc


def _parse_smiles(smiles, on_invalid):
    Chem = _require_rdkit()
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    valid = np.array([m is not None for m in mols])
    if not valid.all():
        n_bad = int((~valid).sum())
        examples = [s for s, ok in zip(smiles, valid) if not ok][:3]
        if on_invalid == "raise":
            raise ValueError(f"{n_bad} invalid SMILES, e.g. {examples}")
        if on_invalid != "drop":
            raise ValueError('on_invalid must be "drop" or "raise"')
        warnings.warn(f"dropped {n_bad} invalid SMILES (e.g. {examples})",
                      stacklevel=3)
    return [m for m in mols if m is not None], valid


def _fingerprints(mols, method, radius, n_bits):
    if method == "maccs":
        from rdkit.Chem import DataStructs, MACCSkeys
        rows = []
        for mol in mols:
            arr = np.zeros(167, dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), arr)
            rows.append(arr)
        return np.array(rows), [f"maccs_{i}" for i in range(167)]

    from rdkit.Chem import rdFingerprintGenerator as gen
    generators = {
        "morgan": lambda: gen.GetMorganGenerator(radius=radius, fpSize=n_bits),
        "rdkit": lambda: gen.GetRDKitFPGenerator(fpSize=n_bits),
        "atompair": lambda: gen.GetAtomPairGenerator(fpSize=n_bits),
        "torsion": lambda: gen.GetTopologicalTorsionGenerator(fpSize=n_bits),
    }
    generator = generators[method]()
    rows = np.array([generator.GetFingerprintAsNumPy(mol) for mol in mols])
    return rows, [f"{method}_{i}" for i in range(rows.shape[1])]


def _rdkit_descriptors(mols, names):
    from rdkit.Chem import Descriptors
    if names == "all":
        records = [Descriptors.CalcMolDescriptors(mol) for mol in mols]
        return pd.DataFrame.from_records(records)
    names = list(CURATED_DESCRIPTORS if names is None else names)
    functions = [getattr(Descriptors, name) for name in names]
    rows = [[f(mol) for f in functions] for mol in mols]
    return pd.DataFrame(rows, columns=names)


def _mordred_descriptors(mols):
    try:
        from mordred import Calculator, descriptors
    except ImportError as exc:
        raise ImportError(
            'method="mordred" requires the mordred package; install it '
            "with: pip install mordredcommunity"
        ) from exc
    calculator = Calculator(descriptors, ignore_3D=True)
    # nproc=1 keeps the computation in-process: mordred's multiprocessing
    # breaks under the spawn start method (macOS/Windows, notebooks).
    frame = calculator.pandas(mols, quiet=True, nproc=1)
    frame.columns = [str(c) for c in frame.columns]
    # Failed descriptors come back as error objects; coerce them to NaN.
    return frame.apply(pd.to_numeric, errors="coerce").reset_index(drop=True)


def featurize(smiles, y=None, method="morgan", on_invalid="drop",
              radius=2, n_bits=1024, descriptor_names=None):
    """Turn SMILES strings into a numeric feature table.

    Args:
        smiles: Iterable of SMILES strings.
        y: Optional label array of the same length; rows dropped for
            invalid SMILES are dropped from ``y`` too, keeping the two
            aligned.
        method: One of:

            - ``"morgan"`` — Morgan/ECFP circular fingerprint (default;
              ``radius``, ``n_bits``),
            - ``"maccs"`` — 167 standard MACCS keys,
            - ``"rdkit"`` — RDKit path-based fingerprint (``n_bits``),
            - ``"atompair"`` — atom-pair fingerprint (``n_bits``),
            - ``"torsion"`` — topological torsion fingerprint (``n_bits``),
            - ``"descriptors"`` — RDKit physicochemical descriptors
              (a curated set by default; see ``descriptor_names``),
            - ``"mordred"`` — ~1,600 2D descriptors via the Mordred
              package (``pip install mordredcommunity``); descriptors
              that fail for a molecule become NaN.
        on_invalid: ``"drop"`` (default) removes molecules whose SMILES
            RDKit cannot parse, with a warning; ``"raise"`` raises
            instead.
        radius: Morgan fingerprint radius.
        n_bits: Fingerprint length for the bit-vector methods.
        descriptor_names: For ``method="descriptors"``: a list of RDKit
            descriptor names, or ``"all"`` for every 2D descriptor RDKit
            offers; ``None`` uses :data:`CURATED_DESCRIPTORS`.

    Returns:
        ``pandas.DataFrame`` of shape ``(n_valid_molecules, n_features)``
        with named columns — or the tuple ``(X, y)`` when ``y`` is given.

    Requires RDKit (``pip install ikn-library[chem]``).
    """
    smiles = list(smiles)
    if y is not None and len(y) != len(smiles):
        raise ValueError("smiles and y must have the same length")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")

    mols, valid = _parse_smiles(smiles, on_invalid)

    if method in _FINGERPRINT_METHODS:
        values, columns = _fingerprints(mols, method, radius, n_bits)
        X = pd.DataFrame(values, columns=columns)
    elif method == "descriptors":
        X = _rdkit_descriptors(mols, descriptor_names)
    else:  # mordred
        X = _mordred_descriptors(mols)

    if y is None:
        return X
    return X, np.asarray(y)[valid]
