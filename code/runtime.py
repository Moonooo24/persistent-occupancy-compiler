"""Pinned conference primitives; one process owns its single-threaded runtime."""
from pathlib import Path
import hashlib
import importlib.util
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'vendor' / 'conference_v21'
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.update(CUDA_VISIBLE_DEVICES='', QISKIT_PARALLEL='FALSE', QISKIT_IN_PARALLEL='TRUE', QISKIT_NUM_PROCS='1')
sys.dont_write_bytecode = True
site = os.environ.get('JOURNAL_QISKIT_SITE')
if site:
    sys.path.insert(0, site)
sys.path.insert(0, str(VENDOR / 'code/mis_phase_portable_v5'))

import numpy as np
import psutil
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector
import mis_phase_core as core
import lossless_qpy_io as io


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def import_vendor(name, relative):
    path = VENDOR / relative
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


G = import_vendor('journal_pinned_care_primitives', 'code/ours_generic_clique_partition_development_v1/prototype.py')
assert core.SEED == 20261101 and core.POLICY == 'native_phase_fsum_x_v2'
QC = QuantumCircuit
