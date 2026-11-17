# Copyright 2019 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ADAPTED FILE: project minimal dependency extraction from Cirq v1.6.1.
# Cirq serialization/equality/measurement/convenience methods and abstract base
# are omitted. The selected numerical update/amplitude methods below are verbatim.
# Two exact binary digit adapters replace the general Cirq value-module helpers.
from __future__ import annotations
from typing import Any, Sequence
from types import SimpleNamespace
import numpy as np

def big_endian_int_to_digits(value, *, digit_count, base=2):
    value=int(value)
    if base!=2 or value<0 or value>=1<<digit_count:
        raise ValueError("Expected a valid fixed-width binary integer")
    return [(value>>(digit_count-1-i))&1 for i in range(digit_count)]

cirq=SimpleNamespace(big_endian_int_to_bits=lambda value,bit_count:
                     big_endian_int_to_digits(value,digit_count=bit_count))
value=SimpleNamespace(Scalar=complex)

class StabilizerStateChForm:
    """Official Cirq numerical CH state with preserved omega; no exponential helpers."""
    def __init__(self, num_qubits: int, initial_state: int = 0) -> None:
        """Initializes StabilizerStateChForm
        Args:
            num_qubits: The number of qubits in the system.
            initial_state: The computational basis representation of the
                state as a big endian int.
        """
        self.n = num_qubits

        # The state is represented by a set of binary matrices and vectors.
        # See Section IVa of Bravyi et al
        self.G = np.eye(self.n, dtype=bool)

        self.F = np.eye(self.n, dtype=bool)
        self.M = np.zeros((self.n, self.n), dtype=bool)
        self.gamma = np.zeros(self.n, dtype=int)

        self.v = np.zeros(self.n, dtype=bool)
        self.s = np.zeros(self.n, dtype=bool)

        self.omega: complex = 1

        # Apply X for every non-zero element of initial_state
        for i, val in enumerate(
            big_endian_int_to_digits(initial_state, digit_count=num_qubits, base=2)
        ):
            if val:
                self.apply_x(i)


    def inner_product_of_state_and_x(self, x: int) -> complex:
        """Returns the amplitude of x'th element of
        the state vector, i.e. <x|psi>"""
        if type(x) == int:
            y = cirq.big_endian_int_to_bits(x, bit_count=self.n)

        mu = sum(y * self.gamma)

        u = np.zeros(self.n, dtype=bool)
        for p in range(self.n):
            if y[p]:
                u ^= self.F[p, :]
                mu += 2 * (sum(self.M[p, :] & u) % 2)
        return (
            self.omega
            * 2 ** (-sum(self.v) / 2)
            * 1j**mu
            * (-1) ** sum(self.v & u & self.s)
            * bool(np.all(self.v | (u == self.s)))
        )


    def _S_right(self, q):
        r"""Right multiplication version of S gate."""
        self.M[:, q] ^= self.F[:, q]
        self.gamma[:] = (self.gamma[:] - self.F[:, q]) % 4


    def _CZ_right(self, q, r):
        r"""Right multiplication version of CZ gate."""
        self.M[:, q] ^= self.F[:, r]
        self.M[:, r] ^= self.F[:, q]
        self.gamma[:] = (self.gamma[:] + 2 * self.F[:, q] * self.F[:, r]) % 4


    def _CNOT_right(self, q, r):
        r"""Right multiplication version of CNOT gate."""
        self.G[:, q] ^= self.G[:, r]
        self.F[:, r] ^= self.F[:, q]
        self.M[:, q] ^= self.M[:, r]


    def update_sum(self, t, u, delta=0, alpha=0):
        """Implements the transformation (Proposition 4 in Bravyi et al)

        ``i^alpha U_H (|t> + i^delta |u>) = omega W_C W_H |s'>``
        """
        if np.all(t == u):
            self.s = t
            self.omega *= 1 / np.sqrt(2) * (-1) ** alpha * (1 + 1j**delta)
            return
        set0 = np.where((~self.v) & (t ^ u))[0]
        set1 = np.where(self.v & (t ^ u))[0]

        # implement Vc
        if len(set0) > 0:
            q = set0[0]
            for i in set0:
                if i != q:
                    self._CNOT_right(q, i)
            for i in set1:
                self._CZ_right(q, i)
        elif len(set1) > 0:
            q = set1[0]
            for i in set1:
                if i != q:
                    self._CNOT_right(i, q)

        e = np.zeros(self.n, dtype=bool)
        e[q] = True

        if t[q]:
            y = u ^ e
            z = u
        else:
            y = t
            z = t ^ e

        (omega, a, b, c) = self._H_decompose(self.v[q], y[q], z[q], delta)

        self.s = y
        self.s[q] = c
        self.omega *= (-1) ** alpha * omega

        if a:
            self._S_right(q)
        self.v[q] ^= b ^ self.v[q]


    def _H_decompose(self, v, y, z, delta):
        """Determines the transformation

                H^v (|y> + i^delta |z>) = omega S^a H^b |c>

        where the state represents a single qubit.

        Input: v,y,z are boolean; delta is an integer (mod 4)
        Outputs: a,b,c are boolean; omega is a complex number

        Precondition: y != z"""
        if y == z:
            raise ValueError('|y> is equal to |z>')

        if not v:
            omega = (1j) ** (delta * int(y))

            delta2 = ((-1) ** y * delta) % 4
            c = bool((delta2 >> 1))
            a = bool(delta2 & 1)
            b = True
        else:
            if not (delta & 1):
                a = False
                b = False
                c = bool(delta >> 1)
                omega = (-1) ** (c & y)
            else:
                omega = 1 / np.sqrt(2) * (1 + 1j**delta)
                b = True
                a = True
                c = not ((delta >> 1) ^ y)

        return omega, a, b, c


    def apply_x(self, axis: int, exponent: float = 1, global_shift: float = 0):
        if exponent % 2 != 0:
            if exponent % 0.5 != 0.0:
                raise ValueError('X exponent must be half integer')  # pragma: no cover
            self.apply_h(axis)
            self.apply_z(axis, exponent)
            self.apply_h(axis)
        self.omega *= _phase(exponent, global_shift)


    def apply_y(self, axis: int, exponent: float = 1, global_shift: float = 0):
        if exponent % 0.5 != 0.0:
            raise ValueError('Y exponent must be half integer')  # pragma: no cover
        shift = _phase(exponent, global_shift)
        if exponent % 2 == 0:
            self.omega *= shift
        elif exponent % 2 == 0.5:
            self.apply_z(axis)
            self.apply_h(axis)
            self.omega *= shift * (1 + 1j) / (2**0.5)
        elif exponent % 2 == 1:
            self.apply_z(axis)
            self.apply_h(axis)
            self.apply_z(axis)
            self.apply_h(axis)
            self.omega *= shift * 1j
        elif exponent % 2 == 1.5:
            self.apply_h(axis)
            self.apply_z(axis)
            self.omega *= shift * (1 - 1j) / (2**0.5)


    def apply_z(self, axis: int, exponent: float = 1, global_shift: float = 0):
        if exponent % 2 != 0:
            if exponent % 0.5 != 0.0:
                raise ValueError('Z exponent must be half integer')  # pragma: no cover
            effective_exponent = exponent % 2
            for _ in range(int(effective_exponent * 2)):
                # Prescription for S left multiplication.
                # Reference: https://arxiv.org/abs/1808.00128 Proposition 4 end
                self.M[axis, :] ^= self.G[axis, :]
                self.gamma[axis] = (self.gamma[axis] - 1) % 4
        self.omega *= _phase(exponent, global_shift)


    def apply_h(self, axis: int, exponent: float = 1, global_shift: float = 0):
        if exponent % 2 != 0:
            if exponent % 1 != 0:
                raise ValueError('H exponent must be integer')  # pragma: no cover
            # Prescription for H left multiplication
            # Reference: https://arxiv.org/abs/1808.00128
            # Equations 48, 49 and Proposition 4
            t = self.s ^ (self.G[axis, :] & self.v)
            u = self.s ^ (self.F[axis, :] & (~self.v)) ^ (self.M[axis, :] & self.v)
            alpha = sum(self.G[axis, :] & (~self.v) & self.s) % 2
            beta = sum(self.M[axis, :] & (~self.v) & self.s)
            beta += sum(self.F[axis, :] & self.v & self.M[axis, :])
            beta += sum(self.F[axis, :] & self.v & self.s)
            beta %= 2
            delta = (self.gamma[axis] + 2 * (alpha + beta)) % 4
            self.update_sum(t, u, delta=delta, alpha=alpha)
        self.omega *= _phase(exponent, global_shift)


    def apply_cz(
        self, control_axis: int, target_axis: int, exponent: float = 1, global_shift: float = 0
    ):
        if exponent % 2 != 0:
            if exponent % 1 != 0:
                raise ValueError('CZ exponent must be integer')  # pragma: no cover
            # Prescription for CZ left multiplication.
            # Reference: https://arxiv.org/abs/1808.00128 Proposition 4 end
            self.M[control_axis, :] ^= self.G[target_axis, :]
            self.M[target_axis, :] ^= self.G[control_axis, :]
        self.omega *= _phase(exponent, global_shift)


    def apply_cx(
        self, control_axis: int, target_axis: int, exponent: float = 1, global_shift: float = 0
    ):
        if exponent % 2 != 0:
            if exponent % 1 != 0:
                raise ValueError('CX exponent must be integer')  # pragma: no cover
            # Prescription for CX left multiplication.
            # Reference: https://arxiv.org/abs/1808.00128 Proposition 4 end
            self.gamma[control_axis] = (
                self.gamma[control_axis]
                + self.gamma[target_axis]
                + 2 * (sum(self.M[control_axis, :] & self.F[target_axis, :]) % 2)
            ) % 4
            self.G[target_axis, :] ^= self.G[control_axis, :]
            self.F[control_axis, :] ^= self.F[target_axis, :]
            self.M[control_axis, :] ^= self.M[target_axis, :]
        self.omega *= _phase(exponent, global_shift)


    def apply_global_phase(self, coefficient: value.Scalar):
        self.omega *= coefficient


def _phase(exponent, global_shift):
    return np.exp(1j * np.pi * global_shift * exponent)

