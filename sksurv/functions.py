# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import annotations

from typing import List

import numpy as np
from sklearn.utils import check_consistent_length

__all__ = ['StepFunction', 'sum_fn', 'avg_fn']


class StepFunction:
    """Callable step function. x can be infinite.

    .. math::

        f(z) = a * y_i + b,
        x_i \\leq z < x_{i + 1}

    Parameters
    ----------
    x : ndarray, shape = (n_points,)
        Values on the x axis in ascending order.

    y : ndarray, shape = (n_points,)
        Corresponding values on the y axis.

    a : float, optional, default: 1.0
        Constant to multiply by.

    b : float, optional, default: 0.0
        Constant offset term.
    """
    def __init__(self, x, y, a=1., b=0.):
        check_consistent_length(x, y)
        self.x = np.concatenate([[-np.inf], x, [np.inf]])
        self.y = np.concatenate([[y[0]], y, [y[-1]]])
        self.a = a
        self.b = b

    def __call__(self, x):
        """Evaluate step function.

        Parameters
        ----------
        x : float|array-like, shape=(n_values,)
            Values to evaluate step function at.

        Returns
        -------
        y : float|array-like, shape=(n_values,)
            Values of step function at `x`.
        """
        x = np.atleast_1d(x)
        if np.min(x) < self.x[0] or np.max(x) > self.x[-1]:
            raise ValueError(
                "x must be within [%f; %f]" % (self.x[0], self.x[-1]))
        i = np.searchsorted(self.x, x, side='left')
        not_exact = self.x[i] != x
        i[not_exact] -= 1
        value = self.a * self.y[i] + self.b
        if value.shape[0] == 1:
            return value[0]
        return value

    def __repr__(self):
        return "StepFunction(x=%r, y=%r, a=%r, b=%r)" % (self.x, self.y, self.a, self.b)

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return (
                all(self.x == other.x)
                and all(self.y == other.y)
                and self.a == other.a
                and self.b == other.b
            )
        return False

    def __add__(self, other) -> StepFunction:
        if type(other) in [int, float] or np.isscalar(other):
            return StepFunction(self.x, self.y + other)
        elif type(other) != StepFunction:
            raise TypeError(f"Cannot sum {type(self)} and {type(other)}.")
        x = np.unique(np.sort(np.concatenate([other.x, self.x])))
        y = self(x) + other(x)
        return StepFunction(x, y)

    def __mul__(self, other) -> StepFunction:
        if type(other) not in [int, float] or not np.isscalar(other):
            raise TypeError(f"Cannot multiply {type(self)} and {type(other)}.")
        return StepFunction(self.x, self.y * other)

    def __truediv__(self, other) -> StepFunction:
        if type(other) not in [int, float] or not np.isscalar(other):
            raise TypeError(f"Cannot divide {type(self)} and {type(other)}.")
        return StepFunction(self.x, self.y / other)


def sum_fn(fns: List[StepFunction]) -> StepFunction:
    if len(fns) == 0:
        raise RuntimeError("Empty list.")
    if len(fns) == 1:
        return fns[0]
    return sum(fns[1:], start=fns[0])


def avg_fn(fns: List[StepFunction]) -> StepFunction:
    if len(fns) == 0:
        raise RuntimeError("Empty list.")
    if len(fns) == 1:
        return fns[0]
    return sum(fns[1:], start=fns[0]) / len(fns)
