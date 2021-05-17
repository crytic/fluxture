from functools import wraps
from math import sqrt
from typing import Iterable, Iterator, List, Optional, Tuple, Union


Numeric = Union[int, float]


def memoize(func):
    member_name = f"_{func.__name__}_memoized"

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if hasattr(self, member_name):
            return getattr(self, member_name)
        result = func(self, *args, **kwargs)
        setattr(self, member_name, result)
        return result

    return wrapper


class Statistics:
    def __init__(self, iterable: Iterable[Numeric]):
        self._iter: Optional[Iterator[Numeric]] = iter(iterable)
        self._data: List[Numeric] = []

    def __getitem__(self, index: int) -> Numeric:
        while self._iter is not None and index >= len(self._data):
            try:
                self._data.append(next(self._iter))
            except StopIteration:
                self._iter = None
        return self._data[index]

    def __iter__(self) -> Iterator[Numeric]:
        if self._iter is None:
            yield from self._data
            return
        i = 0
        while True:
            try:
                yield self[i]
                i += 1
            except IndexError:
                break

    def __len__(self):
        while self._iter is not None:
            try:
                _ = self[len(self._data)]
            except IndexError:
                break
        return len(self._data)

    def __bool__(self):
        if self._data:
            return True
        try:
            _ = next(iter(self))
            return True
        except StopIteration:
            return False

    @property
    @memoize
    def average(self) -> float:
        if not self:
            return 0.0
        return sum(self) / len(self)

    @property
    @memoize
    def std_dev(self) -> float:
        if not self:
            return 0.0
        avg = self.average
        return sqrt(sum((x - avg)**2.0 for x in self) / len(self))

    @property
    @memoize
    def ordered(self) -> Tuple[Numeric, ...]:
        return tuple(sorted(self))

    @property
    @memoize
    def median(self) -> Numeric:
        n = len(self)
        ordered = self.ordered
        if n % 2 == 0:
            return (ordered[(n - 1) // 2] + ordered[(n + 1) // 2]) / 2.0
        else:
            return ordered[n // 2]

    def __str__(self):
        return f"μ {self.average} σ {self.std_dev} Med {self.median}"
