import inspect
import random
from tqdm import tqdm, trange
from typing import List
from unittest import TestCase

from blockscraper.types import *


class TestTypes(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.packable_types: List[Type[Packable]] = [
            t for t in globals().values()
            if inspect.isclass(t) and issubclass(t, Packable) and (
                    not hasattr(t, "__abstractmethods__") or not t.__abstractmethods__
            )
        ]
        cls.sized_integer_types: List[Type[SizedInteger]] = [
            ty for ty in cls.packable_types if issubclass(ty, SizedInteger) and ty is not SizedInteger
        ]

    def test_sized_integers(self):
        for int_type in tqdm(self.sized_integer_types, desc="testing sized integers", unit=" types", leave=False):
            for _ in trange(1000, desc=f"testing {int_type.__name__}", unit=" tests", leave=False):
                value = random.randint(int_type.MIN_VALUE, int_type.MAX_VALUE)
                packed = int_type(value).pack()
                self.assertEqual(int_type.unpack(packed), value)
