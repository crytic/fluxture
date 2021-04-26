import inspect
import random

from fluxture.structures import PackableStruct
from tqdm import tqdm, trange
from typing import List
from unittest import TestCase

from fluxture.serialization import *


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

    def test_int_enum(self):
        class EnumTest(IntEnum):
            FOO = 0
            BAR = 10

        self.assertEqual(EnumTest.FOO, 0)
        self.assertEqual(EnumTest.BAR, 10)
        self.assertIsInstance(EnumTest.FOO, EnumTest)
        self.assertIsInstance(EnumTest.BAR, EnumTest)
        self.assertEqual(EnumTest["FOO"], EnumTest.FOO)
        self.assertEqual(EnumTest["BAR"], EnumTest.BAR)
        self.assertIs(EnumTest.get_type(), UInt8)
        self.assertEqual(EnumTest.DEFAULT, EnumTest.FOO)

        class SignedEnum(IntEnum):
            FOO = -1
            BAR = 100

        self.assertIs(SignedEnum.get_type(), Int8)

        class LargeEnum(IntEnum):
            FOO = -500
            BAR = 35000

        self.assertIs(LargeEnum.get_type(), Int32)

    def test_bad_int_enum(self):
        def make_bad_enum():
            class BadEnum(IntEnum):
                NOT_INT = "foo"

        self.assertRaises(TypeError, make_bad_enum)

        def make_oversized_enum():
            class OversizedEnum(IntEnum):
                FOO = 99999999999999999999999

        self.assertRaises(TypeError, make_oversized_enum)

    def test_sized_integers(self):
        for int_type in tqdm(self.sized_integer_types, desc="testing sized integers", unit=" types", leave=False):
            for _ in trange(1000, desc=f"testing {int_type.__name__}", unit=" tests", leave=False):
                value = random.randint(int_type.MIN_VALUE, int_type.MAX_VALUE)
                packed = int_type(value).pack()
                self.assertEqual(int_type.unpack(packed), value)

    def test_empty_struct(self):
        class EmptyStruct(PackableStruct):
            pass
        s = EmptyStruct()
        self.assertEqual(s, EmptyStruct.unpack(s.pack()))

    def test_struct_comparison(self):
        class S1(PackableStruct):
            a: Int32
            b: UnsignedChar
            c: UInt64

        class S2(PackableStruct):
            a: Int32
            b: UnsignedChar
            c: UInt64

        self.assertRaises(ValueError, S1, (0, 1))
        self.assertRaises(ValueError, S1, (0, 1, 2, 3))
        self.assertEqual(S1(0, 1, 2), S2(0, 1, 2))
        self.assertNotEqual(S1(0, 1, 2), S2(0, 1, 3))

    def test_struct_packing(self):
        class S3(PackableStruct):
            a: Int32
            b: UInt64
            c: Int16

        s3 = S3(0, 1, 2)
        self.assertEqual(S3.unpack(s3.pack()), s3)

    def test_byte_arrays(self):
        class HasArrays(PackableStruct):
            a: SizedByteArray[1024]
            b: SizedByteArray[0]
            c: SizedByteArray[10]

        self.assertRaises(ValueError, HasArrays, (b"abcd", b"defg", b"hijk"))
        has_arrays = HasArrays(b"abcd", b"", b"hijk")
        self.assertEqual(HasArrays.unpack(has_arrays.pack()), has_arrays)
