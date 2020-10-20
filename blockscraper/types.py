from abc import ABC, ABCMeta, abstractmethod
from collections import OrderedDict
from typing import OrderedDict as OrderedDictType, Type, TypeVar
import struct


P = TypeVar("P")


class Packable(ABC):
    @abstractmethod
    def pack(self) -> bytes:
        raise NotImplementedError()

    @classmethod
    def unpack(cls: Type[P], data: bytes) -> P:
        raise NotImplementedError()


class SizedIntegerMeta(ABCMeta):
    FORMAT: str
    BITS: int
    BYTES: int
    SIGNED: bool
    MAX_VALUE: int
    MIN_VALUE: int

    def __init__(cls, name, bases, clsdict):
        if name != "SizedInteger" and "FORMAT" not in clsdict:
            raise ValueError(f"{name} subclasses `SizedInteger` but does not define a `FORMAT` class member")
        super().__init__(name, bases, clsdict)
        if name != "SizedInteger":
            setattr(cls, "BYTES", struct.calcsize(cls.FORMAT))
            setattr(cls, "BITS", cls.BYTES * 8)
            setattr(cls, "SIGNED", cls.FORMAT.islower())
            setattr(cls, "MAX_VALUE", 2**(cls.BITS - [0, 1][cls.SIGNED]) - 1)
            setattr(cls, "MIN_VALUE", [0, -2**(cls.BYTES * 8 - [0, 1][cls.SIGNED])][cls.SIGNED])


class SizedInteger(int, Packable, metaclass=SizedIntegerMeta):
    def __new__(cls: SizedIntegerMeta, value: int):
        retval: SizedInteger = int.__new__(cls, value)
        if not (cls.MIN_VALUE <= retval <= cls.MAX_VALUE):
            raise ValueError(f"{retval} is not in the range [{cls.MIN_VALUE}, {cls.MAX_VALUE}]")
        return retval

    def pack(self) -> bytes:
        return struct.pack(self.FORMAT, self)

    @classmethod
    def unpack(cls, data: bytes) -> "SizedInteger":
        return cls(struct.unpack(cls.FORMAT, data)[0])


class Char(SizedInteger):
    FORMAT = 'b'


class UnsignedChar(SizedInteger):
    FORMAT = 'B'


class Short(SizedInteger):
    FORMAT = 'h'


class UnsignedShort(SizedInteger):
    FORMAT = 'H'


class Int(SizedInteger):
    FORMAT = 'i'


class UnsignedInt(SizedInteger):
    FORMAT = 'I'


class Long(SizedInteger):
    FORMAT = 'l'


class UnsignedLong(SizedInteger):
    FORMAT = 'L'


class LongLong(SizedInteger):
    FORMAT = 'q'


class UnsignedLongLong(SizedInteger):
    FORMAT = 'Q'


Int8 = Char
UInt8 = UnsignedChar
Int16 = Short
UInt16 = UnsignedShort
Int32 = Long
UInt32 = UnsignedLong
Int64 = LongLong
UInt64 = UnsignedLongLong


class StructMeta(ABCMeta):
    def __init__(cls, name, bases, clsdict):
        fields = OrderedDict()
        if "__annotations__" in clsdict:
            for field_name, field_type in clsdict["__annotations__"].items():
                if field_name == "FIELDS":
                    continue
                if isinstance(field_type, Packable):
                    fields[field_name] = field_type
                else:
                    raise TypeError(f"Field {field_name} of {name} must be a SizedInteger or Binary Message, "
                                    f"not {field_type}")
        super().__init__(name, bases, clsdict)
        setattr(cls, "FIELDS", fields)


class Struct(Packable, metaclass=StructMeta):
    FIELDS: OrderedDictType[str, Type[Packable]]

    def __init__(self, *args, **kwargs):
        unsatisfied_fields = [name for name in self.FIELDS.keys() if name not in kwargs]
        if len(args) > len(unsatisfied_fields):
            raise ValueError(f"Unexpected positional argument: {args[len(unsatisfied_fields)]}")
        elif len(args) < len(unsatisfied_fields):
            raise ValueError(f"Missing argument for {unsatisfied_fields[0]}")
        for name, value in kwargs.items():
            setattr(self, name, value)
        for name, value in zip(unsatisfied_fields, args):
            setattr(self, name, value)

    def pack(self) -> bytes:
        return b"".join(getattr(self, field_name).pack() for field_name in self.FIELDS.keys())

    @classmethod
    def unpack(cls: Type[P], data: bytes) -> P:
        return cls(*struct.unpack("".join(field_type.FORMAT for field_type in cls.FIELDS.values()), data))
