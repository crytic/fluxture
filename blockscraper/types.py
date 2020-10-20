from abc import ABC, ABCMeta, abstractmethod
from collections import OrderedDict
from typing import OrderedDict as OrderedDictType, Type, TypeVar
import struct


P = TypeVar("P")


class Packable(metaclass=ABCMeta):
    @abstractmethod
    def pack(self) -> bytes:
        raise NotImplementedError()

    @classmethod
    def unpack(cls: Type[P], data: bytes) -> P:
        raise NotImplementedError()


class SizedInteger(int, Packable, ABC):
    FORMAT: str
    BITS: int
    SIGNED: bool

    def __init__(self, value: int):
        if 2**self.BITS - 1 < value:
            raise ValueError(f"{value} is more than {self.BITS} bits!")
        elif value < 0 and not self.SIGNED:
            raise ValueError(f"{value} is negative but the integer is not signed!")
        super().__init__(value)

    def pack(self) -> bytes:
        return struct.pack(self.FORMAT, int(self))

    @classmethod
    def unpack(cls, data: bytes) -> "SizedInteger":
        return cls(struct.unpack(cls.FORMAT, data)[0])


class Char(SizedInteger):
    FORMAT = 'c'
    BITS = 1
    SIGNED = True


class UnsignedChar(SizedInteger):
    FORMAT = 'c'
    BITS = 1
    SIGNED = False


class Short(SizedInteger):
    FORMAT = 'h'
    BITS = 2
    SIGNED = True


class UnsignedShort(SizedInteger):
    FORMAT = 'H'
    BITS = 2
    SIGNED = False


class Int(SizedInteger):
    FORMAT = 'i'
    BITS = 4
    SIGNED = True


class UnsignedInt(SizedInteger):
    FORMAT = 'I'
    BITS = 4
    SIGNED = False


class Long(SizedInteger):
    FORMAT = 'l'
    BITS = 4
    SIGNED = True


class UnsignedLong(SizedInteger):
    FORMAT = 'L'
    BITS = 4
    SIGNED = False


class LongLong(SizedInteger):
    FORMAT = 'q'
    BITS = 8
    SIGNED = True


class UnsignedLongLong(SizedInteger):
    FORMAT = 'Q'
    BITS = 8
    SIGNED = False


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
