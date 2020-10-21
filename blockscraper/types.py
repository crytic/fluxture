import itertools
from abc import ABCMeta, ABC, abstractmethod
from collections import OrderedDict
from collections.abc import ValuesView
from enum import Enum
from typing import (
    Iterator, KeysView, OrderedDict as OrderedDictType, Tuple, Type, TypeVar, ValuesView as ValuesViewType
)
from typing_extensions import Protocol, runtime_checkable
import struct


P = TypeVar("P")


class ByteOrder(Enum):
    NATIVE = "@"
    LITTLE = "<"
    BIG = ">"
    NETWORK = "!"


class UnpackError(RuntimeError):
    pass


@runtime_checkable
class Packable(Protocol):
    def pack(self, byte_order: ByteOrder = ByteOrder.NETWORK) -> bytes:
        ...

    @classmethod
    def unpack(cls: Type[P], data: bytes, byte_order: ByteOrder = ByteOrder.NETWORK) -> P:
        ...

    @classmethod
    def unpack_partial(cls: Type[P], data: bytes, byte_order: ByteOrder = ByteOrder.NETWORK) -> Tuple[P, bytes]:
        ...


class AbstractPackable(ABC):
    @abstractmethod
    def pack(self, byte_order: ByteOrder = ByteOrder.NETWORK) -> bytes:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def unpack_partial(cls: Type[P], data: bytes, byte_order: ByteOrder = ByteOrder.NETWORK) -> Tuple[P, bytes]:
        raise NotImplementedError()

    @classmethod
    def unpack(cls: Type[P], data: bytes, byte_order: ByteOrder = ByteOrder.NETWORK) -> P:
        ret, remaining = cls.unpack_partial(data, byte_order)
        if remaining:
            raise ValueError(f"Unexpected trailing bytes: {remaining!r}")
        return ret


class SizeMeta(type):
    num_bytes_is_defined: bool = False
    dependent_type_is_defined: bool = False

    @property
    def num_bytes(cls) -> int:
        if not cls.num_bytes_is_defined:
            raise TypeError(f"{cls.__name__} must be subscripted with its size when used in a Struct! "
                            f"(E.g., {cls.__name__}[1024] will specify that it is 1024 bytes.)")
        return cls._num_bytes

    @property
    def size_field_name(cls) -> str:
        if not cls.dependent_type_is_defined:
            raise TypeError(f"{cls.__name__} must be subscripted with the name of its size field when used in a Struct!"
                            f" (E.g., {cls.__name__}[\"length\"] will specify that its length is specified by the "
                            "`length` field.)")
        return cls._size_field_name

    def __getitem__(self, item):
        if isinstance(item, int):
            if item < 0:
                raise ValueError(f"Fixed size {item} must be non-negative")
            typename = f"{self.__name__}{item}"
            return type(typename, (self,), {"_num_bytes": item, "num_bytes_is_defined": True})
        elif isinstance(item, str):
            typename = f"{self.__name__}{item}"
            return type(typename, (self,), {
                "_size_field_name": item, "dependent_type_is_defined": True
            })
        else:
            raise KeyError(item)


class Sized(metaclass=SizeMeta):
    num_bytes_is_defined: bool = False
    dependent_type_is_defined: bool = False

    @property
    def num_bytes(self) -> int:
        if self.num_bytes_is_defined:
            return super().num_bytes
        elif self.dependent_type_is_defined:
            raise NotImplementedError()
        else:
            raise ValueError(f"{self} does not have its size set!")

    @property
    def has_size(self) -> bool:
        return self.num_bytes_is_defined or self.dependent_type_is_defined


class SizedByteArray(bytes, Sized):
    @property
    def num_bytes(self) -> int:
        if not self.has_size:
            return len(self)
        else:
            return super().num_bytes

    def __new__(cls, value: bytes, pad: bool = True):
        if cls.num_bytes_is_defined and cls.num_bytes < len(value):
            raise ValueError(f"{cls.__name__} can hold at most {cls.num_bytes} bytes, but {value!r} is longer!")
        elif cls.num_bytes_is_defined and cls.num_bytes > len(value):
            value = value + b"\0" * (cls.num_bytes - len(value))
        return bytes.__new__(cls, value)

    def pack(self, byte_order: ByteOrder = ByteOrder.NETWORK) -> bytes:
        return self

    @classmethod
    def unpack(cls: Type[P], data: bytes, byte_order: ByteOrder = ByteOrder.NETWORK) -> P:
        return cls(data)

    @classmethod
    def unpack_partial(cls: Type[P], data: bytes, byte_order: ByteOrder = ByteOrder.NETWORK) -> Tuple[P, bytes]:
        return cls(data[:cls.num_bytes]), data[cls.num_bytes:]


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
            setattr(cls, "BYTES", struct.calcsize(f"{ByteOrder.NETWORK.value}{cls.FORMAT}"))
            setattr(cls, "BITS", cls.BYTES * 8)
            setattr(cls, "SIGNED", cls.FORMAT.islower())
            setattr(cls, "MAX_VALUE", 2**(cls.BITS - [0, 1][cls.SIGNED]) - 1)
            setattr(cls, "MIN_VALUE", [0, -2**(cls.BITS - 1)][cls.SIGNED])

    @property
    def c_type(cls) -> str:
        return f"{['u',''][cls.SIGNED]}int{cls.BITS}_t"


class SizedInteger(int, metaclass=SizedIntegerMeta):
    def __new__(cls: SizedIntegerMeta, value: int):
        retval: SizedInteger = int.__new__(cls, value)
        if not (cls.MIN_VALUE <= retval <= cls.MAX_VALUE):
            raise ValueError(f"{retval} is not in the range [{cls.MIN_VALUE}, {cls.MAX_VALUE}]")
        return retval

    def pack(self, byte_order: ByteOrder = ByteOrder.NETWORK) -> bytes:
        return struct.pack(f"{byte_order.value}{self.__class__.FORMAT}", self)

    @classmethod
    def unpack(cls, data: bytes, byte_order: ByteOrder = ByteOrder.NETWORK) -> "SizedInteger":
        return cls(struct.unpack(f"{byte_order.value}{cls.FORMAT}", data)[0])

    @classmethod
    def unpack_partial(cls: Type[P], data: bytes, byte_order: ByteOrder = ByteOrder.NETWORK) -> Tuple[P, bytes]:
        try:
            return cls(struct.unpack(f"{byte_order.value}{cls.FORMAT}", data[:cls.BYTES])[0]), data[cls.BYTES:]
        except struct.error:
            pass
        raise UnpackError(f"Error unpacking {cls.__name__} from the front of {data!r}")

    def __str__(self):
        return f"{self.__class__.c_type}({super().__str__()})"


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
Bool = UInt8
Int16 = Short
UInt16 = UnsignedShort
Int32 = Long
UInt32 = UnsignedLong
Int64 = LongLong
UInt64 = UnsignedLongLong


class StructMeta(ABCMeta):
    FIELDS: OrderedDictType[str, Type[Packable]]

    def __init__(cls, name, bases, clsdict):
        fields = OrderedDict()
        if "non_serialized" in clsdict:
            non_serialized = set(clsdict["non_serialized"])
        else:
            non_serialized = set()
        non_serialized |= {"FIELDS", "non_serialized"}
        if "__annotations__" in clsdict:
            for field_name, field_type in clsdict["__annotations__"].items():
                if field_name in non_serialized:
                    continue
                if isinstance(field_type, Packable):
                    fields[field_name] = field_type
                else:
                    raise TypeError(f"Field {field_name} of {name} must be Packable, not {field_type}")
        super().__init__(name, bases, clsdict)
        setattr(cls, "FIELDS", fields)


class Struct(metaclass=StructMeta):
    def __init__(self, *args, **kwargs):
        unsatisfied_fields = [name for name in self.__class__.FIELDS.keys() if name not in kwargs]
        if len(args) > len(unsatisfied_fields):
            raise ValueError(f"Unexpected positional argument: {args[len(unsatisfied_fields)]}")
        elif len(args) < len(unsatisfied_fields):
            raise ValueError(f"Missing argument for {unsatisfied_fields[0]} in {self.__class__}")
        for name, value in itertools.chain(kwargs.items(), zip(unsatisfied_fields, args)):
            if isinstance(value, self.__class__.FIELDS[name]):
                # the value was already passed as the correct type
                setattr(self, name, value)
            else:
                # we need to construct the correct type
                setattr(self, name, self.__class__.FIELDS[name](value))
        super().__init__()

    def pack(self, byte_order: ByteOrder = ByteOrder.NETWORK) -> bytes:
        # TODO: Combine the formats and use a single struct.pack instead
        return b"".join(getattr(self, field_name).pack(byte_order) for field_name in self.__class__.FIELDS.keys())

    @classmethod
    def unpack(cls: Type[P], data: bytes, byte_order: ByteOrder = ByteOrder.NETWORK) -> P:
        ret, remaining = cls.unpack_partial(data, byte_order)
        if remaining:
            raise ValueError(f"Unexpected trailing bytes: {remaining!r}")
        return ret

    @classmethod
    def unpack_partial(cls: Type[P], data: bytes, byte_order: ByteOrder = ByteOrder.NETWORK) -> Tuple[P, bytes]:
        remaining_data = data
        args = []
        for field_name, field_type in cls.FIELDS.items():
            try:
                field, remaining_data = field_type.unpack_partial(remaining_data, byte_order)
                errored = False
            except UnpackError:
                errored = True
            if errored:
                raise UnpackError(f"Error parsing field {cls.__name__}.{field_name} (field {len(args)+1}) of type "
                                  f"{field_type.__name__} from bytes {remaining_data!r}")
            args.append(field)
        return cls(*args), remaining_data

    def __contains__(self, field_name: str):
        return field_name in self.__class__.FIELDS

    def __getitem__(self, field_name: str) -> Packable:
        if field_name not in self:
            raise KeyError(field_name)
        return getattr(self, field_name)

    def __len__(self) -> int:
        return len(self.__class__.FIELDS)

    def __iter__(self) -> Iterator[str]:
        return iter(self.__class__.FIELDS.keys())

    def items(self) -> Iterator[Tuple[str, Packable]]:
        for field_name in self:
            yield field_name, getattr(self, field_name)

    def keys(self) -> KeysView[str]:
        return self.__class__.FIELDS.keys()

    def values(self) -> ValuesViewType[Packable]:
        return ValuesView(self)

    def __eq__(self, other):
        return isinstance(other, Struct) and len(self) == len(other) and all(
            a == b for (_, a), (_, b) in zip(self.items(), other.items())
        )

    def __ne__(self, other):
        return not (self == other)

    def __str__(self):
        types = "".join(f"    {field_name} = {field_value!s};\n" for field_name, field_value in self.items())
        newline = "\n"
        return f"typedef struct {{{['', newline][len(types) > 0]}{types}}} {self.__class__.__name__}"
