import asyncio
import ipaddress
from abc import ABCMeta, ABC, abstractmethod
from enum import Enum
from typing import (
    Tuple, Type, TypeVar, Union
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

    @classmethod
    async def read(cls: Type[P], reader: asyncio.StreamReader, byte_order: ByteOrder = ByteOrder.NETWORK) -> P:
        ...


class BigEndian:
    def __class_getitem__(cls, item: Type[Packable]):
        def big_endian_pack(self, byte_order: ByteOrder = ByteOrder.BIG) -> bytes:
            return item.pack(self, byte_order=ByteOrder.BIG)

        def big_endian_unpack(data: bytes, byte_order: ByteOrder = ByteOrder.BIG):
            return item.unpack(data, byte_order=ByteOrder.BIG)

        def big_endian_unpack_partial(data: bytes, byte_order: ByteOrder = ByteOrder.BIG):
            return item.unpack_partial(data, byte_order=ByteOrder.BIG)

        async def big_endian_read(reader: asyncio.StreamReader, byte_order: ByteOrder = ByteOrder.BIG):
            return item.read(reader, byte_order=ByteOrder.BIG)

        return type(f"{item.__name__}BigEndian", (item,), {
            "pack": big_endian_pack,
            "unpack": big_endian_unpack,
            "unpack_partial": big_endian_unpack_partial,
            "read": big_endian_read
        })


class LittleEndian:
    def __class_getitem__(cls, item: Type[Packable]):
        def little_endian_pack(self, byte_order: ByteOrder = ByteOrder.LITTLE) -> bytes:
            return item.pack(self, byte_order=ByteOrder.LITTLE)

        def little_endian_unpack(data: bytes, byte_order: ByteOrder = ByteOrder.LITTLE):
            return item.unpack(data, byte_order=ByteOrder.LITTLE)

        def little_endian_unpack_partial(data: bytes, byte_order: ByteOrder = ByteOrder.LITTLE):
            return item.unpack_partial(data, byte_order=ByteOrder.LITTLE)

        async def little_endian_read(reader: asyncio.StreamReader, byte_order: ByteOrder = ByteOrder.LITTLE):
            return item.read(reader, byte_order=ByteOrder.LITTLE)

        return type(f"{item.__name__}LittleEndian", (item,), {
            "pack": little_endian_pack,
            "unpack": little_endian_unpack,
            "unpack_partial": little_endian_unpack_partial,
            "read": little_endian_read
        })


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

    @classmethod
    @abstractmethod
    async def read(cls: Type[P], reader: asyncio.StreamReader, byte_order: ByteOrder = ByteOrder.NETWORK) -> P:
        raise NotImplementedError()


@runtime_checkable
class FixedSize(Protocol):
    num_bytes: int


class IPv6Address(ipaddress.IPv6Address, AbstractPackable):
    num_bytes: int = 16

    def __init__(self, address: Union[str, bytes, ipaddress.IPv6Address, ipaddress.IPv4Address]):
        if isinstance(address, str) or isinstance(address, bytes):
            address = ipaddress.ip_address(address)
        if isinstance(address, ipaddress.IPv4Address):
            # convert ip4 to rfc 3056 IPv6 6to4 address
            # http://tools.ietf.org/html/rfc3056#section-2
            prefix6to4 = int(ipaddress.IPv6Address("2002::"))
            ipv4 = address
            address = ipaddress.IPv6Address(prefix6to4 | (int(ipv4) << 80))
            assert address.sixtofour == ipv4
        super().__init__(address.packed)

    def pack(self, byte_order: ByteOrder = ByteOrder.BIG) -> bytes:
        if byte_order == ByteOrder.BIG:
            return self.packed
        else:
            return bytes(reversed(self.packed))

    @classmethod
    def unpack_partial(cls: Type[P], data: bytes, byte_order: ByteOrder = ByteOrder.NETWORK) -> Tuple[P, bytes]:
        if byte_order == ByteOrder.BIG:
            return cls(data[:16]), data[16:]
        else:
            return cls(bytes(reversed(data[:16]))), data[16:]

    @classmethod
    async def read(cls: Type[P], reader: asyncio.StreamReader, byte_order: ByteOrder = ByteOrder.NETWORK) -> P:
        return cls.unpack(await reader.readexactly(cls.num_bytes), byte_order=byte_order)


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

    @classmethod
    async def read(cls: Type[P], reader: asyncio.StreamReader, byte_order: ByteOrder = ByteOrder.NETWORK) -> P:
        data = await reader.read(cls.num_bytes)
        return cls.unpack(data, byte_order)


class SizedIntegerMeta(ABCMeta):
    FORMAT: str
    BITS: int
    BYTES: int
    SIGNED: bool
    MAX_VALUE: int
    MIN_VALUE: int

    def __init__(cls, name, bases, clsdict):
        if name != "SizedInteger" and "FORMAT" not in clsdict and (not isinstance(cls.FORMAT, str) or not cls.FORMAT):
            raise ValueError(f"{name} subclasses `SizedInteger` but does not define a `FORMAT` class member")
        super().__init__(name, bases, clsdict)
        if name != "SizedInteger":
            setattr(cls, "BYTES", struct.calcsize(f"{ByteOrder.NETWORK.value}{cls.FORMAT}"))
            setattr(cls, "BITS", cls.BYTES * 8)
            setattr(cls, "SIGNED", cls.FORMAT.islower())
            setattr(cls, "MAX_VALUE", 2**(cls.BITS - [0, 1][cls.SIGNED]) - 1)
            setattr(cls, "MIN_VALUE", [0, -2**(cls.BITS - 1)][cls.SIGNED])

    @property
    def num_bytes(cls) -> int:
        return cls.BYTES

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

    @classmethod
    async def read(cls: Type[P], reader: asyncio.StreamReader, byte_order: ByteOrder = ByteOrder.NETWORK) -> P:
        data = await reader.read(cls.num_bytes)
        return cls.unpack(data, byte_order)

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
