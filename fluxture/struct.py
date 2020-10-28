import asyncio
import itertools
from abc import ABCMeta
from collections import OrderedDict
from typing import Generic, Iterator, KeysView, Type, TypeVar, Tuple, ValuesView as ValuesViewType, ValuesView, \
    OrderedDict as OrderedDictType

from fluxture.serialization import ByteOrder, FixedSize, P, Packable, UnpackError


F = TypeVar("F")


class StructMeta(ABCMeta, Generic[F]):
    FIELDS: OrderedDictType[str, Type[F]]

    def __init__(cls, name, bases, clsdict):
        fields = OrderedDict()
        if "non_serialized" in clsdict:
            non_serialized = set(clsdict["non_serialized"])
        else:
            non_serialized = set()
        non_serialized |= {"FIELDS", "non_serialized"}
        if "__annotations__" in clsdict:
            for field_name, field_type in clsdict["__annotations__"].items():
                if field_name not in non_serialized:
                    fields[field_name] = field_type
        super().__init__(name, bases, clsdict)
        cls.validate_fields(fields)
        setattr(cls, "FIELDS", fields)
        # are all fields fixed size? if so, we are fixed size, too!
        if all(hasattr(field, "num_bytes") for field in fields.values()):
            cls.num_bytes = sum(field.num_bytes for field in fields.values())  # type: ignore
            assert isinstance(cls, FixedSize)

    def validate_fields(cls, fields: OrderedDictType[str, Type[F]]):
        pass


class Struct(Generic[F], metaclass=StructMeta[F]):
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

    def __contains__(self, field_name: str):
        return field_name in self.__class__.FIELDS

    def __getitem__(self, field_name: str) -> Type[F]:
        if field_name not in self:
            raise KeyError(field_name)
        return getattr(self, field_name)

    def __len__(self) -> int:
        return len(self.__class__.FIELDS)

    def __iter__(self) -> Iterator[str]:
        return iter(self.__class__.FIELDS.keys())

    def items(self) -> Iterator[Tuple[str, Type[F]]]:
        for field_name in self:
            yield field_name, getattr(self, field_name)

    def keys(self) -> KeysView[str]:
        return self.__class__.FIELDS.keys()

    def values(self) -> ValuesViewType[Type[F]]:
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

    def __repr__(self):
        args = [f"{name}={getattr(self, name)!r}" for name in self.__class__.FIELDS.keys()]
        return f"{self.__class__.__name__}({', '.join(args)})"


class PackableStruct(Generic[P], Struct[P]):
    def pack(self, byte_order: ByteOrder = ByteOrder.NETWORK) -> bytes:
        # TODO: Combine the formats and use a single struct.pack instead
        return b"".join(getattr(self, field_name).pack(byte_order) for field_name in self.__class__.FIELDS.keys())

    @classmethod
    def validate_fields(cls, fields: OrderedDictType[str, Type[F]]):
        for field_name, field_type in fields.items():
            if not isinstance(field_type, Packable):
                raise TypeError(f"Field {field_name} of {cls.__name__} must be Packable, not {field_type}")

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
                parsed_fields = [f"{field_name} = {arg!r}" for field_name, arg in zip(cls.FIELDS.keys(), args)]
                parsed_fields = ", ".join(parsed_fields)
                raise UnpackError(f"Error parsing field {cls.__name__}.{field_name} (field {len(args)+1}) of type "
                                  f"{field_type.__name__} from bytes {remaining_data!r}. Prior parsed field values: "
                                  f"{parsed_fields}")
            args.append(field)
        return cls(*args), remaining_data

    @classmethod
    async def read(cls: Type[P], reader: asyncio.StreamReader, byte_order: ByteOrder = ByteOrder.NETWORK) -> P:
        if hasattr(cls, "num_bytes"):
            data = await reader.read(cls.num_bytes)
            return cls.unpack(data, byte_order)
        # we need to read it one field at a time
        args = []
        for field_name, field_type in cls.FIELDS.items():
            try:
                field = field_type.read(reader, byte_order)
                errored = False
            except UnpackError:
                errored = True
            if errored:
                parsed_fields = [f"{field_name} = {arg!r}" for field_name, arg in zip(cls.FIELDS.keys(), args)]
                parsed_fields = ", ".join(parsed_fields)
                raise UnpackError(f"Error parsing field {cls.__name__}.{field_name} (field {len(args) + 1}) of type "
                                  f"{field_type.__name__}. Prior parsed field values: {parsed_fields}")
            args.append(field)
        return cls(*args)
