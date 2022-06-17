import asyncio
from abc import ABC, abstractmethod
from typing import Optional, TypeVar

from fluxture.structures import PackableStruct

from .serialization import ByteOrder

M = TypeVar("M", bound="Message")
B = TypeVar("B", bound="BinaryMessage")


class Message(ABC):
    @abstractmethod
    def serialize(self) -> bytes:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def deserialize(cls: M, data: bytes) -> M:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    async def next_message(cls: M, reader: asyncio.StreamReader) -> Optional[M]:
        raise NotImplementedError()


class BinaryMessage(PackableStruct, Message):
    non_serialized = ("byte_order",)
    byte_order: ByteOrder = ByteOrder.NETWORK

    def serialize(self) -> bytes:
        return self.pack(self.byte_order)

    @classmethod
    def deserialize(cls: B, data: bytes) -> B:
        return cls.unpack(data, cls.byte_order)

    @classmethod
    async def next_message(cls: B, reader: asyncio.StreamReader) -> Optional[B]:
        return cls.read(reader, cls.byte_order)
