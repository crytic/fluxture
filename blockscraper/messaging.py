from abc import ABC, abstractmethod
from typing import Generic, Type, TypeVar

from .types import Struct


M = TypeVar("M", bound="Message")
R = TypeVar("R", bound="Message")


class Message(Generic[R], ABC):
    reply_type: Type[R]

    @abstractmethod
    def serialize(self) -> bytes:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def deserialize(cls: M, data: bytes) -> M:
        raise NotImplementedError()

    def process_reply(self, data: bytes) -> R:
        return self.reply_type.deserialize(data)


class BinaryMessage(Generic[R], Message[R], Struct, ABC):
    def serialize(self) -> bytes:
        return self.pack()

    @classmethod
    def deserialize(cls: M, data: bytes) -> M:
        return cls.unpack(data)
