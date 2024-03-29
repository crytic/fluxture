import asyncio
import socket
from abc import ABCMeta, abstractmethod
from ipaddress import IPv4Address, IPv6Address, ip_address
from typing import (AsyncIterator, Dict, FrozenSet, Generic, Optional, Tuple,
                    Type, TypeVar, Union)

from . import serialization
from .messaging import Message


class BlockchainError(RuntimeError):
    pass


class Miner(serialization.IntEnum):
    UNKNOWN = 0
    MINER = 1
    NOT_MINER = 2


def get_public_ip() -> Union[IPv4Address, IPv6Address]:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    try:
        return ip_address(s.getsockname()[0])
    finally:
        s.close()


class Version:
    def __init__(self, version: str, timestamp: int):
        self.version: str = version
        self.timestamp: int = timestamp


class Node(metaclass=ABCMeta):
    def __init__(
        self,
        address: Union[str, bytes, IPv4Address, IPv6Address],
        port: int,
        source: str = "peer",
    ):
        if not isinstance(address, IPv6Address):
            self.address: IPv6Address = serialization.IPv6Address(address)
        else:
            self.address = address
        self.port: int = port
        self.source: str = source
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._entries: int = 0
        self._stop: Optional[asyncio.Event] = None

    @property
    def is_running(self) -> bool:
        return (
            self._reader is not None
            and self._stop is not None
            and not self._stop.is_set()
        )

    def terminate(self):
        if self._stop is not None:
            self._stop.set()

    async def join(self):
        if self._stop is not None:
            await self._stop.wait()

    @property
    async def reader(self) -> asyncio.StreamReader:
        if self._reader is None:
            await self.connect()
        return self._reader

    @property
    async def writer(self) -> asyncio.StreamWriter:
        if self._writer is None:
            await self.connect()
        return self._writer

    async def connect(self):
        if self._reader is None:
            self._reader, self._writer = await asyncio.open_connection(
                str(self.address),
                self.port,
                happy_eyeballs_delay=0.25,  # this causes IPv4 and IPv6 attempts to be interleaved
            )
            if self._stop is None:
                self._stop = asyncio.Event()
            elif self._stop.is_set():
                self._stop.clear()

    async def close(self):
        if self._writer is not None:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except BrokenPipeError:
                # this is expected
                pass
            self._reader = None
            self._writer = None
            if not self._stop.is_set():
                self._stop.set()

    async def __aenter__(self):
        self._entries += 1
        if self._entries == 1 and self._reader is None:
            await self.connect()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._entries -= 1
        if self._entries == 0 and self._reader is not None:
            await self.close()

    async def send_message(self, message: Message):
        writer = await self.writer
        writer.write(message.serialize())
        await writer.drain()

    def __hash__(self):
        return hash((self.address, self.port))

    def __eq__(self, other):
        return (
            isinstance(other, Node)
            and other.address == self.address
            and other.port == self.port
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(address={self.address!r}, port={self.port!r})"
        )

    @abstractmethod
    async def run(self) -> AsyncIterator[Message]:
        raise NotImplementedError()


N = TypeVar("N", bound=Node)


BLOCKCHAINS: Dict[str, Type["Blockchain[Node]"]] = {}


class Blockchain(Generic[N], metaclass=ABCMeta):
    DEFAULT_SEEDS: Tuple[N, ...] = ()
    name: str
    node_type: Type[N]

    def __init_subclass__(cls, **kwargs):
        if not hasattr(cls, "name") or cls.name is None:
            raise TypeError("Subclasses of `Blockchain` must define a `name`")
        if not hasattr(cls, "node_type") or cls.node_type is None:
            raise TypeError("Subclasses of `Blockchain` must define a `node_type`")
        BLOCKCHAINS[cls.name] = cls

    @classmethod
    @abstractmethod
    async def default_seeds(cls) -> AsyncIterator[N]:
        raise NotImplementedError()

    @abstractmethod
    async def get_neighbors(self, node: N) -> FrozenSet[N]:
        raise NotImplementedError()

    @abstractmethod
    async def get_version(self, node: N) -> Optional[Version]:
        raise NotImplementedError()

    @abstractmethod
    async def is_miner(self, node: N) -> Miner:
        raise NotImplementedError()

    @abstractmethod
    async def get_miners(self) -> FrozenSet[N]:
        raise NotImplementedError()
