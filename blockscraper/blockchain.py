import asyncio
import socket
from abc import ABCMeta, abstractmethod
from ipaddress import ip_address, IPv4Address, IPv6Address
from typing import FrozenSet, Generic, Optional, Tuple, TypeVar, Union

from .messaging import Message


def get_public_ip() -> Union[IPv4Address, IPv6Address]:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    try:
        return ip_address(s.getsockname()[0])
    finally:
        s.close()


class Node:
    def __init__(self, address: Union[str, IPv4Address, IPv6Address], port: int):
        if isinstance(address, str):
            self.address: Union[IPv4Address, IPv6Address] = ip_address(socket.gethostbyname(address))
        elif isinstance(address, IPv4Address) or isinstance(address, IPv6Address):
            self.address = address
        else:
            raise ValueError(f"Invalid node address: {address}")
        self.port: int = port
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._entries: int = 0

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
            self._reader, self._writer = await asyncio.open_connection(str(self.address), self.port)

    async def close(self):
        if self._writer is not None:
            self._writer.close()
            await self._writer.wait_closed()
            self._reader = None
            self._writer = None

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
        return isinstance(other, Node) and other.address == self.address and other.port == self.port

    def __repr__(self):
        return f"{self.__class__.__name__}(address={self.address!r}, port={self.port!r})"


N = TypeVar('N', bound=Node)


class Blockchain(Generic[N], metaclass=ABCMeta):
    DEFAULT_SEEDS: Tuple[N] = ()

    @abstractmethod
    async def get_neighbors(self, node: N) -> FrozenSet[N]:
        raise NotImplementedError()
