import asyncio
import socket
from abc import ABCMeta, abstractmethod
from ipaddress import ip_address, IPv4Address, IPv6Address
from typing import FrozenSet, Generic, Tuple, TypeVar, Union

from .messaging import Message, R


class Node:
    def __init__(self, address: Union[str, IPv4Address, IPv6Address], port: int):
        if isinstance(address, str):
            self.address: Union[IPv4Address, IPv6Address] = ip_address(socket.gethostbyname(address))
        elif isinstance(address, IPv4Address) or isinstance(address, IPv6Address):
            self.address = address
        else:
            raise ValueError(f"Invalid node address: {address}")
        self.port: int = port

    async def send_message(self, message: Message[R]) -> R:
        reader, writer = await asyncio.open_connection(str(self.address), self.port)
        writer.write(message.serialize())
        await writer.drain()
        data = await reader.read()
        writer.close()
        await writer.wait_closed()
        return message.process_reply(data)

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
