import asyncio
import socket
from ipaddress import IPv4Address, IPv6Address
from typing import FrozenSet, Union

from .blockchain import Blockchain, Node
from .messaging import BinaryMessage
from . import types


class NetAddr(BinaryMessage):
    time: types.UInt32
    services: types.UInt64
    ip: bytes
    port: types.UInt16


class VersionMessage(BinaryMessage):
    version: types.Int32
    services: types.UInt64
    timestamp: types.Int64
    addr_recv: NetAddr


class BitcoinNode(Node):
    def __init__(self, address: Union[str, IPv4Address, IPv6Address], port: int = 53):
        super().__init__(address, port)


class Bitcoin(Blockchain[BitcoinNode]):
    DEFAULT_SEEDS = (
        BitcoinNode("dnsseed.bitcoin.dashjr.org"),
        BitcoinNode("dnsseed.bluematt.me"),
        BitcoinNode("seed.bitcoin.jonasschnelli.ch"),
        BitcoinNode("seed.bitcoin.sipa.be"),
        BitcoinNode("seed.bitcoinstats.com"),
        BitcoinNode("seed.btc.petertodd.org")
    )

    async def get_neighbors(self, node: BitcoinNode) -> FrozenSet[BitcoinNode]:
        loop = asyncio.get_running_loop()

        return frozenset(
            BitcoinNode(addr[4][0])
            for addr in await loop.getaddrinfo(str(node.address), node.port, proto=socket.IPPROTO_TCP)
        )
