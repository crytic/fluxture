import asyncio
import socket
from ipaddress import ip_address, IPv4Address, IPv6Address
from time import time as current_time
from typing import FrozenSet, Optional, Tuple, Type, Union

from .blockchain import Blockchain, get_public_ip, Node
from .messaging import BinaryMessage
from . import types
from .types import ByteOrder, P


class BitcoinMessage(BinaryMessage):
    non_serialized = "byte_order",
    byte_order = ByteOrder.LITTLE


class VarInt(int, types.AbstractPackable):
    def __new__(cls, value: int):
        return int.__new__(cls, value)

    def pack(self, byte_order: types.ByteOrder = types.ByteOrder.LITTLE) -> bytes:
        value = int(self)
        if value < 0xFD:
            return bytes([value])
        elif value <= 0xFFFF:
            return b"\xFD" + types.UInt16(value).pack(byte_order)
        elif value <= 0xFFFFFFFF:
            return b"\xFE" + types.UInt32(value).pack(byte_order)
        elif value <= types.UInt64.MAX_VALUE:
            return b"\xFF" + types.UInt64(value).pack(byte_order)
        else:
            raise ValueError(f"Value {value} must be less than {types.UInt64.MAX_VALUE}")

    @classmethod
    def unpack_partial(cls: Type[P], data: bytes, byte_order: ByteOrder = ByteOrder.LITTLE) -> Tuple[P, bytes]:
        if data[0] < 0xFD:
            return cls(data[0]), data[1:]
        elif data[0] == 0xFD:
            return cls(types.UInt16.unpack(data[1:3], byte_order=byte_order)), data[3:]
        elif data[0] == 0xFE:
            return cls(types.UInt32.unpack(data[1:5], byte_order=byte_order)), data[5:]
        elif data[0] == 0xFF:
            return cls(types.UInt64.unpack(data[1:9], byte_order=byte_order)), data[9:]
        else:
            raise ValueError(f"Unexpected data: {data!r}")


class VarStr(bytes, types.AbstractPackable):
    def __new__(cls, value: bytes):
        return bytes.__new__(cls, value)

    def pack(self, byte_order: types.ByteOrder = types.ByteOrder.LITTLE) -> bytes:
        return VarInt(len(self)).pack(byte_order) + self

    @classmethod
    def unpack_partial(cls: Type[P], data: bytes, byte_order: ByteOrder = ByteOrder.LITTLE) -> Tuple[P, bytes]:
        length, remainder = VarInt.unpack_partial(data, byte_order=byte_order)
        if len(remainder) < length:
            raise ValueError(f"Expected a byte sequence of length {length} but instead got {remainder!r}")
        return remainder[:length], remainder[length:]


class NetAddr(BitcoinMessage):
    time: types.UInt32
    services: types.UInt64
    ip: types.SizedByteArray[16]
    port: types.UInt16

    def __init__(
            self,
            time: Optional[int] = None,
            services: int = 0,
            ip: Optional[Union[IPv4Address, IPv6Address, str, types.SizedByteArray[16]]] = None,
            port: int = 53
    ):
        if ip is None:
            ip = get_public_ip()
        elif isinstance(ip, str):
            ip = ip_address(socket.gethostbyname(ip))
        if not isinstance(ip, types.SizedByteArray):
            # IP is big-endian in Bitcoin
            ip = types.SizedByteArray(int(ip).to_bytes(16, byteorder='big'))
        if time is None:
            time = int(current_time())
        super().__init__(time=time, services=services, ip=ip, port=port)


class VersionMessage(BitcoinMessage):
    version: types.Int32
    services: types.UInt64
    timestamp: types.Int64
    addr_recv: NetAddr
    addr_from: NetAddr
    user_agent: VarStr
    start_height: types.Int32
    relay: types.Bool


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
