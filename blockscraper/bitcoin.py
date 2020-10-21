import asyncio
import socket
from hashlib import sha256
from ipaddress import ip_address, IPv4Address, IPv6Address
from time import time as current_time
from typing import Dict, FrozenSet, Optional, Tuple, Type, TypeVar, Union

from .blockchain import Blockchain, get_public_ip, Node
from .messaging import BinaryMessage
from . import types
from .types import ByteOrder, P


BITCOIN_MAINNET_MAGIC = b"\xf9\xbe\xb4\xd9"


B = TypeVar('B', bound="BitcoinMessage")


class BitcoinMessageHeader(BinaryMessage):
    non_serialized = "byte_order"
    byte_order = ByteOrder.LITTLE

    magic: types.SizedByteArray[4]
    command: types.SizedByteArray[12]
    length: types.UInt32
    checksum: types.SizedByteArray[4]

    @property
    def decoded_command(self) -> str:
        decoded = self.command.decode("utf-8")
        first_null_byte = decoded.find("\0")
        if any(c != "\0" for c in decoded[first_null_byte:]):
            raise ValueError(f"Command name {self.command!r} includes bytes after the null terminator!")
        return decoded[:first_null_byte]


MESSAGES_BY_COMMAND: Dict[str, Type["BitcoinMessage"]] = {}


def bitcoin_checksum(payload: bytes) -> bytes:
    return sha256(sha256(payload).digest()).digest()[:4]


class BitcoinMessage(BinaryMessage):
    non_serialized = "byte_order", "command"
    byte_order = ByteOrder.LITTLE
    command: Optional[str] = None

    def __init_subclass__(cls, **kwargs):
        if cls.command is None:
            raise TypeError(f"{cls.__name__} extends BitcoinMessage but does not speficy a command string!")
        elif cls.command in MESSAGES_BY_COMMAND:
            raise TypeError(f"The command {cls.command} is already registered to message class "
                            f"{MESSAGES_BY_COMMAND[cls.command]}")
        MESSAGES_BY_COMMAND[cls.command] = cls

    def serialize(self) -> bytes:
        payload = super().serialize()
        return BitcoinMessageHeader(
            magic=BITCOIN_MAINNET_MAGIC,
            command=self.command.encode("utf-8"),
            length=len(payload),
            checksum=bitcoin_checksum(payload)
        ).serialize() + payload

    @classmethod
    def deserialize(cls: B, data: bytes) -> B:
        header, payload = BitcoinMessageHeader.unpack_partial(data, byte_order=BitcoinMessageHeader.byte_order)
        if header.magic != BITCOIN_MAINNET_MAGIC:
            raise ValueError(f"Message header magic was {header.magic}, but expected {BITCOIN_MAINNET_MAGIC!r} "
                             "for Bitcoin mainnet!")
        elif header.length != len(payload):
            raise ValueError(f"Invalid length value of {header.length}; expected {len(payload)}")
        elif header.decoded_command not in MESSAGES_BY_COMMAND:
            raise NotImplementedError(f"TODO: Implement Bitcoin command \"{header.command}\"")
        decoded_command = header.decoded_command
        expected_checksum = bitcoin_checksum(payload)
        if header.checksum != expected_checksum:
            raise ValueError(f"Invalid message checksum; got {header.checksum!r} but expected {expected_checksum!r}")
        return MESSAGES_BY_COMMAND[decoded_command].unpack(payload, MESSAGES_BY_COMMAND[decoded_command].byte_order)


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


class NetAddr(types.Struct):
    time: types.UInt32
    services: types.UInt64
    ip: types.SizedByteArray[16]
    port: types.BigEndian[types.UInt16]

    def __init__(
            self,
            time: Optional[int] = None,
            services: int = 0,
            ip: Optional[Union[IPv4Address, IPv6Address, str, types.SizedByteArray[16]]] = None,
            port: int = 8333
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
    command = "version"
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
