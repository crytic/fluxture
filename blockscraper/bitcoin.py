import asyncio
import socket
from hashlib import sha256
from ipaddress import ip_address, IPv4Address, IPv6Address
from time import time as current_time
from typing import Dict, FrozenSet, Generic, Optional, Tuple, Type, TypeVar, Union

from .blockchain import Blockchain, get_public_ip, Node
from .messaging import BinaryMessage
from . import serialization
from .serialization import ByteOrder, P, UnpackError


BITCOIN_MAINNET_MAGIC = b"\xf9\xbe\xb4\xd9"


B = TypeVar('B', bound="BitcoinMessage")


class BitcoinMessageHeader(BinaryMessage):
    non_serialized = "byte_order"
    byte_order = ByteOrder.LITTLE

    magic: serialization.SizedByteArray[4]
    command: serialization.SizedByteArray[12]
    length: serialization.UInt32
    checksum: serialization.SizedByteArray[4]

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


class BitcoinMessage(Generic[B], BinaryMessage[B]):
    non_serialized = "byte_order", "command", "reply_type"
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


class VarInt(int, serialization.AbstractPackable):
    def __new__(cls, value: int):
        return int.__new__(cls, value)

    def pack(self, byte_order: serialization.ByteOrder = serialization.ByteOrder.LITTLE) -> bytes:
        value = int(self)
        if value < 0xFD:
            return bytes([value])
        elif value <= 0xFFFF:
            return b"\xFD" + serialization.UInt16(value).pack(byte_order)
        elif value <= 0xFFFFFFFF:
            return b"\xFE" + serialization.UInt32(value).pack(byte_order)
        elif value <= serialization.UInt64.MAX_VALUE:
            return b"\xFF" + serialization.UInt64(value).pack(byte_order)
        else:
            raise ValueError(f"Value {value} must be less than {serialization.UInt64.MAX_VALUE}")

    @classmethod
    def unpack_partial(cls: Type[P], data: bytes, byte_order: ByteOrder = ByteOrder.LITTLE) -> Tuple[P, bytes]:
        if data[0] < 0xFD:
            return cls(data[0]), data[1:]
        elif data[0] == 0xFD:
            return cls(serialization.UInt16.unpack(data[1:3], byte_order=byte_order)), data[3:]
        elif data[0] == 0xFE:
            return cls(serialization.UInt32.unpack(data[1:5], byte_order=byte_order)), data[5:]
        elif data[0] == 0xFF:
            return cls(serialization.UInt64.unpack(data[1:9], byte_order=byte_order)), data[9:]
        else:
            raise ValueError(f"Unexpected data: {data!r}")


class VarStr(bytes, serialization.AbstractPackable):
    def __new__(cls, value: bytes):
        return bytes.__new__(cls, value)

    def pack(self, byte_order: serialization.ByteOrder = serialization.ByteOrder.LITTLE) -> bytes:
        return VarInt(len(self)).pack(byte_order) + self

    @classmethod
    def unpack_partial(cls: Type[P], data: bytes, byte_order: ByteOrder = ByteOrder.LITTLE) -> Tuple[P, bytes]:
        length, remainder = VarInt.unpack_partial(data, byte_order=byte_order)
        if len(remainder) < length:
            raise UnpackError(f"Expected a byte sequence of length {length} but instead got {remainder!r}")
        return remainder[:length], remainder[length:]


class NetAddr(serialization.Struct):
    services: serialization.UInt64
    ip: serialization.BigEndian[serialization.IPv6Address]
    port: serialization.BigEndian[serialization.UInt16]

    def __init__(
            self,
            services: int = 0,
            ip: Optional[Union[serialization.IPv6Address, str, bytes]] = None,
            port: int = 8333
    ):
        if ip is None:
            ip = get_public_ip()
        if not isinstance(ip, serialization.IPv6Address):
            # IP is big-endian in Bitcoin
            ip = serialization.IPv6Address(ip)
        super().__init__(services=services, ip=ip, port=port)


class VerackMessage(BitcoinMessage):
    command = "verack"


class VersionMessage(BitcoinMessage[VerackMessage]):
    command = "version"
    reply_type = VerackMessage

    version: serialization.Int32
    services: serialization.UInt64
    timestamp: serialization.Int64
    addr_recv: NetAddr
    addr_from: NetAddr
    nonce: serialization.UInt64
    user_agent: VarStr
    start_height: serialization.Int32
    relay: serialization.Bool


class BitcoinNode(Node):
    def __init__(self, address: Union[str, IPv4Address, IPv6Address], port: int = 8333):
        super().__init__(address, port)

    async def connect(self):
        t = int(current_time())
        verack = await self.send_message(VersionMessage(
            version=70015,
            services=0,
            timestamp=t,
            addr_recv=NetAddr(ip=self.address, port=self.port),
            addr_from=NetAddr(ip="::ffff:127.0.0.1", port=8333),
            nonce=0,
            user_agent=b"BlockScraper",
            start_height=0,
            relay=True
        ))
        assert isinstance(verack, VerackMessage)


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
