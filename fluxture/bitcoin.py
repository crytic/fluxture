import asyncio
import socket
from abc import ABC
from hashlib import sha256
from ipaddress import IPv4Address, IPv6Address
from time import time as current_time
from typing import AsyncIterator, Dict, FrozenSet, Generic, List, Optional, Tuple, Type, TypeVar, Union

import fluxture.structures

from .blockchain import Blockchain, get_public_ip, Node
from .messaging import BinaryMessage
from . import serialization
from .serialization import ByteOrder, P, UnpackError


BITCOIN_MAINNET_MAGIC = b"\xf9\xbe\xb4\xd9"


B = TypeVar('B', bound="BitcoinMessage")


class BitcoinError(RuntimeError):
    pass


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

    @classmethod
    async def next_message(cls, reader: asyncio.StreamReader) -> Optional["BitcoinMessageHeader"]:
        data = await reader.read(4 + 12 + serialization.UInt32.BYTES + 4)
        if not data:
            return None
        return cls.deserialize(data)

    def __repr__(self):
        return f"{self.__class__.__name__}(magic={self.magic!r}, command={self.decoded_command!r}, "\
               f"length={self.length!r}, checksum={self.checksum!r})"


MESSAGES_BY_COMMAND: Dict[str, Type["BitcoinMessage"]] = {}


def bitcoin_checksum(payload: bytes) -> bytes:
    return sha256(sha256(payload).digest()).digest()[:4]


class BitcoinMessage(BinaryMessage, ABC):
    non_serialized = "byte_order", "command"
    byte_order = ByteOrder.LITTLE
    command: Optional[str] = None

    def __init_subclass__(cls, **kwargs):
        if cls.command is None:
            raise TypeError(f"{cls.__name__} extends BitcoinMessage but does not specify a command string!")
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
    def deserialize_partial(
            cls,
            data: bytes,
            header: Optional[BitcoinMessageHeader] = None
    ) -> Tuple["BitcoinMessage", bytes]:
        if header is None:
            header, payload = BitcoinMessageHeader.unpack_partial(data, byte_order=BitcoinMessageHeader.byte_order)
        else:
            payload = data
        if header.magic != BITCOIN_MAINNET_MAGIC:
            raise ValueError(f"Message header magic was {header.magic}, but expected {BITCOIN_MAINNET_MAGIC!r} "
                             "for Bitcoin mainnet!")
        elif header.length < len(payload):
            raise ValueError(f"Invalid payload length of {len(payload)}; expected at least {header.length} bytes")
        elif header.decoded_command not in MESSAGES_BY_COMMAND:
            raise NotImplementedError(f"TODO: Implement Bitcoin command \"{header.command}\"")
        payload, remainder = payload[:header.length], payload[header.length:]
        decoded_command = header.decoded_command
        expected_checksum = bitcoin_checksum(payload)
        if header.checksum != expected_checksum:
            raise ValueError(f"Invalid message checksum; got {header.checksum!r} but expected {expected_checksum!r}")
        return (
            MESSAGES_BY_COMMAND[decoded_command].unpack(payload, MESSAGES_BY_COMMAND[decoded_command].byte_order),
            remainder
        )

    @classmethod
    def deserialize(cls, data: bytes) -> "BitcoinMessage":
        message, remainder = cls.deserialize_partial(data)
        if remainder:
            raise ValueError(f"Unexpected bytes trailing message: {remainder!r}")
        return message

    @classmethod
    async def next_message(cls, reader: asyncio.StreamReader) -> Optional["BitcoinMessage"]:
        header = await BitcoinMessageHeader.next_message(reader)
        if header is None:
            return None
        try:
            payload = await reader.readexactly(header.length)
        except asyncio.IncompleteReadError:
            raise ValueError(f"Expected {header.length} bytes when reading the message with header {header!r}")
        msg, remainder = cls.deserialize_partial(payload, header=header)
        assert len(remainder) == 0
        return msg


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
            raise UnpackError(f"Unexpected data: {data!r}")

    @classmethod
    async def read(cls: Type[P], reader: asyncio.StreamReader, byte_order: ByteOrder = ByteOrder.NETWORK) -> P:
        first_byte = await reader.read(1)
        if len(first_byte) < 1:
            raise BitcoinError()
        elif first_byte[0] < 0xFD:
            return cls(first_byte[0])
        elif first_byte[0] == 0xFD:
            data_type = serialization.UInt16
        elif first_byte[0] == 0xFE:
            data_type = serialization.UInt32
        elif first_byte[0] == 0xFF:
            data_type = serialization.UInt64
        else:
            raise BitcoinError()
        return cls(await data_type.read(reader, byte_order=byte_order))


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

    @classmethod
    async def read(cls: Type[P], reader: asyncio.StreamReader, byte_order: ByteOrder = ByteOrder.NETWORK) -> P:
        length = await VarInt.read(reader, byte_order=byte_order)
        string = await reader.read(length)
        if len(string) < length:
            raise UnpackError(f"Expected a byte sequence of length {length} but instead got {string!r}")
        return string


class NetAddr(fluxture.structures.PackableStruct):
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


class NetIP(fluxture.structures.PackableStruct):
    time: serialization.UInt32
    addr: NetAddr

    def __init__(
            self,
            time: Optional[int] = None,
            addr: Optional[NetAddr] = None
    ):
        if time is None:
            time = int(current_time())
        if addr is None:
            addr = NetAddr()
        super().__init__(time=time, addr=addr)


class VerackMessage(BitcoinMessage):
    command = "verack"


class SendHeaders(BitcoinMessage):
    command = "sendheaders"


class SendCmpct(BitcoinMessage):
    command = "sendcmpct"

    announce: serialization.Bool
    version: serialization.UInt64


class Ping(BitcoinMessage):
    command = "ping"

    nonce: serialization.UInt64


class Pong(BitcoinMessage):
    command = "pong"

    nonce: serialization.UInt64


class VersionMessage(BitcoinMessage):
    command = "version"

    version: serialization.Int32
    services: serialization.UInt64
    timestamp: serialization.Int64
    addr_recv: NetAddr
    addr_from: NetAddr
    nonce: serialization.UInt64
    user_agent: VarStr
    start_height: serialization.Int32
    relay: serialization.Bool


class FeeFilter(BitcoinMessage):
    command = "feefilter"

    feerate: serialization.UInt64


class GetAddrMessage(BitcoinMessage):
    command = "getaddr"


class MempoolMessage(BitcoinMessage):
    command = "mempool"


class AbstractList(list, Generic[P], List[P], serialization.AbstractPackable, ABC):
    ELEMENT_TYPE: Type[P]

    def __new__(cls, *args, **kwargs):
        return list.__new__(cls, *args, **kwargs)

    def pack(self, byte_order: ByteOrder = ByteOrder.NETWORK) -> bytes:
        return VarInt(len(self)).pack(byte_order) + b"".join(element.pack(byte_order) for element in self)

    @classmethod
    def unpack_partial(cls: Type[P], data: bytes, byte_order: ByteOrder = ByteOrder.NETWORK) -> Tuple[P, bytes]:
        length, remainder = VarInt.unpack_partial(data, byte_order)
        num_bytes = length * cls.ELEMENT_TYPE.num_bytes
        if num_bytes > len(remainder):
            raise UnpackError(f"Expected {num_bytes} bytes, but got {remainder!r}")
        iters = [iter(remainder[:num_bytes])] * cls.ELEMENT_TYPE.num_bytes
        return cls(
            cls.ELEMENT_TYPE.unpack(bytes(data), byte_order=byte_order) for data in zip(*iters)
        ), remainder[num_bytes:]

    @classmethod
    async def read(cls: Type[P], reader: asyncio.StreamReader, byte_order: ByteOrder = ByteOrder.NETWORK) -> P:
        length = await VarInt.read(reader, byte_order)
        return cls(cls.ELEMENT_TYPE.read(reader, byte_order=byte_order) for _ in range(length))


class AddressList(AbstractList[NetIP]):
    ELEMENT_TYPE = NetIP


class Identifier(serialization.UInt32):
    MSG_TX = serialization.UInt32(1)
    MSG_BLOCK = serialization.UInt32(2)
    MSG_FILTERED_BLOCK = serialization.UInt32(3)
    MSG_CMPCT_BLOCK = serialization.UInt32(4)
    MSG_WITNESS_TX = serialization.UInt32((1 << 30) | 1)
    MSG_WITNESS_BLOCK = serialization.UInt32((1 << 30) | 2)
    MSG_FILTERED_WITNESS_BLOCK = serialization.UInt32((1 << 30) | 3)


class Inventory(fluxture.structures.PackableStruct):
    identifier: Identifier
    hash: serialization.SizedByteArray[32]


class Inventories(AbstractList[Inventory]):
    ELEMENT_TYPE = Inventory


class InvMessage(BitcoinMessage):
    command = "inv"

    inventories: Inventories


class AddrMessage(BitcoinMessage):
    command = "addr"

    addresses: AddressList


class BitcoinNode(Node):
    def __init__(self, address: Union[str, IPv4Address, IPv6Address], port: int = 8333):
        super().__init__(address, port)
        self.connected: bool = False
        self.connecting: bool = False

    async def receive_message(self) -> Optional["BitcoinMessage"]:
        return await BitcoinMessage.next_message(await self.reader)

    async def connect(self):
        if self.connected or self.connecting:
            return
        await super().connect()
        self.connecting = True
        t = int(current_time())
        await self.send_message(VersionMessage(
            version=70015,
            services=0,
            timestamp=t,
            addr_recv=NetAddr(ip=self.address, port=self.port),
            addr_from=NetAddr(ip="::ffff:127.0.0.1", port=8333),
            nonce=0,
            user_agent=b"fluxture",
            start_height=0,
            relay=True
        ))
        async for reply in self.run():
            if isinstance(reply, VerackMessage):
                self.connected = True
                break
        if not self.connected:
            raise BitcoinError(f"Did not receive a Verack message from client {self.address}:{self.port}")
        self.connecting = False

    async def get_neighbors(self) -> AddrMessage:
        async with self:
            await self.send_message(GetAddrMessage())
            async for msg in self.run():
                if isinstance(msg, AddrMessage):
                    return msg
        raise BitcoinError(f"Node {self.address}:{self.port} closed the connection before replying to our "
                           "GetAddr message")

    async def get_mempool(self) -> InvMessage:
        async with self:
            await self.send_message(MempoolMessage())
            async for msg in self.run():
                if isinstance(msg, InvMessage):
                    return [i.hash for i in msg.inventories]
        raise BitcoinError(f"Node {self.address}:{self.port} closed the connection before replying to our "
                           "MempoolMessage message")

    async def run(self) -> AsyncIterator["BitcoinMessage"]:
        async with self:
            await self.connect()
            while True:
                done, pending = await asyncio.wait(
                    [self.join(), self.receive_message()],
                    return_when=asyncio.FIRST_COMPLETED
                )
                gather = asyncio.gather(*pending)
                gather.cancel()
                try:
                    await gather
                except asyncio.CancelledError:
                    pass
                message = done.pop().result()
                if message is None:
                    break
                elif self.is_running:
                    # print(f"{self.address}:{self.port} {message}")
                    if isinstance(message, VersionMessage):
                        await self.send_message(VerackMessage())
                    elif isinstance(message, Ping):
                        await self.send_message(Pong(nonce=message.nonce))
                    yield message


async def collect_addresses(url: str, port: int = 8333) -> Tuple[BitcoinNode, ...]:
    return tuple(
        BitcoinNode(addr[4][0])
        for addr in await asyncio.get_running_loop().getaddrinfo(url, port, proto=socket.IPPROTO_TCP)
    )


async def collect_defaults(*args: Union[Tuple[str], Tuple[str, int]]) -> Tuple[BitcoinNode, ...]:
    results = []
    for result in await asyncio.gather(*(collect_addresses(*arg) for arg in args)):
        results.extend(result)
    return tuple(results)


class Bitcoin(Blockchain[BitcoinNode]):
    name = "bitcoin"
    node_type = BitcoinNode
    DEFAULT_SEEDS = asyncio.run(collect_defaults(
        ("dnsseed.bitcoin.dashjr.org",),
        ("dnsseed.bluematt.me",),
        ("seed.bitcoin.jonasschnelli.ch",),
        ("seed.bitcoin.sipa.be",),
        ("seed.bitcoinstats.com",),
        ("seed.btc.petertodd.org",)
    ))

    async def get_neighbors(self, node: BitcoinNode) -> FrozenSet[BitcoinNode]:
        assert node.is_running
        neighbor_addrs = await node.get_neighbors()
        return frozenset(
            BitcoinNode(addr.addr.ip, addr.addr.port)
            for addr in neighbor_addrs.addresses
            if addr.addr.ip != node.address or addr.addr.port != node.port
        )

    async def get_pending_transactions(self, node: BitcoinNode) -> FrozenSet[bytes]:
        assert node.is_running
        mempool = await node.get_mempool()
        return frozenset(mempool)
