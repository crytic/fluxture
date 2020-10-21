import time
from ipaddress import ip_address
from unittest import TestCase

from blockscraper.bitcoin import BitcoinMessage, NetAddr, VersionMessage
from blockscraper.serialization import ByteOrder

EXAMPLE_VERSION_MESSAGE = b"".join([
    b"\x72\x11\x01\x00",                  # Protocol version: 70002
    b"\x01\x00\x00\x00\x00\x00\x00\x00",  # Services: NODE_NETWORK
    b"\xbc\x8f\x5e\x54\x00\x00\x00\x00",  # [Epoch time][unix epoch time]: 1415483324
    b"\x01\x00\x00\x00\x00\x00\x00\x00",  # Receiving node's services
    b"\x00\x00\x00\x00\x00\x00\x00\x00",
    b"\x00\x00\xff\xff\xc6\x1b\x64\x09",  # Receiving node's IPv6 address
    b"\x20\x8d",                          # Receiving node's port number
    b"\x01\x00\x00\x00\x00\x00\x00\x00",  # Transmitting node's services
    b"\x00\x00\x00\x00\x00\x00\x00\x00",
    b"\x00\x00\xff\xff\xcb\x00\x71\xc0",  # Transmitting node's IPv6 address
    b"\x20\x8d",                          # Transmitting node's port number
    b"\x12\x80\x35\xcb\xc9\x79\x53\xf8",  # Nonce
    b"\x0F"                               # Bytes in user agent string: 15
    b"\x2f\x53\x61\x74\x6f\x73\x68\x69",
    b"\x3a\x30\x2e\x39\x2e\x33\x2f",      # User agent: /Satoshi:0.9.3/
    b"\xcf\x05\x05\x00",                  # Start height: 329167
    b"\x01",                              # Relay flag: true
])


class TestBitcoin(TestCase):
    def test_version_message(self):
        msg = VersionMessage(
            version=70015,
            services=0,
            timestamp=int(time.time()),
            addr_recv=NetAddr(),
            addr_from=NetAddr(),
            nonce=0,
            user_agent=b"BlockScraper",
            start_height=123,
            relay=True
        )
        self.assertEqual(msg, BitcoinMessage.deserialize(msg.serialize()))
        msg = VersionMessage.unpack(EXAMPLE_VERSION_MESSAGE, byte_order=ByteOrder.LITTLE)
        self.assertIsInstance(msg, VersionMessage)
        self.assertEqual(msg.version, 70002)
        self.assertEqual(msg.timestamp, 1415483324)
        self.assertEqual(msg.addr_recv.port, 8333)
        self.assertEqual(msg.addr_from.port, 8333)
        self.assertEqual(msg.addr_recv.ip, ip_address("::ffff:c61b:6409"))
        self.assertEqual(msg.addr_from.ip, ip_address("::ffff:cb00:71c0"))
