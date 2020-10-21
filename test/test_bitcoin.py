import time
from unittest import TestCase

from blockscraper.bitcoin import BitcoinMessage, NetAddr, VersionMessage

EXAMPLE_VERSION_MESSAGE = \
    b"\xf9\xbe\xb4\xd9\x76\x65\x72\x73\x69\x6f\x6e\x00\x00\x00\x00\x00" \
    b"\x64\x00\x00\x00\x35\x8d\x49\x32\x62\xea\x00\x00\x01\x00\x00\x00" \
    b"\x00\x00\x00\x00\x11\xb2\xd0\x50\x00\x00\x00\x00\x01\x00\x00\x00" \
    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff" \
    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00" \
    b"\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff\x00\x00\x00\x00\x00\x00" \
    b"\x3b\x2e\xb3\x5d\x8c\xe6\x17\x65\x0f\x2f\x53\x61\x74\x6f\x73\x68" \
    b"\x69\x3a\x30\x2e\x37\x2e\x32\x2f\xc0\x3e\x03\x00"


class TestBitcoin(TestCase):
    def test_version_message(self):
        t = int(time.time())
        msg = VersionMessage(
            version=70015,
            services=0,
            timestamp=t,
            addr_recv=NetAddr(time=t),
            addr_from=NetAddr(time=t),
            user_agent=b"BlockScraper",
            start_height=123,
            relay=True
        )
        #self.assertEqual(msg, BitcoinMessage.deserialize(msg.serialize()))
        msg = BitcoinMessage.deserialize(EXAMPLE_VERSION_MESSAGE)
        self.assertIsInstance(msg, VersionMessage)
