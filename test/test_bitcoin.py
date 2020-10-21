import ipaddress
import time
from unittest import TestCase

from blockscraper.bitcoin import NetAddr, VersionMessage


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
        print(msg.serialize())
