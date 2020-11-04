import asyncio
import sys
import traceback
from abc import ABCMeta
from argparse import ArgumentParser, Namespace
from asyncio import ensure_future, Future
from collections import deque
from typing import Deque, Dict, FrozenSet, Generic, Iterable, List, Optional

from .blockchain import Blockchain, BLOCKCHAINS
from .crawl_schema import Crawl, CrawlDatabase, DatabaseCrawl, N
from .fluxture import Command
from .geolocation import CITY_DB_PARSER, GeoIP2Error, GeoIP2Locator, Geolocator


class Crawler(Generic[N], metaclass=ABCMeta):
    def __init__(
            self,
            blockchain: Blockchain[N],
            crawl: Crawl[N],
            geolocator: Optional[Geolocator] = None,
            max_connections: int = 1024
    ):
        self.blockchain: Blockchain[N] = blockchain
        self.crawl: Crawl[N] = crawl
        self.geolocator: Optional[Geolocator] = geolocator
        self.nodes: Dict[N, N] = {}
        self.max_connections: int = max_connections

    async def _crawl_node(self, node: N) -> FrozenSet[N]:
        if self.geolocator is not None:
            self.crawl.set_location(node.address, self.geolocator.locate(node.address))
        async with node:
            neighbors = []
            new_neighbors = set()
            for neighbor in await self.blockchain.get_neighbors(node):
                if neighbor in self.nodes:
                    neighbors.append(self.nodes[neighbor])
                else:
                    self.nodes[neighbor] = neighbor
                    neighbors.append(neighbor)
                    new_neighbors.add(neighbor)
            self.crawl.set_neighbors(node, frozenset(neighbors))
            return frozenset(new_neighbors)

    async def _crawl(self, seeds: Optional[Iterable[N]] = None):
        if seeds is None:
            seeds = self.blockchain.DEFAULT_SEEDS
        futures: List[Future] = []
        queue: Deque[N] = deque(seeds)
        while futures or queue:
            print(f"Discovered {len(self.nodes)} nodes; crawled {len(self.crawl)}; "
                  f"crawling {len(futures)}; waiting to crawl {len(queue)}...")
            if futures:
                waiting_on = futures
                done, pending = await asyncio.wait(waiting_on, return_when=asyncio.FIRST_COMPLETED)
                futures = list(pending)
                for result in await asyncio.gather(*done, return_exceptions=True):
                    if isinstance(result, Exception):
                        traceback.print_tb(result.__traceback__)
                        print(result)
                    else:
                        queue.extend(result)
            new_nodes_to_crawl = min(self.max_connections - len(futures), len(queue))
            if new_nodes_to_crawl:
                nodes_to_crawl = []
                for i in range(new_nodes_to_crawl):
                    node = queue.popleft()
                    if node in self.nodes:
                        nodes_to_crawl.append(self.nodes[node])
                    else:
                        nodes_to_crawl.append(node)
                        self.nodes[node] = node
                futures.extend(ensure_future(self._crawl_node(node)) for node in nodes_to_crawl)

        for node in self.nodes.values():
            if node.is_running:
                node.terminate()
                await node.join()

    def do_crawl(self, seeds: Optional[Iterable[N]] = None):
        asyncio.run(self._crawl(seeds))


class CrawlCommand(Command):
    name = "crawl"
    help = "crawl a blockchain"
    parent_parsers = CITY_DB_PARSER,

    def __init_arguments__(self, parser: ArgumentParser):
        parser.add_argument("--database", "-db", type=str, default=":memory:",
                            help="path to the crawl database (default is to run in memory)")
        parser.add_argument("BLOCKCHAIN_NAME", type=str, help="the name of the blockchain to crawl",
                            choices=BLOCKCHAINS.keys())

    def run(self, args: Namespace):
        try:
            geo = GeoIP2Locator(args.city_db_path, args.maxmind_license_key)
        except GeoIP2Error as e:
            sys.stderr.write(f"Warning: {e}\nCrawl IPs will not be geolocated!\n")
            geo = None

        if args.database == ":memory:":
            sys.stderr.write("Warning: Using an in-memory crawl database. Results will not be saved!\n"
                             "Run with `--database` to set a path for the database to be saved.\n")

        blockchain_type = BLOCKCHAINS[args.BLOCKCHAIN_NAME]

        def crawl():
            Crawler(
                blockchain=blockchain_type(),
                crawl=DatabaseCrawl(blockchain_type.node_type, CrawlDatabase(args.database)),
                geolocator=geo
            ).do_crawl()

        if geo is None:
            crawl()
        else:
            with geo:
                crawl()
