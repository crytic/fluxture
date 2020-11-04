import asyncio
import sys
import traceback
from abc import ABCMeta, abstractmethod
from argparse import ArgumentParser, Namespace
from asyncio import ensure_future, Future
from collections import deque
from ipaddress import IPv4Address, IPv6Address as IPv6AddressPython
from typing import (
    Callable, Deque, Dict, FrozenSet, Generic, Iterable, List, Optional, Set, Sized, TypeVar, Union
)

from .blockchain import Blockchain, BLOCKCHAINS, Node
from .db import Cursor, Database, ForeignKey, Model, Table
from .fluxture import Command
from .geolocation import CITY_DB_PARSER, GeoIP2Error, GeoIP2Locator, Geolocation, Geolocator
from .serialization import DateTime, IPv6Address

N = TypeVar("N", bound=Node)


class CrawledNode(Model["CrawlDatabase"]):
    ip: IPv6Address
    port: int

    def get_events(self) -> Cursor["CrawlEvent"]:
        return self.db.events.select(node=self.rowid, order_by="timestamp DESC")

    def get_location(self) -> Optional[Geolocation]:
        try:
            return next(iter(self.db[Geolocation].select(ip=self.ip, order_by="timestamp DESC")))
        except StopIteration:
            return None

    def last_crawled(self) -> Optional[DateTime]:
        max_edge = Cursor(
            self.db.edges,
            "SELECT * FROM edges WHERE from_node=? AND "
            "timestamp=(SELECT max(timestamp) FROM edges WHERE from_node=?) LIMIT=1",
            (self.ip, self.ip)
        ).fetchone()
        if max_edge is None:
            return None
        return max_edge.timestamp

    def get_latest_edges(self) -> Set["CrawledNode"]:
        return {
            edge.to_node
            for edge in Cursor(
                    self.db.edges,
                    "SELECT * FROM edges WHERE from_node=? AND "
                    "timestamp=(SELECT max(timestamp) FROM edges WHERE from_node=?)",
                    (self.ip, self.ip)
            )
        }


class Edge(Model):
    from_node: ForeignKey["nodes", CrawledNode]
    to_node: ForeignKey["nodes", CrawledNode]
    timestamp: DateTime


class CrawlEvent(Model):
    node: ForeignKey["nodes", CrawledNode]
    timestamp: DateTime
    event: str
    description: str


class CrawlDatabase(Database):
    nodes: Table[CrawledNode]
    events: Table[CrawlEvent]
    locations: Table[Geolocation]
    edges: Table[Edge]

    def __init__(self, path: str = ":memory:"):
        super().__init__(path)


class Crawl(Generic[N], Sized):
    @abstractmethod
    def __contains__(self, node: N) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def set_location(self, ip: IPv6Address, location: Geolocation):
        raise NotImplementedError()

    @abstractmethod
    def get_neighbors(self, node: N) -> FrozenSet[N]:
        raise NotImplementedError()

    @abstractmethod
    def set_neighbors(self, node: N, neighbors: FrozenSet[N]):
        raise NotImplementedError()


class DatabaseCrawl(Generic[N], Crawl[N]):
    def __init__(
            self,
            constructor: Callable[[Union[str, IPv4Address, IPv6AddressPython], int], N],
            db: CrawlDatabase
    ):
        super().__init__()
        self.constructor: Callable[[Union[str, IPv4Address, IPv6AddressPython], int], N] = constructor
        self.db: CrawlDatabase = db

    def __contains__(self, node: N) -> bool:
        return self.db.nodes.select(ip=node.ip, port=node.port).fetchone() is not None

    def get_node(self, node: N) -> CrawledNode:
        try:
            return next(iter(self.db.nodes.select(ip=node.address, port=node.port)))
        except StopIteration:
            # this is a new node
            pass
        ret = CrawledNode(ip=node.address, port=node.port)
        self.db.nodes.append(ret)
        return ret

    def get_neighbors(self, node: N) -> FrozenSet[N]:
        return frozenset({
            self.constructor(neighbor.ip, neighbor.port) for neighbor in self.get_node(node).get_latest_edges()
        })

    def set_neighbors(self, node: N, neighbors: FrozenSet[N]):
        crawled_node = self.get_node(node)
        timestamp = DateTime()
        with self.db:
            self.db.edges.extend([
                Edge(
                    from_node=crawled_node,
                    to_node=self.get_node(neighbor),
                    timestamp=timestamp
                )
                for neighbor in neighbors
            ])

    def set_location(self, ip: IPv6Address, location: Geolocation):
        self.db.locations.append(location)

    def __len__(self) -> int:
        return len(self.db.nodes)


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
