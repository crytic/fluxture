import asyncio
import sys
import traceback
from abc import ABCMeta
from argparse import ArgumentParser, Namespace
from asyncio import ensure_future, Future
from collections import deque
from inspect import isabstract
from typing import Any, Coroutine, Deque, Dict, FrozenSet, Generic, Iterable, List, Optional, Union

from .blockchain import Blockchain, BLOCKCHAINS, Miner, Node
from .crawl_schema import Crawl, CrawlDatabase, DatabaseCrawl, DateTime, N
from .fluxture import Command
from .geolocation import GeoIP2Error, GeoIP2Locator, Geolocator


CRAWL_LISTENERS: List["CrawlListener"] = []


class CrawlListener:
    has_on_crawl_node: bool = False
    has_on_miner: bool = False
    has_on_complete: bool = False

    async def on_crawl_node(self, crawler: "Crawler", node: Node):
        pass

    async def on_miner(self, crawler: "Crawler", node: Node, miner: Miner):
        pass

    async def on_complete(self, crawler: "Crawler"):
        pass

    def __init_subclass__(cls, **kwargs):
        if not isabstract(cls):
            for func in dir(cls):
                if func.startswith("on_") and hasattr(CrawlListener, func):
                    setattr(cls, f"has_{func}", getattr(cls, func) != getattr(CrawlListener, func))
            CRAWL_LISTENERS.append(cls())


class MinerTask(CrawlListener):
    async def on_crawl_node(self, crawler: "Crawler", node: Node):
        is_miner = await crawler.blockchain.is_miner(node)
        crawler.crawl.set_miner(node, is_miner)
        crawler.add_tasks(*(
            listener.on_miner(crawler, node, is_miner) for listener in CRAWL_LISTENERS if listener.has_on_miner
        ))
        if is_miner == Miner.MINER:
            print(f"Node {node} is a miner")
        elif is_miner == Miner.NOT_MINER:
            print(f"Node {node} is not a miner")


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
        self.listener_tasks: List[Future] = []

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
            version = await self.blockchain.get_version(node)
            if version is not None:
                crawled_node = self.crawl.get_node(node)
                self.crawl.add_event(crawled_node, event="version", description=version.version,
                                     timestamp=DateTime(version.timestamp))
            return frozenset(new_neighbors)

    def add_tasks(self, *tasks: Union[Future, Coroutine[Any, Any, None]]):
        for task in tasks:
            if isinstance(task, Coroutine):
                self.listener_tasks.append(ensure_future(task))
            else:
                self.listener_tasks.append(task)

    async def _check_miner(self, node: N):
        is_miner = await self.blockchain.is_miner(node)
        self.crawl.set_miner(node, is_miner)
        return node, is_miner

    async def _crawl(self, seeds: Optional[Iterable[N]] = None):
        if seeds is None:
            seeds = await self.blockchain.default_seeds()
        futures: List[Future] = []
        queue: Deque[N] = deque(seeds)
        while futures or queue or self.listener_tasks:
            print(f"Discovered {len(self.nodes)} nodes; crawled {len(self.crawl)}; "
                  f"crawling {len(futures)}; waiting to crawl {len(queue)}...")
            if futures:
                waiting_on = futures
                done, pending = await asyncio.wait(waiting_on, return_when=asyncio.FIRST_COMPLETED)
                futures = list(pending)
                for result in await asyncio.gather(*done, return_exceptions=True):
                    if isinstance(result, Exception):
                        # TODO: Save the exception to the database
                        # self.crawl.add_event(node, event="Exception", description=str(result))
                        if isinstance(result, (ConnectionError, OSError, BrokenPipeError)):
                            print(str(result))
                        else:
                            traceback.print_tb(result.__traceback__)
                            print(result)
                    else:
                        queue.extend(result)
            if self.listener_tasks:
                waiting_on = self.listener_tasks
                done, pending = await asyncio.wait(waiting_on, return_when=asyncio.FIRST_COMPLETED, timeout=0.5)
                for result in await asyncio.gather(*done, return_exceptions=True):
                    if isinstance(result, Exception):
                        # TODO: Save the exception to the database
                        # self.crawl.add_event(node, event="Exception", description=str(result))
                        traceback.print_tb(result.__traceback__)
                        print(result)
                self.listener_tasks = list(pending)
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
                self.add_tasks(
                    *(
                        listener.on_crawl_node(crawler=self, node=node)
                        for node in nodes_to_crawl
                        for listener in CRAWL_LISTENERS if listener.has_on_crawl_node
                    )
                )

        for miner in await self.blockchain.get_miners():
            if miner not in self.crawl:
                self.crawl.set_miner(miner, Miner.MINER)

        for node in self.nodes.values():
            if node.is_running:
                node.terminate()
                await node.join()

        self.add_tasks(
            *(
                listener.on_complete(crawler=self)
                for listener in CRAWL_LISTENERS if listener.has_on_complete
            )
        )

        # wait for the on_complete tasks to finish:
        while self.listener_tasks:
            waiting_on = self.listener_tasks
            done, pending = await asyncio.wait(waiting_on, return_when=asyncio.FIRST_COMPLETED, timeout=0.5)
            for result in await asyncio.gather(*done, return_exceptions=True):
                if isinstance(result, Exception):
                    # TODO: Save the exception to the database
                    # self.crawl.add_event(node, event="Exception", description=str(result))
                    traceback.print_tb(result.__traceback__)
                    print(result)
            self.listener_tasks = list(pending)

    def do_crawl(self, seeds: Optional[Iterable[N]] = None):
        asyncio.run(self._crawl(seeds))


CITY_DB_PARSER: ArgumentParser = ArgumentParser(add_help=False)

CITY_DB_PARSER.add_argument("--city-db-path", "-c", type=str, default=None,
                            help="path to a MaxMind GeoLite2 City database (default is "
                            "`~/.config/fluxture/geolite2/GeoLite2-City.mmdb`); "
                            "if omitted and `--maxmind-license-key` is provided, the latest database will be "
                            "downloaded and saved to the default location; "
                            "if both options are omttied, then geolocation will not be performed")
CITY_DB_PARSER.add_argument("--maxmind-license-key", type=str, default=None,
                            help="License key for automatically downloading a GeoLite2 City database; you generate get "
                            "a free license key by registering at https://www.maxmind.com/en/geolite2/signup")


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
