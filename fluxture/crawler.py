import asyncio
import errno
import resource
import sys
import traceback
from abc import ABCMeta
from argparse import ArgumentParser, Namespace
from asyncio import ensure_future, Future
from collections import deque
from inspect import isabstract
from typing import Any, AsyncIterator, Coroutine, Deque, Dict, FrozenSet, Generic, Iterable, List, Optional, Union

from geoip2.errors import AddressNotFoundError
from tqdm import tqdm

from .blockchain import Blockchain, BLOCKCHAINS, BlockchainError, Miner, Node
from .crawl_schema import Crawl, CrawlDatabase, CrawlState, DatabaseCrawl, DateTime, N
from .fluxture import Command
from .geolocation import download_maxmind_db, GeoIP2Error, GeoIP2Locator, Geolocator


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
            max_connections: Optional[int] = None
    ):
        self.blockchain: Blockchain[N] = blockchain
        self.crawl: Crawl[N] = crawl
        self.geolocator: Optional[Geolocator] = geolocator
        self.nodes: Dict[N, N] = {}
        if max_connections is None:
            max_connections = resource.getrlimit(resource.RLIMIT_NOFILE)[0] // 3 * 2
        max_connections = max(max_connections, 1)
        self.max_connections: int = max_connections
        self.listener_tasks: List[Future] = []

    async def _crawl_node(self, node: N) -> FrozenSet[N]:
        crawled_node = self.crawl.get_node(node)
        if self.geolocator is not None and crawled_node.state & CrawlState.GEOLOCATED != CrawlState.GEOLOCATED:
            try:
                self.crawl.set_location(node.address, self.geolocator.locate(node.address))
                self.crawl.add_state(crawled_node, CrawlState.GEOLOCATED)
            except AddressNotFoundError:
                pass
        if crawled_node.state & CrawlState.ATTEMPTED_CONNECTION == CrawlState.ATTEMPTED_CONNECTION:
            raise ValueError(f"Node {node} was already crawled!")
        self.crawl.add_state(crawled_node, CrawlState.ATTEMPTED_CONNECTION)
        try:
            async with node:
                self.crawl.add_state(crawled_node, CrawlState.CONNECTED)
                neighbors = []
                new_neighbors = set()
                self.crawl.add_state(crawled_node, CrawlState.REQUESTED_NEIGHBORS)
                for neighbor in await self.blockchain.get_neighbors(node):
                    if neighbor in self.nodes:
                        # we have already seen this node
                        neighbors.append(self.nodes[neighbor])
                    else:
                        self.nodes[neighbor] = neighbor
                        neighbors.append(neighbor)
                        new_neighbors.add(neighbor)
                self.crawl.set_neighbors(node, frozenset(neighbors))
                self.crawl.add_state(crawled_node, CrawlState.GOT_NEIGHBORS | CrawlState.REQUESTED_VERSION)
                version = await self.blockchain.get_version(node)
                if version is not None:
                    self.crawl.add_state(crawled_node, CrawlState.GOT_VERSION)
                    crawled_node = self.crawl.get_node(node)
                    self.crawl.add_event(crawled_node, event="version", description=version.version,
                                         timestamp=DateTime(version.timestamp))
                return frozenset(new_neighbors)
        except BrokenPipeError:
            self.crawl.add_state(crawled_node, CrawlState.CONNECTION_RESET)
            raise
        except OSError as e:
            if e.errno in (errno.ETIMEDOUT, errno.ECONNREFUSED, errno.EHOSTDOWN, errno.EHOSTUNREACH):
                # Connection failed
                self.crawl.add_state(crawled_node, CrawlState.CONNECTION_FAILED)
            else:
                # Something happened after we connected (e.g., connection reset by peer)
                self.crawl.add_state(crawled_node, CrawlState.CONNECTION_RESET)
            raise
        finally:
            await node.close()

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
            seed_iter: Optional[AsyncIterator[N]] = await self.blockchain.default_seeds()
            queue: Deque[N] = deque()
            futures: List[Future] = [
                ensure_future(seed_iter.__anext__())
            ]
            num_seeds = 0
        else:
            seed_iter = None
            queue = deque(seeds)
            futures: List[Future] = []
            num_seeds = len(seeds)
        while futures or queue or self.listener_tasks:
            print(f"Discovered {len(self.nodes)} nodes ({num_seeds} seeds); crawled {len(self.crawl)}; "
                  f"crawling {len(futures)}; waiting to crawl {len(queue)}...")
            if futures:
                waiting_on = futures
                done, pending = await asyncio.wait(waiting_on, return_when=asyncio.FIRST_COMPLETED)
                futures = list(pending)
                for result in await asyncio.gather(*done, return_exceptions=True):
                    # iterate over all of the new neighbors of the node
                    if isinstance(result, StopAsyncIteration) and seed_iter is not None:
                        seed_iter = None
                    elif isinstance(result, Exception):
                        # TODO: Save the exception to the database
                        # self.crawl.add_event(node, event="Exception", description=str(result))
                        if isinstance(result, (ConnectionError, OSError, BrokenPipeError, BlockchainError)):
                            print(str(result))
                        else:
                            traceback.print_tb(result.__traceback__)
                            print(result)
                    elif seed_iter is not None and isinstance(result, Node):
                        # This is a seed
                        crawled_node = self.crawl.get_node(result)
                        if crawled_node.source != result.source:
                            # this means we already organically encountered this node from another peer
                            # so update its source to be the seed
                            crawled_node.source = result.source
                            self.crawl.update_node(crawled_node)
                        self.crawl.add_state(crawled_node, CrawlState.DISCOVERED)
                        # Check if we have already encountered this node
                        queue.append(result)
                        num_seeds += 1
                        futures.append(ensure_future(seed_iter.__anext__()))
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

    def crawl_node(self, node: N) -> FrozenSet[N]:
        """Return the neighbors for a single node"""
        return asyncio.run(self._crawl_node(node))


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


class UpdateMaxmindDBCommand(Command):
    name = "update-geo-db"
    help = "download the latest MaxMind GeoLite2 database"
    parent_parsers = CITY_DB_PARSER,

    def run(self, args: Namespace):
        if args.maxmind_license_key is None:
            sys.stderr.write("Error: --maxmind-license-key must be provided\n\n")
            sys.exit(1)
        save_path = download_maxmind_db(args.maxmind_license_key, args.city_db_path)
        print(f"Geolocation database saved to {save_path}")


class NodeCommand(Command):
    name = "node"
    help = "connect to and interrogate a specific node"

    def __init_arguments__(self, parser: ArgumentParser):
        parser.add_argument("BLOCKCHAIN_NAME", type=str, help="the name of the blockchain to crawl",
                            choices=BLOCKCHAINS.keys())
        parser.add_argument("IP_ADDRESS", type=str, help="IP address of the node to interrogate")

    def run(self, args: Namespace):
        blockchain_type = BLOCKCHAINS[args.BLOCKCHAIN_NAME]
        with CrawlDatabase() as db:
            for neighbor in sorted(str(n.address) for n in Crawler(
                    blockchain=blockchain_type(),
                    crawl=DatabaseCrawl(blockchain_type.node_type, db),
            ).crawl_node(blockchain_type.node_type(args.IP_ADDRESS))):
                print(neighbor)


class GeolocateCommand(Command):
    name = "geolocate"
    help = "re-run geolocation for already crawled nodes (e.g., after a call to the `update-geo-db` command)"
    parent_parsers = CITY_DB_PARSER,

    def __init_arguments__(self, parser: ArgumentParser):
        parser.add_argument("CRAWL_DATABASE", type=str, help="path to the crawl database to update")
        parser.add_argument("--process-all", "-a", action="store_true", help="by default, this command only geolocates "
                                                                             "nodes that do not already have a "
                                                                             "location; this option will re-process "
                                                                             "all nodes")

    def run(self, args: Namespace):
        geo = GeoIP2Locator(args.city_db_path, args.maxmind_license_key)

        with CrawlDatabase(args.CRAWL_DATABASE) as db:
            added = 0
            updated = 0
            with tqdm(db.nodes, leave=False, desc="geolocating", unit=" nodes") as t:
                for node in t:
                    old_location = node.get_location()
                    was_none = old_location is None
                    if not args.process_all and not was_none:
                        continue
                    try:
                        new_location = geo.locate(node.ip)
                    except AddressNotFoundError:
                        continue
                    if new_location is not None:
                        if was_none:
                            db.locations.append(new_location)
                            added += 1
                        elif any(
                                    a != b for (field_name_a, a), (field_name_b, b) in
                                    zip(new_location.items(), old_location.items())
                                    if (
                                            field_name_a != "rowid" and field_name_b != "rowid" and
                                            field_name_a != "timestamp" and field_name_b != "timestamp"
                                    )
                        ):
                            # the location was updated
                            new_location.rowid = old_location.rowid
                            new_location.db = db
                            db.locations.update(new_location)
                            updated += 1
                        else:
                            continue
                        t.desc = f"geolocating ({added} added, {updated} updated)"
            print(f"Added {added} new locations and updated {updated} existing ones")


class CrawlCommand(Command):
    name = "crawl"
    help = "crawl a blockchain"
    parent_parsers = CITY_DB_PARSER,

    def __init_arguments__(self, parser: ArgumentParser):
        parser.add_argument("--database", "-db", type=str, default=":memory:",
                            help="path to the crawl database (default is to run in memory)")
        max_file_descriptors, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        parser.add_argument("--max-connections", "-m", type=int, default=None,
                            help="the maximum number of connections to open at once during the crawl, capped at "
                                 f"⅔ of `ulimit -n` = {max(max_file_descriptors // 3 * 2, 1)} (default is to use the "
                                 "maximum possible)")
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

        if args.max_connections is None:
            max_file_handles, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
            if sys.stderr.isatty() and sys.stdin.isatty():
                if max_file_handles < 1024:
                    while True:
                        sys.stderr.write(f"`uname -n` is {max_file_handles}, which is low and will cause the crawl to "
                                         "be very slow.\nWould you like to increase this value to 32768? [Yn] ")
                        choice = input("")
                        if choice.lower() == "y" or len(choice.strip()) == 0:
                            resource.setrlimit(resource.RLIMIT_NOFILE, (32768, resource.RLIM_INFINITY))
                            max_file_handles, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
                        elif choice.lower() == "n":
                            break
            max_connections = max(max_file_handles // 3 * 2, 1)
        else:
            max_connections = args.max_connections

        def crawl():
            with CrawlDatabase(args.database) as db:
                Crawler(
                    blockchain=blockchain_type(),
                    crawl=DatabaseCrawl(blockchain_type.node_type, db),
                    geolocator=geo,
                    max_connections=max_connections
                ).do_crawl()

        if geo is None:
            crawl()
        else:
            with geo:
                crawl()
