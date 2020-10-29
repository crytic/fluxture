import asyncio
import traceback
from abc import ABCMeta, abstractmethod
from asyncio import ensure_future, Future
from collections import deque
from typing import Deque, Dict, FrozenSet, Generic, Iterable, List, Optional, Sized, TypeVar

from .blockchain import Blockchain, Node
from .db import Database, ForeignKey, Model, primary_key
from .serialization import DateTime, IPv6Address

N = TypeVar("N", bound=Node)


class Geolocation(Model):
    id:
    ip: IPv6Address
    timestamp: DateTime


class CrawledNode(Model):
    ip: primary_key(IPv6Address)
    port: int


class CrawlEvent(Model):
    node: ForeignKey[CrawledNode]
    timestamp: DateTime
    event: str
    description: str


class Crawl(Generic[N], Sized):
    @abstractmethod
    def __contains__(self, node: N) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def get_neighbors(self, node: N) -> FrozenSet[N]:
        raise NotImplementedError()

    @abstractmethod
    def set_neighbors(self, node: N, neighbors: FrozenSet[N]):
        raise NotImplementedError()


class DatabaseCrawl(Generic[N], Crawl[N], Database):
    nodes: CrawledNode
    events: CrawlEvent


class InMemoryCrawl(Generic[N], Crawl[N]):
    def __init__(self):
        self.neighbors: Dict[N, FrozenSet[N]] = {}

    def __contains__(self, node: N):
        return node in self.neighbors

    def __len__(self):
        return len(self.neighbors)

    def get_neighbors(self, node: N) -> FrozenSet[N]:
        return self.neighbors[node]

    def set_neighbors(self, node: N, neighbors: FrozenSet[N]):
        print(f"{node} -> {neighbors!r}")
        self.neighbors[node] = neighbors


class Crawler(Generic[N], metaclass=ABCMeta):
    def __init__(self, blockchain: Blockchain[N], crawl: Crawl[N], max_connections: int = 1024):
        self.blockchain: Blockchain[N] = blockchain
        self.crawl: Crawl[N] = crawl
        self.nodes: Dict[N, N] = {}
        self.max_connections: int = max_connections

    async def _crawl_node(self, node: N) -> FrozenSet[N]:
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
