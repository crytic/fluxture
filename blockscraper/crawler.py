import asyncio
from abc import ABCMeta, abstractmethod
from asyncio import ensure_future, Future
from typing import Dict, Generic, Iterable, List, Optional, TypeVar

from .blockchain import Blockchain, FrozenSet, Node

N = TypeVar("N", bound=Node)


class Crawl(Generic[N]):
    @abstractmethod
    def __contains__(self, node: N) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def get_neighbors(self, node: N) -> FrozenSet[N]:
        raise NotImplementedError()

    @abstractmethod
    def set_neighbors(self, node: N, neighbors: FrozenSet[N]):
        raise NotImplementedError()


class InMemoryCrawl(Generic[N], Crawl[N]):
    def __init__(self):
        self.neighbors: Dict[N, FrozenSet[N]] = {}

    def __contains__(self, node: N):
        return node in self.neighbors

    def get_neighbors(self, node: N) -> FrozenSet[N]:
        return self.neighbors[node]

    def set_neighbors(self, node: N, neighbors: FrozenSet[N]):
        print(f"{node} -> {neighbors!r}")
        self.neighbors[node] = neighbors


class Crawler(Generic[N], metaclass=ABCMeta):
    def __init__(self, blockchain: Blockchain[N], crawl: Crawl[N]):
        self.blockchain: Blockchain[N] = blockchain
        self.crawl: Crawl[N] = crawl

    async def _crawl_node(self, node: N, futures: List[Future]):
        neighbors = await self.blockchain.get_neighbors(node)
        self.crawl.set_neighbors(node, neighbors)
        futures.extend([
            ensure_future(self._crawl_node(neighbor, futures))
            for neighbor in neighbors if neighbor not in self.crawl
        ])

    @staticmethod
    async def _crawl(futures: List[Future]):
        for future in futures:
            await future

    def do_crawl(self, seeds: Optional[Iterable[N]] = None):
        if seeds is None:
            seeds = self.blockchain.DEFAULT_SEEDS
        loop = asyncio.new_event_loop()
        futures: List[Future] = []
        futures.extend([
            ensure_future(self._crawl_node(seed_node, futures), loop=loop) for seed_node in seeds
        ])
        loop.run_until_complete(Crawler._crawl(futures))
