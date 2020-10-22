import asyncio
from abc import ABCMeta, abstractmethod
from asyncio import ensure_future, Future
from typing import Dict, FrozenSet, Generic, Iterable, List, Optional, Tuple, TypeVar

from .blockchain import Blockchain, Node

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
        self.nodes: Dict[N, N] = {}

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
        futures: List[Future] = [
            ensure_future(self._crawl_node(seed_node)) for seed_node in seeds
        ]
        while futures:
            print(f"Discovered {len(self.nodes)} nodes; waiting to crawl {len(futures)}...")
            waiting_on = futures
            done, pending = await asyncio.wait(waiting_on, return_when=asyncio.FIRST_COMPLETED)
            futures = list(pending)
            for result in await asyncio.gather(*done):
                if isinstance(result, Exception):
                    print(result.__traceback__)
                    print(result)
                else:
                    futures.extend(ensure_future(self._crawl_node(node)) for node in result)
        for node in self.nodes.values():
            if node.is_running:
                node.terminate()
                await node.join()

    def do_crawl(self, seeds: Optional[Iterable[N]] = None):
        asyncio.run(self._crawl(seeds))
