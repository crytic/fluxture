from abc import abstractmethod
from ipaddress import IPv4Address, IPv6Address as IPv6AddressPython
from typing import Callable, FrozenSet, Generic, Optional, Set, Sized, TypeVar, Union

from .blockchain import Node
from .db import Cursor, Database, ForeignKey, Model, Table
from .geolocation import Geolocation
from .serialization import DateTime, IPv6Address


N = TypeVar("N", bound=Node)


class CrawledNode(Model["CrawlDatabase"]):
    ip: IPv6Address
    port: int

    def __hash__(self):
        return hash((self.ip, self.port))

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
            (self.rowid, self.rowid)
        ).fetchone()
        if max_edge is None:
            return None
        return max_edge.timestamp

    def get_latest_edges(self) -> Set["CrawledNode"]:
        return {
            edge.to_node.row
            for edge in Cursor(
                    self.db.edges,
                    "SELECT * FROM edges WHERE from_node=? AND "
                    "timestamp=(SELECT max(timestamp) FROM edges WHERE from_node=?)",
                    (self.rowid, self.rowid)
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
    def add_event(self, node: CrawledNode, event: str, description: str, timestamp: Optional[DateTime] = None):
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

    def add_event(self, node: CrawledNode, event: str, description: str, timestamp: Optional[DateTime] = None):
        if timestamp is None:
            timestamp = DateTime()
        self.db.events.append(CrawlEvent(
            node=node.rowid,
            event=event,
            description=description,
            timestamp=timestamp
        ))

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
