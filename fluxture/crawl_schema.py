from abc import abstractmethod
from ipaddress import IPv4Address, IPv6Address as IPv6AddressPython
from typing import Callable, FrozenSet, Generic, Optional, Set, Sized, TypeVar, Union

from .blockchain import Miner, Node, Version
from .db import Cursor, Database, ForeignKey, Model, Table
from .geolocation import Geolocation
from .serialization import DateTime, IntFlag, IPv6Address


N = TypeVar("N", bound=Node)


class HostInfo(Model):
    ip: IPv6Address
    isp: str
    os: str
    timestamp: DateTime

    def __hash__(self):
        return hash(self.ip)


class CrawlState(IntFlag):
    DISCOVERED = 1
    GEOLOCATED = 2
    ATTEMPTED_CONNECTION = DISCOVERED | 4
    CONNECTED = ATTEMPTED_CONNECTION | 8
    REQUESTED_NEIGHBORS = CONNECTED | 16
    GOT_NEIGHBORS = REQUESTED_NEIGHBORS | 32
    REQUESTED_VERSION = CONNECTED | 64
    GOT_VERSION = REQUESTED_VERSION | 128
    UNKNOWN = 0


class CrawledNode(Model["CrawlDatabase"]):
    ip: IPv6Address
    port: int
    is_miner: Miner
    state: CrawlState

    def __hash__(self):
        return hash((self.ip, self.port))

    def get_events(self) -> Cursor["CrawlEvent"]:
        return self.db.events.select(node=self.rowid, order_by="timestamp", order_direction="DESC")

    def get_version(self) -> Optional[Version]:
        for version_event in self.db.events.select(
            node=self.rowid, order_by="timestamp", order_direction="DESC", limit=1, event="version"
        ):
            return Version(version_event.description, version_event.timestamp)
        return None

    def get_location(self) -> Optional[Geolocation]:
        return self.db.locations.select(ip=self.ip, order_by="timestamp DESC", limit=1).fetchone()

    def last_crawled(self) -> Optional[DateTime]:
        max_edge = Cursor(
            self.db.edges,
            "SELECT a.* FROM edges a LEFT OUTER JOIN edges b ON a.rowid = b.rowid AND a.timestamp < b.timestamp "
            "WHERE b.rowid is NULL AND a.from_node = ? LIMIT 1",
            (self.rowid,)
        ).fetchone()
        if max_edge is None:
            return None
        return max_edge.timestamp

    def get_latest_edges(self) -> Set["CrawledNode"]:
        return {
            edge.to_node.row
            for edge in Cursor(
                self.db.edges,
                "SELECT a.* FROM edges a LEFT OUTER JOIN edges b ON a.rowid = b.rowid AND a.timestamp < b.timestamp "
                "WHERE b.rowid is NULL AND a.from_node = ?",
                (self.rowid,)
            )
        }


class Edge(Model):
    from_node: ForeignKey["nodes", CrawledNode]  # noqa: F821
    to_node: ForeignKey["nodes", CrawledNode]  # noqa: F821
    timestamp: DateTime


class CrawlEvent(Model):
    node: ForeignKey["nodes", CrawledNode]  # noqa: F821
    timestamp: DateTime
    event: str
    description: str


class CrawlDatabase(Database):
    nodes: Table[CrawledNode]
    events: Table[CrawlEvent]
    locations: Table[Geolocation]
    edges: Table[Edge]
    hosts: Table[HostInfo]

    def __init__(self, path: str = ":memory:"):
        super().__init__(path)

    @property
    def crawled_nodes(self) -> Cursor[CrawledNode]:
        return Cursor(
            self.nodes,
            f"SELECT DISTINCT n.*, n.rowid FROM {self.nodes.name} n "
            f"LEFT JOIN {self.edges.name} e ON e.from_node = n.rowid "
            f"LEFT JOIN {self.events.name} t ON t.node = n.rowid "
            f"WHERE e.timestamp IS NOT NULL OR t.event = \"version\""
        )


class Crawl(Generic[N], Sized):
    @abstractmethod
    def __contains__(self, node: N) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, node: N) -> CrawledNode:
        raise NotImplementedError()

    @abstractmethod
    def get_node(self, node: N) -> CrawledNode:
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

    @abstractmethod
    def set_miner(self, node: N, miner: Miner):
        raise NotImplementedError()

    @abstractmethod
    def set_host_info(self, host_info: HostInfo):
        raise NotImplementedError()

    @abstractmethod
    def add_state(self, node: N, state: CrawlState):
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

    def __getitem__(self, node: N) -> CrawledNode:
        try:
            return next(iter(self.db.nodes.select(ip=node.address, port=node.port)))
        except StopIteration:
            pass
        raise KeyError(node)

    def get_node(self, node: N) -> CrawledNode:
        try:
            return self[node]
        except KeyError:
            # this is a new node
            pass
        ret = CrawledNode(ip=node.address, port=node.port)
        self.db.nodes.append(ret)
        return ret

    def add_event(self, node: CrawledNode, event: str, description: str, timestamp: Optional[DateTime] = None):
        with self.db:
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
        with self.db:
            crawled_node = self.get_node(node)
            timestamp = DateTime()
            self.db.edges.extend([
                Edge(
                    from_node=crawled_node,
                    to_node=self.get_node(neighbor),
                    timestamp=timestamp
                )
                for neighbor in neighbors
            ])
            self.add_state(node, CrawlState.GOT_NEIGHBORS)

    def set_location(self, ip: IPv6Address, location: Geolocation):
        with self.db:
            self.db.locations.append(location)

    def set_miner(self, node: N, miner: Miner):
        with self.db:
            crawled_node = self.get_node(node)
            crawled_node.is_miner = miner
            self.db.nodes.update(crawled_node)

    def set_host_info(self, host_info: HostInfo):
        with self.db:
            self.db.hosts.append(host_info)

    def add_state(self, node: N, state: CrawlState):
        with self.db:
            crawled_node = self.get_node(node)
            crawled_node.state = crawled_node.state | state
            self.db.nodes.update(crawled_node)

    def __len__(self) -> int:
        return len(self.db.nodes)
