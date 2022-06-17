from abc import ABC, abstractmethod
from argparse import ArgumentParser, FileType
from collections import defaultdict
from math import pi, sin
from typing import Dict, Iterable, List, Optional, OrderedDict, Set

from fastkml import IconStyle, LineStyle, Placemark, Style, kml
from fastkml.geometry import Geometry, LineString, Point
from shapely.geometry import MultiPoint
from tqdm import tqdm

from .blockchain import Miner
from .crawl_schema import CrawledNode
from .crawler import CrawlDatabase
from .fluxture import Command
from .geolocation import Geolocation, Location
from .topology import CrawlGraph, ProbabilisticWeightedCrawlGraph

KML_NS = "{http://www.opengis.net/kml/2.2}"
EARTH_CIRCUMFERENCE = 40075000.0
EARTH_DIAMETER = 12742000.0


class KMLGraphNode(ABC, Location):
    @abstractmethod
    def neighbors(self) -> Iterable["KMLGraphNode"]:
        raise NotImplementedError()

    @abstractmethod
    def to_placemark(self, style: Optional[Style] = None) -> Placemark:
        raise NotImplementedError()

    @abstractmethod
    def description(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def uid(self) -> str:
        raise NotImplementedError()

    def __eq__(self, other):
        return isinstance(other, KMLGraphNode) and self.uid() == other.uid()

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash(self.uid())

    def __str__(self):
        return self.uid()


class KMLGeolocation(KMLGraphNode):
    def __init__(
        self, location: Geolocation, db: CrawlDatabase, is_miner: bool = False
    ):
        self.location: Geolocation = location
        self.db: CrawlDatabase = db
        self.lat = location.lat
        self.lon = location.lon
        self.is_miner: bool = is_miner

    def __eq__(self, other):
        return (
            isinstance(other, KMLGeolocation) and other.location.ip == self.location.ip
        ) or super().__eq__(other)

    def uid(self) -> str:
        return str(self.location.ip)

    def neighbors(self) -> Iterable["KMLGeolocation"]:
        possible_nodes_by_port: Dict[int, CrawledNode] = {}
        for possible_node in self.db.nodes.select(ip=self.location.ip):
            port = possible_node.port
            if port in possible_nodes_by_port:
                # choose the version that was crawled most recently
                if (
                    possible_nodes_by_port[port].last_crawled()
                    >= possible_node.last_crawled()
                ):
                    continue
            possible_nodes_by_port[port] = possible_node
        locations: Dict[Geolocation, Set[CrawledNode]] = defaultdict(set)
        for node in possible_nodes_by_port.values():
            for neighbor in node.get_latest_edges():
                neighbor_location = self.db.locations.select(
                    ip=neighbor.ip,
                    limit=1,
                    order_by="timestamp",
                    order_direction="DESC",
                ).fetchone()
                if neighbor_location is None:
                    continue
                locations[neighbor_location].add(neighbor)
        return (
            KMLGeolocation(
                loc, self.db, is_miner=any(n.is_miner == Miner.MINER for n in nodes)
            )
            for loc, nodes in locations.items()
            if loc is not None
        )

    @property
    def ip_str(self) -> str:
        if self.location.ip.ipv4_mapped is not None:
            return str(self.location.ip.ipv4_mapped)
        else:
            return str(self.location.ip)

    def description(self) -> str:
        if self.is_miner:
            miner_str = f"Likely a Miner "
        else:
            miner_str = ""
        return (
            f"{miner_str}{self.ip_str}: {self.location.city} ({self.lat}°N, {self.lon}°E) @ "
            f"{self.location.timestamp!s}"
        )

    def to_placemark(self, style: Optional[Style] = None) -> kml.Placemark:
        if style is None:
            style = Style(KML_NS, styles=[IconStyle(KML_NS, id="ip")])
        p = kml.Placemark(KML_NS, self.uid(), self.ip_str, self.description())
        p.append_style(style)
        p.geometry = Point(self.lon, self.lat)
        return p


class ScaledKMLGraphNode(KMLGraphNode):
    def __init__(self, wrapped: KMLGraphNode, scale: float):
        self.node: KMLGraphNode = wrapped
        self.scale: float = scale
        self.lat = wrapped.lat
        self.lon = wrapped.lon

    def neighbors(self) -> Iterable["KMLGraphNode"]:
        return self.node.neighbors()

    def to_placemark(self, style: Optional[Style] = None) -> Placemark:
        scaled_style = Style(
            KML_NS, styles=[IconStyle(KML_NS, id="ip", scale=self.scale)]
        )
        if style is not None:
            scaled_style.append_style(style)
        p = self.node.to_placemark(style=scaled_style)
        return p

    def description(self) -> str:
        return self.node.description()

    def uid(self) -> str:
        return self.node.uid()


class KMLGraphNodeCollection(KMLGraphNode):
    def __init__(
        self,
        name: str,
        subnodes: Iterable[KMLGraphNode] = (),
        neighbors: Iterable[KMLGraphNode] = (),
    ):
        self.name: str = name
        self._neighbors: List[KMLGraphNode] = list(neighbors)
        self._subnodes: List[KMLGraphNode] = []
        self.subnodes = subnodes

    def set_neighbors(self, neighbors: Iterable[KMLGraphNode]):
        self._neighbors = list(neighbors)
        assert self not in self._neighbors

    def uid(self) -> str:
        return self.name

    @property
    def subnodes(self) -> List[KMLGraphNode]:
        return self._subnodes

    @subnodes.setter
    def subnodes(self, nodes: Iterable[KMLGraphNode]):
        self._subnodes = list(nodes)
        if self._subnodes:
            points = MultiPoint([(node.lon, node.lat) for node in self.subnodes])
            centroid = points.convex_hull.centroid
            self.lon = centroid.x
            self.lat = centroid.y

    def neighbors(self) -> Iterable["KMLGraphNode"]:
        return self._neighbors

    def description(self) -> str:
        return "\n".join(n.description() for n in self.subnodes)

    def to_placemark(self, style: Optional[Style] = None) -> kml.Placemark:
        if style is None:
            style = Style(KML_NS, styles=[IconStyle(KML_NS, id="ip")])
        p = kml.Placemark(KML_NS, self.uid(), self.name, self.description())
        p.append_style(style)
        p.geometry = Point(self.lon, self.lat)
        return p


def to_kml(
    locations: Iterable[KMLGraphNode],
    doc_id: str,
    doc_name: str,
    doc_description: str,
    max_altitude: float = EARTH_DIAMETER / 4.0,
) -> kml.KML:
    k = kml.KML()
    d = kml.Document(KML_NS, doc_id, doc_name, doc_description)
    k.append(d)
    f = kml.Folder(KML_NS, "ips", "IPs", "Geolocalized IPs")
    d.append(f)
    edge_folder = kml.Folder(
        KML_NS, "topology", "Topology", "The network topology discovered in the crawl"
    )
    d.append(edge_folder)
    edge_color = (0, 255, 0)
    edge_hex_color = "7f%02x%02x%02x" % tuple(reversed(edge_color))
    edge_style = Style(
        KML_NS, styles=[LineStyle(KML_NS, id="edge", color=edge_hex_color, width=3)]
    )
    for geolocation in tqdm(
        locations, leave=False, desc="Generating KML", unit=" locations"
    ):
        f.append(geolocation.to_placemark())
        for neighbor in geolocation.neighbors():
            if neighbor is None or geolocation == neighbor:
                continue
            p = kml.Placemark(
                KML_NS,
                f"{geolocation!s}->{neighbor!s}",
                f"{geolocation!s}->{neighbor!s}",
                f"Edge between {geolocation!s} and {neighbor!s}",
            )
            p.append_style(edge_style)
            num_segments = 20
            distance = geolocation.distance_to(neighbor)
            peak_altitude = max_altitude * distance / (EARTH_CIRCUMFERENCE / 2.0)
            p.geometry = Geometry(
                geometry=LineString(
                    [
                        (lon, lat, sin(i / (num_segments - 1) * pi) * peak_altitude)
                        for i, (lon, lat) in enumerate(
                            geolocation.path_to(
                                neighbor, intermediate_points=num_segments - 2
                            )
                        )
                    ]
                ),
                tessellate=False,
                extrude=False,
                altitude_mode="relativeToGround",
            )
            edge_folder.append(p)
    return k


def calculate_rank(
    loc: KMLGraphNode, pr: OrderedDict[CrawledNode, float], db: CrawlDatabase
) -> float:
    if isinstance(loc, KMLGeolocation):
        nodes = db.nodes.select(ip=loc.location.ip)
    elif isinstance(loc, KMLGraphNodeCollection):
        return sum(calculate_rank(subnode, pr, db) for subnode in loc.subnodes)
    else:
        raise NotImplementedError(f"Add support for locations of type {type(loc)}")
    return sum(pr[node] for node in nodes if node in pr)


class ToKML(Command):
    name = "kml"
    help = "export a KML file visualizing the crawled data"

    def __init_arguments__(self, parser: ArgumentParser):
        parser.add_argument(
            "CRAWL_DB_FILE", type=str, help="path to the crawl database"
        )
        parser.add_argument(
            "KML_FILE",
            type=FileType("w"),
            help="path to which to save the KML, or '-' for STDOUT (the default)",
        )
        parser.add_argument(
            "--no-pagerank",
            action="store_true",
            help="do not scale the placemarks by their pagerank in the network topology",
        )
        parser.add_argument(
            "--group-by",
            "-g",
            default="ip",
            choices=["ip", "city", "country", "continent"],
            help="grouping of pins (default: %(default)s)",
        )

    def run(self, args):
        with CrawlDatabase(args.CRAWL_DB_FILE) as db:
            locations = (KMLGeolocation(loc, db) for loc in db.locations)
            if args.group_by != "ip":
                if args.group_by == "city":

                    def grouper(loc: KMLGeolocation) -> str:
                        return loc.location.city

                elif args.group_by == "country":

                    def grouper(loc: KMLGeolocation) -> str:
                        return loc.location.country_code

                elif args.group_by == "continent":

                    def grouper(loc: KMLGeolocation) -> str:
                        return loc.location.continent_code

                else:
                    raise NotImplementedError(
                        f"TODO: Implement support for --group-by={args.group_by}"
                    )
                groups: Dict[str, List[KMLGeolocation]] = defaultdict(list)
                for loc in locations:
                    groups[grouper(loc)].append(loc)
                collections: Dict[str, KMLGraphNodeCollection] = {
                    group: KMLGraphNodeCollection(group, subnodes=subnodes)
                    for group, subnodes in groups.items()
                }
                for group, c in collections.items():
                    all_neighbors = set()
                    for member in groups[group]:
                        all_neighbors |= {
                            grouper(neighbor) for neighbor in member.neighbors()
                        }
                    all_neighbors -= {group}
                    c.set_neighbors(
                        (
                            collections[neighbor_group]
                            for neighbor_group in all_neighbors
                        )
                    )
                locations = collections.values()
            if not args.no_pagerank:
                graph = CrawlGraph.load(db)
                graph.prune()
                pr = ProbabilisticWeightedCrawlGraph(graph).pagerank()
                max_rank = max(pr.values())
                if max_rank == 0.0:
                    max_rank = 1.0
                new_locations = []
                for loc in locations:
                    scale = 1.0 + calculate_rank(loc, pr, db) / max_rank * 4.0
                    new_locations.append(ScaledKMLGraphNode(loc, scale))
                locations = new_locations
            args.KML_FILE.write(
                to_kml(
                    locations=locations,
                    doc_id=f"{args.CRAWL_DB_FILE}_IPs",
                    doc_name=f"{args.CRAWL_DB_FILE} IPs",
                    doc_description=f"Geolocalized IPs from crawl {args.CRAWL_DB_FILE}",
                ).to_string(prettyprint=True)
            )
