from argparse import ArgumentParser, FileType
from fastkml import Placemark, IconStyle, Style
from typing import Dict, Iterable

from .crawler import CrawlDatabase, CrawledNode, IPv6Address
from .fluxture import Command
from .geolocation import Geolocation, KML_NS, to_kml
from .topology import CrawlGraph


class ToKML(Command):
    name = "kml"
    help = "export a KML file visualizing the crawled data"

    def __init_arguments__(self, parser: ArgumentParser):
        parser.add_argument("CRAWL_DB_FILE", type=str,
                            help="path to the crawl database")
        parser.add_argument("KML_FILE", type=FileType("w"),
                            help="path to which to save the KML, or '-' for STDOUT (the default)")
        parser.add_argument("--no-pagerank", action="store_true",
                            help="do not scale the placemarks by their pagerank in the network topology")

    def run(self, args):
        with CrawlDatabase(args.CRAWL_DB_FILE) as db:
            def edges(location: Geolocation) -> Iterable[IPv6Address]:
                possible_nodes_by_port: Dict[int, CrawledNode] = {}
                for possible_node in db.nodes.select(ip=location.ip):
                    port = possible_node.port
                    if port in possible_nodes_by_port:
                        # choose the version that was crawled most recently
                        if possible_nodes_by_port[port].last_crawled() >= possible_node.last_crawled():
                            continue
                    possible_nodes_by_port[port] = possible_node
                neighbors = set()
                for node in possible_nodes_by_port.values():
                    neighbors |= {neighbor.ip for neighbor in node.get_latest_edges()}
                return neighbors
            if args.no_pagerank:
                def to_placemark(loc: Geolocation) -> Placemark:
                    return loc.to_placemark()
            else:
                graph = CrawlGraph.load(db)
                graph.prune()
                pr = graph.pagerank()
                max_rank = max(pr.values())
                if max_rank == 0.0:
                    max_rank = 1.0

                def to_placemark(loc: Geolocation) -> Placemark:
                    rank = sum(pr[node] for node in db.nodes.select(ip=loc.ip) if node in pr)
                    style = Style(KML_NS, styles=[IconStyle(KML_NS, id="ip", scale=1.0 + rank / max_rank * 4.0)])
                    p = loc.to_placemark(style=style)
                    # p.geometry = Geometry(
                    #     geometry=Point(loc.lon, loc.lat, altitude),
                    #     altitude_mode="relativeToGround",
                    #     extrude=True
                    # )
                    return p
            args.KML_FILE.write(to_kml(
                table=db.locations,
                doc_id=f"{args.CRAWL_DB_FILE}_IPs",
                doc_name=f"{args.CRAWL_DB_FILE} IPs",
                doc_description=f"Geolocalized IPs from crawl {args.CRAWL_DB_FILE}",
                edges=edges,
                to_placemark=to_placemark
            ).to_string(prettyprint=True))
