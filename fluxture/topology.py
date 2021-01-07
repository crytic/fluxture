import sys
from argparse import ArgumentParser

import networkx as nx
from tqdm import tqdm

from .crawler import CrawlDatabase
from .fluxture import Command

class CrawlGraph(nx.DiGraph):
    @staticmethod
    def load(db: CrawlDatabase) -> "CrawlGraph":
        graph = CrawlGraph()
        for node in tqdm(db.nodes, leave=False, desc="Constructing Topology", unit=" Nodes"):
            for to_node in node.get_latest_edges():
                graph.add_edge(node, to_node)
        return graph

    def prune(self):
        # remove all nodes that are in strongly connected components by themselves
        to_remove = set()
        for connected_component in nx.strongly_connected_components(self):
            if len(connected_component) <= 2:
                to_remove |= connected_component
        self.remove_nodes_from(to_remove)

    def pagerank(self):
        return nx.pagerank(self)


class Topology(Command):
    name = "topology"
    help = "analyze the topology of a network"

    def __init_arguments__(self, parser: ArgumentParser):
        parser.add_argument("CRAWL_DB_FILE", type=str,
                            help="path to the crawl database")

    def run(self, args):
        graph = CrawlGraph.load(CrawlDatabase(args.CRAWL_DB_FILE))
        if len(graph) == 0:
            sys.stderr.write("Error: The crawl contains no nodes!\n")
            return 1
        graph.prune()
        if len(graph) == 0:
            sys.stderr.write("Error: The crawl is insufficient; all of the nodes are in their own connected "
                             "components\n")
            return 1
        print(graph.pagerank())
        return 0
