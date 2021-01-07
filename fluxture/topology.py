import sys
from argparse import ArgumentParser
from collections import OrderedDict
from typing import Callable, Optional

import graphviz
import networkx as nx
from tqdm import tqdm

from .crawl_schema import CrawledNode
from .crawler import CrawlDatabase
from .fluxture import Command


class CrawlGraph(nx.DiGraph):
    @staticmethod
    def load(db: CrawlDatabase) -> "CrawlGraph":
        graph = CrawlGraph()
        for node in tqdm(db.nodes, leave=False, desc="Constructing Topology", unit=" nodes"):
            for to_node in node.get_latest_edges():
                graph.add_edge(node, to_node)
                # assume that all edges are bidirectional
                graph.add_edge(to_node, node)
        return graph

    def to_dot(
        self, comment: Optional[str] = None, labeler: Optional[Callable[[CrawledNode], str]] = None, node_filter=None
    ) -> graphviz.Digraph:
        if comment is not None:
            dot = graphviz.Digraph(comment=comment)
        else:
            dot = graphviz.Digraph()

        def default_labeler(node: CrawledNode):
            return f"[{node.ip!s}]:{node.port}"

        if labeler is None:
            labeler = default_labeler
        node_ids = {node: i for i, node in enumerate(self.nodes)}
        for node in self.nodes:
            if node_filter is None or node_filter(node):
                dot.node(f"node{node_ids[node]}", label=labeler(node))
        for caller, callee in self.edges:
            if node_filter is None or (node_filter(caller) and node_filter(callee)):
                dot.edge(f"node{node_ids[caller]}", f"node{node_ids[callee]}")
        return dot

    def prune(self):
        # remove all nodes that are in strongly connected components by themselves
        to_remove = set()
        for connected_component in tqdm(
                nx.strongly_connected_components(self),
                leave=False,
                desc="Pruning trivial connected components ",
                unit=" components"
        ):
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
        # graph.to_dot().save("graph.dot")
        graph.prune()
        if len(graph) == 0:
            sys.stderr.write("Error: The crawl is insufficient; all of the nodes are in their own connected "
                             "components\n")
            return 1
        pr = OrderedDict(sorted(graph.pagerank().items(), key=lambda item: item[1], reverse=True))
        print(f"SUM: {sum(val for val in pr.values())}")
        for node, rank in pr.items():
            print(f"[{node.ip!s}]:{node.port}\t{rank}")
        return 0
