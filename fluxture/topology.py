import sys
from argparse import ArgumentParser
from collections import defaultdict, OrderedDict
from typing import (
    Callable, Dict, FrozenSet, Generic, Hashable, Optional, OrderedDict as OrderedDictType, Set, TypeVar, Union
)

import graphviz
import networkx as nx
from tqdm import tqdm

from .crawl_schema import CrawledNode
from .crawler import CrawlDatabase
from .fluxture import Command


N = TypeVar("N", bound=Hashable)


class NodeGroup(frozenset, Generic[N], FrozenSet[N]):
    name: str

    def __new__(cls, *args, **kwargs):
        if "name" in kwargs:
            name = kwargs["name"]
            del kwargs["name"]
        else:
            name = None
        retval = frozenset.__new__(cls, *args, **kwargs)
        if name is not None:
            setattr(retval, "name", name)
        return retval


class CrawlGraph(nx.DiGraph, Generic[N]):
    @staticmethod
    def load(db: CrawlDatabase) -> "CrawlGraph[CrawledNode]":
        graph = CrawlGraph()
        for node in tqdm(db.nodes, leave=False, desc="Constructing Topology", unit=" nodes"):
            for to_node in node.get_latest_edges():
                graph.add_edge(node, to_node)
                # assume that all edges are bidirectional
                graph.add_edge(to_node, node)
        return graph

    def to_dot(
            self,
            comment: Optional[str] = None,
            labeler: Optional[Callable[[N], str]] = None,
            node_filter: Optional[Callable[[N], bool]] = None
    ) -> graphviz.Digraph:
        if comment is not None:
            dot = graphviz.Digraph(comment=comment)
        else:
            dot = graphviz.Digraph()

        def default_labeler(node: N):
            if isinstance(node, CrawledNode):
                return f"[{node.ip!s}]:{node.port}"
            else:
                return str(node)

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

    def pagerank(self) -> OrderedDictType[N, float]:
        return OrderedDict(sorted(nx.pagerank(self).items(), key=lambda item: item[1], reverse=True))

    def group_by(self, grouper: Callable[[N], str]) -> "GroupedCrawlGraph[N]":
        groups_by_node: Dict[N, str] = {node: grouper(node) for node in self}
        groups: Dict[str, Set[N]] = defaultdict(set)
        node_groups: Dict[str, NodeGroup[N]] = {}
        graph = GroupedCrawlGraph(self)
        for node, group in groups_by_node.items():
            groups[group].add(node)
        for group, members in groups.items():
            ng = NodeGroup(members, name=group)
            graph.add_node(ng)
            node_groups[group] = ng
        for from_node, to_node in self.edges:
            from_group = node_groups[groups_by_node[from_node]]
            to_group = node_groups[groups_by_node[to_node]]
            if from_group != to_group and not graph.has_edge(from_group, to_group):
                graph.add_edge(from_group, to_group)
        return graph


class GroupedCrawlGraph(CrawlGraph[NodeGroup[N]], Generic[N]):
    def __init__(self, parent: CrawlGraph[N]):
        super().__init__()
        self.parent: CrawlGraph[N] = parent

    def grouped_pagerank(self) -> OrderedDictType[NodeGroup[N], float]:
        parent_ranks = self.parent.pagerank()
        ranks = {
            group: sum(parent_ranks[node] for node in group) for group in self.nodes
        }
        return OrderedDict(sorted(ranks.items(), key=lambda item: item[1], reverse=True))


class Topology(Command):
    name = "topology"
    help = "analyze the topology of a network"

    def __init_arguments__(self, parser: ArgumentParser):
        parser.add_argument("CRAWL_DB_FILE", type=str,
                            help="path to the crawl database")
        parser.add_argument("--group-by",
                            "-g",
                            default="ip",
                            choices=["ip", "city", "country", "continent"],
                            help="grouping of nodes (default: %(default)s)")
        parser.add_argument("--conglomerate", "-c", action="store_true",
                            help="when calculating the PageRank of a group, instead of summing the constituent nodes'"
                                 "ranks (the default), treat each group as its own supernode and use the PageRank of "
                                 "the group nodes in the intersection graph formed by the groups")

    def run(self, args):
        raw_graph = CrawlGraph.load(CrawlDatabase(args.CRAWL_DB_FILE))
        if len(raw_graph) == 0:
            sys.stderr.write("Error: The crawl contains no nodes!\n")
            return 1
        elif args.group_by == "ip":
            graph = raw_graph
            page_rank: OrderedDictType[Union[CrawledNode, NodeGroup[CrawledNode]], float] = graph.pagerank()
        else:
            if args.group_by == "city":
                def grouper(n: CrawledNode) -> str:
                    loc = n.get_location()
                    if loc is None:
                        return "Unknown City"
                    return loc.city
            elif args.group_by == "country":
                def grouper(n: CrawledNode) -> str:
                    loc = n.get_location()
                    if loc is None:
                        return "??"
                    return loc.country_code
            elif args.group_by == "continent":
                def grouper(n: CrawledNode) -> str:
                    loc = n.get_location()
                    if loc is None:
                        return "??"
                    return loc.continent_code
            else:
                raise NotImplementedError(f"TODO: Implement support for --group-by={args.group_by}")
            graph = raw_graph.group_by(grouper)
            if args.conglomerate:
                page_rank = graph.pagerank()
            else:
                page_rank = graph.grouped_pagerank()
        # graph.to_dot().save("graph.dot")
        graph.prune()
        if len(graph) == 0:
            sys.stderr.write("Error: The crawl is insufficient; all of the nodes are in their own connected "
                             "components\n")
            return 1
        for node, rank in page_rank.items():
            if isinstance(node, NodeGroup):
                print(f"{node.name}\t{rank}")
            else:
                print(f"[{node.ip!s}]:{node.port}\t{rank}")
        print(f"Edge Connectivity: {nx.edge_connectivity(graph)}")
        return 0
