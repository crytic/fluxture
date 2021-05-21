import sys
from argparse import ArgumentParser
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import (
    Callable, Dict, FrozenSet, Generic, Hashable, Optional, OrderedDict as OrderedDictType, Set, TypeVar, Union
)

import graphviz
import networkx as nx
from tqdm import tqdm

from .crawl_schema import CrawledNode
from .crawler import CrawlDatabase
from .fluxture import Command
from .statistics import Statistics


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
    def load(
            db: CrawlDatabase, only_crawled_nodes: bool = False, bidirectional_edges: bool = True
    ) -> "CrawlGraph[CrawledNode]":
        graph = CrawlGraph()
        for node in tqdm(db.nodes, leave=False, desc="Constructing Topology", unit=" nodes"):
            if only_crawled_nodes and node.last_crawled() is None:
                continue
            graph.add_node(node)
            for to_node in node.get_latest_edges():
                if only_crawled_nodes and to_node.last_crawled() is None:
                    continue
                graph.add_edge(node, to_node)
                if bidirectional_edges:
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


class ExportCommand(Command):
    name = "export"
    help = "export the crawl data"

    def __init_arguments__(self, parser: ArgumentParser):
        parser.add_argument("CRAWL_DB_FILE", type=str, help="path to the crawl database")
        parser.add_argument("--format", "-f", choices=["arff", "csv"], default="arff", help="the format in which to "
                                                                                            "export the data")

    def run(self, args):
        with tqdm(desc="exporting", leave=False, unit=" steps", total=4) as t:
            graph = CrawlGraph.load(CrawlDatabase(args.CRAWL_DB_FILE), only_crawled_nodes=False,
                                    bidirectional_edges=False)
            t.update(1)
            clean_graph = CrawlGraph.load(CrawlDatabase(args.CRAWL_DB_FILE), only_crawled_nodes=True,
                                          bidirectional_edges=True)
            t.update(1)
            with tqdm(desc="calculating stationary distribution", leave=False, unit=" steps", total=2) as s:
                page_rank = graph.pagerank()
                s.update(1)
                clean_page_rank = clean_graph.pagerank()
                s.update(1)
            del clean_graph
            t.update(1)

            cities: Set[str] = {"?"}
            countries: Set[str] = {"?"}
            continents: Set[str] = {"?"}
            versions: Set[str] = {"?"}
            for node in graph:
                loc = node.get_location()
                if loc is not None:
                    if loc.city is not None:
                        cities.add(loc.city)
                    if loc.country_code is not None:
                        countries.add(loc.country_code)
                    if loc.continent_code is not None:
                        continents.add(loc.continent_code)
                version = node.get_version()
                if version is not None:
                    versions.add(version)

            if args.format == "arff":
                print(f"""% Fluxture crawl
    % Source: {args.CRAWL_DB_FILE}
    
    @RELATION topology
    
    @ATTRIBUTE ip               STRING
    @ATTRIBUTE continent        {{{','.join(map(repr, continents))}}}
    @ATTRIBUTE country          {{{','.join(map(repr, countries))}}}
    @ATTRIBUTE city             {{{','.join(map(repr, cities))}}}
    @ATTRIBUTE crawled          {{TRUE, FALSE}}
    @ATTRIBUTE version          {{{','.join(map(repr, versions))}}}
    @ATTRIBUTE out_degree       NUMERIC
    @ATTRIBUTE in_degree        NUMERIC
    @ATTRIBUTE mutual_neighbors NUMERIC
    @ATTRIBUTE raw_centrality   NUMERIC
    @ATTRIBUTE norm_centrality  NUMERIC
    
    @DATA
    """)
            else:
                # Assume CSV format
                print("ip,continent,country,city,crawled,version,out_degree,in_degree,mutual_neighbors,raw_centrality,"
                      "norm_centrality")
            for node in tqdm(graph, desc="writing", unit=" nodes", leave=False):
                loc = node.get_location()
                if loc is None:
                    city = "?"
                    country = "?"
                    continent = "?"
                else:
                    if loc.city is None or loc.city == "None":
                        city = "?"
                    else:
                        city = loc.city
                    if loc.country_code is None:
                        country = "?"
                    else:
                        country = loc.country_code
                    if loc.continent_code is None:
                        continent = "?"
                    else:
                        continent = loc.continent_code
                    if args.format == "arff":
                        city = repr(city)
                        country = repr(country)
                        continent = repr(continent)
                version = node.get_version()
                if version is None:
                    version_str = "?"
                else:
                    version_str = version.version
                    if args.format == "arff":
                        version_str = repr(version_str)
                num_mutual_neighbors = sum(1 for neighbor in graph.neighbors(node) if graph.has_edge(neighbor, node))
                if node in clean_page_rank:
                    norm_centrality = clean_page_rank[node]
                else:
                    norm_centrality = 0.0
                print(f"{node.ip!s},{continent},{country},{city},{['TRUE', 'FALSE'][node.last_crawled() is None]},"
                      f"{version_str},{graph.out_degree[node]},{graph.in_degree[node]},{num_mutual_neighbors},"
                      f"{page_rank[node]},{norm_centrality}")
            t.update(1)


class Topology(Command):
    name = "topology"
    help = "analyze the topology of a network"

    def __init_arguments__(self, parser: ArgumentParser):
        parser.add_argument("CRAWL_DB_FILE", type=str,
                            help="path to the crawl database")
        parser.add_argument("--group-by",
                            "-g",
                            default="ip",
                            choices=["ip", "city", "country", "continent", "version"],
                            help="grouping of nodes (default: %(default)s)")
        parser.add_argument("--only-crawled-nodes", action="store_true", help="only analyze nodes that were crawled; "
                                                                              "do not include nodes that we discovered "
                                                                              "but did not connect to")
        parser.add_argument("--conglomerate", "-c", action="store_true",
                            help="when calculating the PageRank of a group, instead of summing the constituent nodes'"
                                 "ranks (the default), treat each group as its own supernode and use the PageRank of "
                                 "the group nodes in the intersection graph formed by the groups")
        parser.add_argument("--degree-dist", type=str, default=None,
                            help="an optional path to an output file that will be a GNUplot graph of the degree "
                                 "distribution of the nodes")

    def run(self, args):
        raw_graph = CrawlGraph.load(CrawlDatabase(args.CRAWL_DB_FILE), only_crawled_nodes=args.only_crawled_nodes)
        num_nodes = len(raw_graph)
        if num_nodes == 0:
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
            elif args.group_by == "version":
                def grouper(n: CrawledNode) -> str:
                    v = n.get_version()
                    if v is None:
                        return "?"
                    return v.version
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
                print(f"{node.name}\t{rank}\t{len(node)}\t{len(node) / num_nodes * 100.0:0.1f}")
            else:
                print(f"[{node.ip!s}]:{node.port}\t{rank}\t1\t{1 / num_nodes * 100.0:0.1f}")
        print(f"Edge Connectivity: {nx.edge_connectivity(graph)}")
        print(f"Out-degree: {Statistics((graph.out_degree[node] for node in graph))!s}")
        # print(f"Average shortest path length: {nx.average_shortest_path_length(graph)}")
        if not args.degree_dist:
            return 0

        with open(args.degree_dist, "w") as f:
            pdf_path = f"{Path(args.degree_dist).stem}.pdf"
            f.write(f"""set term pdf enhanced color
set output \"{pdf_path}\"
set logscale y
set ylabel 'Number of Nodes'
set xlabel 'Node Degree'

plot '-' using ($1):(1.0) smooth freq with boxes t ''
""")
            for node in graph:
                f.write(f"{graph.out_degree[node]}\n")
            sys.stderr.write(f"\nDegree distribution graph saved to {args.degree_dist}\n"
                             f"Run `gnuplot {args.degree_dist}` to generate the graph in {pdf_path}\n")
        return 0
