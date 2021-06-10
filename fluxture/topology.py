import sys
from argparse import ArgumentParser
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import (
    Callable, Dict, FrozenSet, Generic, Hashable, Iterable, List, Optional, OrderedDict as OrderedDictType, Set,
    TypeVar, Union
)

import graphviz
import networkx as nx
import numpy as np
from tqdm import tqdm, trange

from .crawl_schema import CrawledNode, Version
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
        if only_crawled_nodes:
            nodes = set(db.crawled_nodes)
        else:
            nodes = db.nodes
        for node in tqdm(nodes, leave=False, desc="Constructing Topology", unit=" nodes"):
            graph.add_node(node)
            for to_node in node.get_latest_edges():
                if only_crawled_nodes and to_node not in nodes:
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

    def filter(self, predicate: Callable[[N], bool]) -> "CrawlGraph[N]":
        filtered: CrawlGraph[N] = CrawlGraph()
        for node in self:
            if predicate(node):
                filtered.add_node(node)
        for n1 in filtered.nodes:
            for n2 in filtered.nodes:
                if self.has_edge(n1, n2):
                    filtered.add_edge(n1, n2)
        return filtered


class OutDegree(Generic[N]):
    def __init__(self, graph: "ProbabilisticWeightedCrawlGraph[N]"):
        self.graph: ProbabilisticWeightedCrawlGraph[N] = graph

    def __getitem__(self, node: N) -> float:
        if node not in self.graph.node_indexes:
            return 0.0
        return sum(self.graph.adjacency[self.graph.node_indexes[node], :])


class InDegree(Generic[N]):
    def __init__(self, graph: "ProbabilisticWeightedCrawlGraph[N]"):
        self.graph: ProbabilisticWeightedCrawlGraph[N] = graph

    def __getitem__(self, node: N) -> float:
        if node not in self.graph.node_indexes:
            return 0.0
        return sum(self.graph.adjacency[:, self.graph.node_indexes[node]])


class ProbabilisticWeightedCrawlGraph(Generic[N]):
    """
    A complete graph with nonexistent edges filled in with probabilistic weights.

    Ideally this class would extend CrawlGraph (which extends nx.DiGraph), but NetworkX is too slow for dealing with
    complete graphs, so we need to use numpy directly.

    """
    def __init__(self, parent: CrawlGraph[N], max_neighbor_percent: float = 0.23):
        """
        Converts the parent crawl graph to a weighted graph where weights correspond to the probability of an edge.

        This is useful for clients like BitCoin Core which only report min(23%, 1000) peers.

        """
        if 1.0 < max_neighbor_percent <= 0.0:
            raise ValueError("max_neighbor_percent must be in the range (0.0, 1.0]")
        self.in_degree: InDegree[N] = InDegree(self)
        self.out_degree: OutDegree[N] = OutDegree(self)
        self.nodes: List[N] = list(parent.nodes)
        self.node_indexes: Dict[N, int] = {node: i for i, node in enumerate(self.nodes)}
        self.expected_actual_degrees: Dict[N, float] = {}
        num_nodes = len(self.nodes)
        self.adjacency = np.full((num_nodes, num_nodes), 0.0, dtype=np.float32)
        for node, row in self.node_indexes.items():
            degree = parent.out_degree[node]
            if degree == 0:
                # this means we didn't crawl the node yet, so use its incoming edges as bidirectional
                degree = parent.in_degree[node]
                for neighbor in parent.predecessors(node):
                    self.adjacency[row][self.node_indexes[neighbor]] = 1.0
            else:
                # add all of the existing outgoing edges:
                for neighbor in parent.neighbors(node):
                    self.adjacency[row][self.node_indexes[neighbor]] = 1.0
            expected_actual_degree = max(degree / max_neighbor_percent, 1.0)
            self.expected_actual_degrees[node] = expected_actual_degree
        self.expected_total_edges: float = sum(self.expected_actual_degrees.values())
        # now add the probabilistic edges
        for node, row in tqdm(self.node_indexes.items(), leave=False, desc="building probabilistic graph",
                              unit=" nodes", total=len(self.nodes)):
            existing_edges = parent.out_degree[node]
            if existing_edges == 0:
                # this means we didn't crawl node yet, so assume the edges are bidirectional:
                existing_edges = parent.in_degree[node]
            weight_to_add = self.expected_actual_degrees[node] - existing_edges
            num_new_neighbors = num_nodes - 1 - existing_edges
            if weight_to_add <= 0.0 or num_new_neighbors <= 0:
                # this node does not need any more edges
                continue
            weight_per_new_neighbor = weight_to_add / num_new_neighbors
            self.adjacency[row, self.adjacency[row][0::] <= 0] = weight_per_new_neighbor

    def __len__(self):
        return len(self.nodes)

    def __iter__(self):
        return iter(self.nodes)

    def __getitem__(self, index: int) -> N:
        return self.nodes[index]

    def probabilistic_shortest_distances(self, tolerance: float = 1.0e-3) -> np.ndarray:
        """Returns a distance matrix for the expected shortest path between nodes"""
        # The values in our weighted adjacency are the probabilities that a path of length 1 exists between each node.
        # self.adjacency * self.adjacency == the probability that there is a path of length 2
        # In general, self.adjacency ** i == the probability that there is a path of length i
        n = len(self)
        adj = self.adjacency
        results = adj.copy()
        prior_probabilities = adj.copy()
        np.fill_diagonal(prior_probabilities, 1.0)
        first_residuals: Optional[float] = None
        last_residuals: Optional[float] = None
        for i in trange(2, len(self), desc="probabilistic shortest path", unit=" distances", leave=False):
            with tqdm(desc="iteration", total=6, unit=" steps", leave=False, initial=1) as t:
                adj = adj @ self.adjacency
                t.update(1)
                adj = adj.clip(min=0.0, max=1.0, out=adj)
                t.update(1)
                # (1 - prior_probabilities) == probability that the actual shortest path is >= i
                additions = (1 - prior_probabilities) * adj
                t.update(1)
                residuals = np.sum(additions)
                t.update(1)
                results += additions * i
                if first_residuals is None:
                    first_residuals = residuals  # type: ignore
                    last_residuals = first_residuals
                elif last_residuals is not None:
                    if residuals > last_residuals:
                        t.write("Warning: Residuals are not converging!", file=sys.stderr)
                    else:
                        last_residuals = residuals  # type: ignore
                        if first_residuals > 0:
                            convergence_percent = (1.0 - (last_residuals / first_residuals)) * 100.0
                            t.desc = f"prob. shortest path ({convergence_percent:.2f}% converged)"
                t.update(1)
                # tqdm.write(f"Residuals: {residuals}\n", file=sys.stderr)
                if residuals <= tolerance:
                    break
                prior_probabilities += additions
        return results

    def pagerank(self, alpha: float = 0.85, max_iterations: int = 100, tolerance: float = 1.0e-6) -> Dict[N, float]:
        # use power iteration because calculating the Eigenvectors is too slow for large matrices

        with tqdm(desc="calculating stationary distributon", total=2, leave=False) as t:
            n = len(self)
            if n == 0:
                return {}
            initial_weight = 1.0 / n
            pi = np.full((n,), initial_weight)

            with tqdm(desc="normalizing adjacency matrix", total=1, initial=1, leave=False):
                # take the L1 norm of the adjancency matrix
                # so all rows sum to 1
                normalized_adj = self.adjacency / np.linalg.norm(self.adjacency, ord=1, axis=1)

            t.update(1)

            with tqdm(max_iterations, desc="power iteration", leave=False, unit=" error") as r:
                last_diff: Optional[float] = None
                for iteration in range(max_iterations):
                    r.desc = f"power iteration {iteration + 1}"
                    # portion we give away:
                    pi_prime = np.dot(normalized_adj, (1.0 - alpha) * pi)
                    # add in the portion we keep:
                    pi_prime = pi * alpha + pi_prime

                    # take the L1 norm of the vector (so all of its values sum to 1):
                    pi_prime = pi_prime / np.linalg.norm(pi_prime, ord=1)

                    # check for convergence by comparing the previous value with the new value
                    diff = np.linalg.norm(pi_prime - pi, ord=1)
                    if last_diff is None:
                        r.total = diff
                        last_diff = diff
                        r.update(0)
                    elif diff > last_diff:
                        # TODO: Maybe bail early because we are diverging?
                        pass
                    else:
                        r.update(last_diff - diff)
                        last_diff = diff
                    pi = pi_prime
                    if diff < n * tolerance:
                        # we are done!
                        break
                else:
                    raise ValueError(f"Power iteration failed to converge in {max_iterations} iterations")

            t.update(1)

            return {self.nodes[i]: v for i, v in enumerate(pi)}


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


# From: https://cbeci.org/mining_map
AVERAGE_SHARE_OF_HASHRATE = {
    "CN": 0.6508,
    "US": 0.0742,
    "RU": 0.0690,
    "KZ": 0.0617,
    "MY": 0.0433,
    "IR": 0.0382,
    "CA": 0.0082,
    "DE": 0.0056,
    "NO": 0.0048,
    "VE": 0.0042
}


def estimate_miner_probability(
        nodes: Iterable[CrawledNode],
        hashrates_by_country: Optional[Dict[str, float]] = None
) -> Dict[N, float]:
    """Calculates the probability of each node being a miner based upon the global distribution of miners"""
    if hashrates_by_country is None:
        hashrates_by_country = AVERAGE_SHARE_OF_HASHRATE
    hashrate_sum = sum(hashrates_by_country.values())
    if 0 > hashrate_sum > 1.0:
        raise ValueError("The hashrates must sum to a value in the range [0, 1]")
    hashrate_remainder = 1.0 - hashrate_sum
    nodes_by_country: Dict[str, List[CrawledNode]] = defaultdict(list)
    nodes_not_in_mapping = 0
    with tqdm(desc="estimating miner distribution", total=2, initial=1, leave=False, unit=" steps") as t:
        for node in tqdm(nodes, desc="geolocating", leave=False, unit=" nodes"):
            loc = node.get_location()
            if loc is None or loc.country_code is None:
                nodes_by_country[""].append(node)
                nodes_not_in_mapping += 1
            else:
                nodes_by_country[loc.country_code].append(node)
                if loc.country_code not in hashrates_by_country:
                    nodes_not_in_mapping += 1
        t.update(1)
        ret: Dict[CrawledNode, float] = {}
        for country_code, nodes_in_country in tqdm(
                nodes_by_country.items(),
                desc="extrapolating hashrate",
                leave=False,
                unit=" countries"
        ):
            if not nodes_in_country:
                # there are no nodes in this country (this should never happen)
                continue
            elif country_code in hashrates_by_country:
                probability = hashrates_by_country[country_code] / len(nodes_in_country)
            elif nodes_not_in_mapping == 0:
                # this should never happen
                continue
            else:
                probability = hashrate_remainder / nodes_not_in_mapping
            ret.update({node: probability for node in nodes_in_country})
    return ret


def expected_average_shortest_distance_to_miner(
        crawl_graph: Union[ProbabilisticWeightedCrawlGraph[CrawledNode], CrawlGraph[CrawledNode]],
        distances: Optional[np.ndarray] = None,
        miner_probability: Optional[Dict[CrawledNode, float]] = None
) -> Dict[CrawledNode, float]:
    """Estimates the average shortest distance to a miner for each node in the graph"""
    if not isinstance(crawl_graph, ProbabilisticWeightedCrawlGraph):
        crawl_graph = ProbabilisticWeightedCrawlGraph(crawl_graph)
    if miner_probability is None:
        miner_probability = estimate_miner_probability(crawl_graph)
    if distances is None:
        distances = crawl_graph.probabilistic_shortest_distances()
    elif distances.ndim != 2 or distances.shape[0] != len(crawl_graph) or distances.shape[1] != len(crawl_graph):
        raise ValueError(f"distances is expected to be an {len(crawl_graph)}x{len(crawl_graph)} matrix")
    return {
        node: sum(distances[index][i] * miner_probability[crawl_graph.nodes[i]] for i in range(len(crawl_graph)))
        for node, index in tqdm((
            (n, crawl_graph.node_indexes[n]) for n in crawl_graph
        ), desc="calculating expected distance to miners", leave=False, unit=" nodes", total=len(crawl_graph))
    }


class ExportCommand(Command):
    name = "export"
    help = "export the crawl data"

    def __init_arguments__(self, parser: ArgumentParser):
        parser.add_argument("CRAWL_DB_FILE", type=str, help="path to the crawl database")
        parser.add_argument("--format", "-f", choices=["arff", "csv"], default="arff", help="the format in which to "
                                                                                            "export the data")
        parser.add_argument("--skip-centrality-analysis", action="store_true", help="skip the slow centrality analysis")
        parser.add_argument("--only-crawled-nodes", action="store_true", help="only export nodes to which a successful "
                                                                              "connection was established")

    def run(self, args):
        with tqdm(desc="exporting", leave=False, unit=" steps", total=8, initial=1) as t:
            with CrawlDatabase(args.CRAWL_DB_FILE) as crawl_db:
                graph = CrawlGraph.load(crawl_db, only_crawled_nodes=args.only_crawled_nodes,
                                        bidirectional_edges=False)
                t.update(1)
                if not args.skip_centrality_analysis:
                    weighted_graph = ProbabilisticWeightedCrawlGraph(graph)
                t.update(1)
                if not args.skip_centrality_analysis:
                    weighted_page_rank = weighted_graph.pagerank()
                t.update(1)
                if not args.skip_centrality_analysis:
                    weighted_crawled_graph = ProbabilisticWeightedCrawlGraph(
                        graph.filter(lambda n: n.last_crawled() is not None or n.get_version() is not None)
                    )
                    weighted_crawled_graph_rank = weighted_crawled_graph.pagerank()
                t.update(1)

                if not args.skip_centrality_analysis:
                    miner_probability = estimate_miner_probability(weighted_graph)
                    t.update(1)
                    distances = weighted_graph.probabilistic_shortest_distances()
                    t.update(1)
                    avg_dist_to_miner = expected_average_shortest_distance_to_miner(
                        crawl_graph=weighted_graph, distances=distances, miner_probability=miner_probability
                    )
                    t.update(1)

                def after_period(obj) -> str:
                    text = str(obj)
                    period_pos = text.find(".")
                    if period_pos > 0:
                        return text[period_pos + 1:]
                    return text

                if args.format == "arff":
                    cities: Set[str] = {"?"}
                    countries: Set[str] = {"?"}
                    continents: Set[str] = {"?"}
                    versions: Set[str] = {"?"}
                    states: Set[str] = set()
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
                            versions.add(repr(version.version))
                        states.add(after_period(node.state))
                    print(f"""% Fluxture crawl
% Source: {args.CRAWL_DB_FILE}
        
@RELATION topology
        
@ATTRIBUTE ip                     STRING
@ATTRIBUTE continent              {{{','.join(map(repr, continents))}}}
@ATTRIBUTE country                {{{','.join(map(repr, countries))}}}
@ATTRIBUTE city                   {{{','.join(map(repr, cities))}}}
@ATTRIBUTE connected              {{TRUE, FALSE}}
@ATTRIBUTE state                  {{{','.join(map(repr, states))}}}
@ATTRIBUTE version                {{{','.join(versions)}}}
@ATTRIBUTE out_degree             NUMERIC
@ATTRIBUTE in_degree              NUMERIC
@ATTRIBUTE mutual_neighbors       NUMERIC
@ATTRIBUTE centrality             NUMERIC
@ATTRIBUTE miner_probability      NUMERIC
@ATTRIBUTE avg_shortest_dist      NUMERIC
@ATTRIBUTE expected_dist_to_miner NUMERIC
@ATTRIBUTE crawled_out_degree     NUMERIC
@ATTRIBUTE crawled_in_degree      NUMERIC
@ATTRIBUTE crawled_centrality     NUMERIC
        
@DATA
""")
                else:
                    # Assume CSV format
                    print("ip,continent,country,city,connected,state,version,out_degree,in_degree,mutual_neighbors,"
                          "centrality,miner_probability,avg_shortest_dist,expected_dist_to_miner,crawled_out_degree,"
                          "crawled_in_degree,crawled_centrality")
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
                    if not args.skip_centrality_analysis:
                        if node in weighted_crawled_graph_rank:
                            weighted_rank = str(weighted_crawled_graph_rank[node])
                        else:
                            weighted_rank = ""
                        base_weighted_rank = str(weighted_page_rank[node])
                        miner_prob = str(miner_probability[node])
                        avg_shortest_distance = str(
                            sum(distances[weighted_graph.node_indexes[node]])/len(weighted_graph)
                        )
                        adtm = str(avg_dist_to_miner[node])
                        weighted_out_degree = str(weighted_crawled_graph.out_degree[node])
                        weighted_in_degree = str(weighted_crawled_graph.in_degree[node])
                    else:
                        weighted_rank = ""
                        base_weighted_rank = ""
                        miner_prob = ""
                        avg_shortest_distance = ""
                        adtm = ""
                        weighted_out_degree = ""
                        weighted_in_degree = ""
                    if args.only_crawled_nodes:
                        out_degree = len(node.get_latest_edges())
                    else:
                        out_degree = graph.out_degree[node]

                    num_mutual_neighbors = sum(
                        1 for neighbor in graph.neighbors(node) if graph.has_edge(neighbor, node)
                    )
                    print(f"{node.ip!s},{continent},{country},{city},{['TRUE', 'FALSE'][node.last_crawled() is None]},"
                          f"{repr(after_period(node.state))},"
                          f"{version_str},{out_degree},{graph.in_degree[node]},{num_mutual_neighbors},"
                          f"{base_weighted_rank},{miner_prob},{avg_shortest_distance},{adtm},{weighted_out_degree},"
                          f"{weighted_in_degree},{weighted_rank}")
                t.update(1)


def kl_divergence(dist1: Iterable[float], dist2: Iterable[float]) -> float:
    """Calculates the Kullback-Liebler divergence of two distributions"""
    values1 = np.asarray(dist1, dtype=np.float)
    values2 = np.asarray(dist2, dtype=np.float)

    if values1.shape != values2.shape:
        raise ValueError("The distrubitons must have the same number of values!")

    # make sure the values sum to 1.0:
    values1 /= values1.sum(0)
    values2 /= values2.sum(0)

    return np.sum(np.where(values1 != 0, values1 * np.log(values1 / values2), 0))  # type: ignore


class UnreachableNodes(Command):
    name = "unreachable"
    help = "reports statistics on nodes that were reported as neighbors by nodes we crawled but were unreachable"

    def __init_arguments__(self, parser: ArgumentParser):
        parser.add_argument("CRAWL_DB_FILE", type=str,
                            help="path to the crawl database")

    def run(self, args):
        with CrawlDatabase(args.CRAWL_DB_FILE) as db:
            num_nodes = len(db.nodes)
            crawled_nodes = set(db.crawled_nodes)
            print(f"% crawled:\t{len(crawled_nodes)/num_nodes}")
            num_out_edges = 0
            num_crawled_out_edges = 0
            for node in crawled_nodes:
                neighbors = node.get_latest_edges()
                num_out_edges += len(neighbors)
                num_crawled_out_edges += sum(1 for neighbor in neighbors if neighbor in crawled_nodes)
            print(f"% unreachable:\t{num_crawled_out_edges/num_out_edges}")


class NodeRemoval(Command):
    name = "removal"
    help = "tests the hypothetical effect on the remaining nodes' consensus if different subgroups of the network " \
           "were removed from the network"

    def __init_arguments__(self, parser: ArgumentParser):
        parser.add_argument("CRAWL_DB_FILE", type=str,
                            help="path to the crawl database")

    def run(self, args):
        graph = CrawlGraph.load(CrawlDatabase(args.CRAWL_DB_FILE), only_crawled_nodes=True, bidirectional_edges=False)
        ordered_nodes = list(graph)
        weighted_graph = ProbabilisticWeightedCrawlGraph(graph)
        weighted_page_rank = weighted_graph.pagerank()
        nodes_by_country: Dict[str, Set[CrawledNode]] = defaultdict(set)
        for node in graph:
            loc = node.get_location()
            if loc is not None:
                if loc.country_code is None:
                    nodes_by_country["?"].add(node)
                else:
                    nodes_by_country[loc.country_code].add(node)
            else:
                nodes_by_country["?"].add(node)
        miner_probability = estimate_miner_probability(weighted_graph)
        distances = weighted_graph.probabilistic_shortest_distances()
        avg_dist_to_miner = expected_average_shortest_distance_to_miner(
            crawl_graph=weighted_graph, distances=distances, miner_probability=miner_probability
        )
        tqdm.write("country,centrality change,distance to miner change", file=sys.stdout)
        for country_to_remove, nodes_to_remove in nodes_by_country.items():
            # Calculate the change in centrality
            centrality_before = [weighted_page_rank[n] for n in ordered_nodes if n not in nodes_to_remove]
            modified_graph = graph.filter(lambda n: n not in nodes_to_remove)
            modified_weighted_graph = ProbabilisticWeightedCrawlGraph(modified_graph)
            modified_weighted_page_rank = modified_weighted_graph.pagerank()
            centrality_after = [modified_weighted_page_rank[n] for n in ordered_nodes if n not in nodes_to_remove]
            centrality_change = kl_divergence(centrality_before, centrality_after)

            # Calculate the change in distance to miners
            distances_before = [avg_dist_to_miner[n] for n in ordered_nodes if n not in nodes_to_remove]
            modified_distances = modified_weighted_graph.probabilistic_shortest_distances()
            modified_avg_dist_to_miner = expected_average_shortest_distance_to_miner(
                crawl_graph=modified_weighted_graph, distances=modified_distances, miner_probability=miner_probability
            )
            distances_after = [modified_avg_dist_to_miner[n] for n in ordered_nodes if n not in nodes_to_remove]
            distances_change = kl_divergence(distances_before, distances_after)

            tqdm.write(f"{country_to_remove},{centrality_change},{distances_change}", file=sys.stdout)


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
            # OrderedDictType[Union[CrawledNode, NodeGroup[CrawledNode]], float]
            page_rank: Dict[Union[CrawledNode, NodeGroup[CrawledNode]], float] = \
                ProbabilisticWeightedCrawlGraph(graph).pagerank()
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
                page_rank = ProbabilisticWeightedCrawlGraph(graph).pagerank()
            else:
                graph.parent = ProbabilisticWeightedCrawlGraph(raw_graph)
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
