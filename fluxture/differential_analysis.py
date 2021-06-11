from argparse import Action, ArgumentParser, ArgumentTypeError
from contextlib import ExitStack
from typing import Dict, Iterable, List, Optional

from tqdm import tqdm

from .db import Cursor
from .crawl_schema import CrawlDatabase, CrawlState
from .fluxture import Command
from .geolocation import Geolocation


def required_length(nmin: int = 0, nmax: Optional[int] = None):
    class RequiredLength(Action):
        def __call__(self, parser, args, values, option_string=None):
            n = len(values)
            if nmax is not None:
                if not (nmin <= n <= nmax):
                    raise ArgumentTypeError(f"argument {self.dest!r} requires between {nmin} and {nmax} arguments")
            elif n < nmin:
                raise ArgumentTypeError(f"argument {self.dest!r} requires at least {nmin} arguments")
            setattr(args, self.dest, values)

    return RequiredLength


class CompareCommand(Command):
    name = "compare"
    help = "compare two or more crawls against each other"

    def __init_arguments__(self, parser: ArgumentParser):
        parser.add_argument("trait", choices=("location", "topology"), help="the trait to compare")
        parser.add_argument("CRAWL_DB_FILE", type=str, nargs="+", action=required_length(2),
                            help="two or more crawl databases to compare")
        parser.add_argument("--only-crawled-nodes", action="store_true", help="only compare nodes to which a "
                                                                              "connection was successful")

    @staticmethod
    def compare_location(*dbs: CrawlDatabase, only_crawled_nodes: bool = False):
        countries: Dict[str, Dict[str, int]] = {}
        for db in tqdm(dbs, leave=False, unit=" crawls", desc="processing"):
            if only_crawled_nodes:
                locations: Iterable[Geolocation] = Cursor(
                    db.locations,
                    f"SELECT DISTINCT l.*, l.rowid FROM {db.locations.name} l, {db.nodes.name} n "
                    f"LEFT OUTER JOIN {db.locations.name} b ON l.ip = b.ip AND l.timestamp < b.timestamp "
                    "WHERE n.state >= ? AND l.ip = n.ip",
                    (CrawlState.CONNECTED,)
                )
            else:
                locations = db.locations
            for loc in tqdm(locations, leave=False, unit=" nodes", desc="locating"):
                if loc.country_code is None:
                    country = "?"
                else:
                    country = loc.country_code
                paths = countries.setdefault(country, {})
                paths[db.path] = paths.get(db.path, 0) + 1
        print(f"country,{','.join(db.path for db in dbs)}")
        for country, paths in countries.items():
            print(f"{country},{','.join(str(paths.get(db.path, 0)) for db in dbs)}")

    def run(self, args):
        if not hasattr(self, f"compare_{args.trait}"):
            raise NotImplementedError(f"TODO: Implement {self.__class__.__name__}.compare_{args.trait}")
        with ExitStack() as stack:
            dbs: List[CrawlDatabase] = []
            for file in args.CRAWL_DB_FILE:
                crawl_db = CrawlDatabase(file)
                stack.enter_context(crawl_db)
                dbs.append(crawl_db)
            return getattr(self, f"compare_{args.trait}")(*dbs, only_crawled_nodes=args.only_crawled_nodes)
