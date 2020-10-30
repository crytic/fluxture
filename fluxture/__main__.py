import argparse

from .bitcoin import Bitcoin, BitcoinNode
from .crawler import Crawler, CrawlDatabase, DatabaseCrawl


def main():
    parser = argparse.ArgumentParser(description="Fluxture: a peer-to-peer network crawler")
    parser.add_argument("--database", "-db", type=str, default=":memory:",
                        help="path to the crawl database (default is to run in memory)")

    args = parser.parse_args()

    Crawler(blockchain=Bitcoin(), crawl=DatabaseCrawl(BitcoinNode, CrawlDatabase(args.database))).do_crawl()


if __name__ == "__main__":
    main()
