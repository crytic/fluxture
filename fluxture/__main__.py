import argparse
import sys

from .bitcoin import Bitcoin, BitcoinNode
from .crawler import Crawler, CrawlDatabase, DatabaseCrawl
from .fluxture import add_command_subparsers
from .geolocation import GeoIP2Error, GeoIP2Locator


def main():
    parser = argparse.ArgumentParser(description="Fluxture: a peer-to-peer network crawler")
    parser.add_argument("--database", "-db", type=str, default=":memory:",
                        help="path to the crawl database (default is to run in memory)")
    parser.add_argument("--city-db-path", "-c", type=str, default=None,
                        help="path to a MaxMind GeoLite2 City database (default is "
                             "`~/.config/fluxture/geolite2/GeoLite2-City.mmdb`); "
                             "if omitted and `--maxmind-license-key` is provided, the latest database will be "
                             "downloaded and saved to the default location; "
                             "if both options are omttied, then geolocation will not be performed")
    parser.add_argument("--maxmind-license-key", type=str, default=None,
                        help="License key for automatically downloading a GeoLite2 City database; you generate get a "
                             "free license key by registering at https://www.maxmind.com/en/geolite2/signup")

    add_command_subparsers(parser)

    args = parser.parse_args()

    args.func(args)
    exit(0)

    try:
        geo = GeoIP2Locator(args.city_db_path, args.maxmind_license_key)
    except GeoIP2Error as e:
        sys.stderr.write(f"Warning: {e}")
        geo = None

    def crawl():
        Crawler(
            blockchain=Bitcoin(),
            crawl=DatabaseCrawl(BitcoinNode, CrawlDatabase(args.database)),
            geolocator=geo
        ).do_crawl()

    if geo is None:
        crawl()
    else:
        with geo:
            crawl()


if __name__ == "__main__":
    main()
