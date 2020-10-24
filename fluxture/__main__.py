from .bitcoin import Bitcoin
from .crawler import Crawler, InMemoryCrawl


def main():
    Crawler(blockchain=Bitcoin(), crawl=InMemoryCrawl()).do_crawl()


if __name__ == "__main__":
    main()
