import sys
from abc import ABC
from argparse import ArgumentParser, Namespace
from getpass import getpass
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import keyring
from shodan import APIError, Shodan

from .fluxture import Command
from .crawl_schema import IPv6Address


KEYRING_NAME: str = "fluxture"


def prompt(
        message: str,
        yes_options: Tuple[str, ...] = ('y', 'yes'),
        no_options: Tuple[str, ...] = ('n', 'no'),
        default: bool = True
) -> bool:
    if default:
        yes_options = yes_options + ('',)
    else:
        no_options = no_options + ('',)
    while True:
        ret = input(message).strip().lower()
        if ret in yes_options:
            return True
        elif ret in no_options:
            return False


class ShodanResult:
    def __init__(self, **kwargs):
        if "ip" in kwargs:
            self.ip: IPv6Address = IPv6Address(kwargs["ip"])
        else:
            raise ValueError(f"Shodan Result does not contain an IP address: {kwargs!r}")
        if "isp" in kwargs:
            self.isp: Optional[str] = kwargs["isp"]
        else:
            self.isp = None
        if "ip_str" in kwargs:
            self.ip_str: Optional[str] = kwargs["ip_str"]
        else:
            self.ip_str = None
        if "os" in kwargs:
            self.os: Optional[str] = kwargs["os"]
        else:
            self.os = None
        self.result: Dict[str, Any] = kwargs

    def __getattr__(self, item):
        if item in self.result:
            return self.result[item]

    def __str__(self):
        if self.ip_str is not None:
            ip = self.ip_str
        else:
            ip = str(self.ip)
        if self.isp is not None:
            isp = f" on {self.isp}"
        else:
            isp = ""
        if self.os is not None:
            os = f" running {self.os}"
        else:
            os = ""
        return f"{ip}{isp}{os}"

    def __repr__(self):
        kwargs = "".join(
            [
                f", {argname!s}={argvalue!r}" for argname, argvalue in self.result.items()
                if argname != "ip" and argname != "isp"
            ]
        )
        return f"{self.__class__.__name__}(ip={self.ip!r}, isp={self.isp!r}{kwargs})"


class Query(ABC):
    def __init__(self, name: str, callback: Optional[Callable[["Query"], Any]] = None):
        self.name: str = name
        self.callback: Optional[Callable[[Query], Any]] = callback


SEARCH_QUERIES: Dict[str, "SearchQuery"] = {}


class SearchQuery(Query):
    def __init__(self, name: str, query: str, callback: Optional[Callable[["Query"], Any]] = None):
        super().__init__(name=name, callback=callback)
        self.query: str = query

    @staticmethod
    def register(name: str, query: str, callback: Optional[Callable[["Query"], Any]] = None) -> "SearchQuery":
        sq = SearchQuery(name=name, query=query, callback=callback)
        if name in SEARCH_QUERIES:
            raise KeyError(f"A search query of name \"{name}\" is already registered: {SEARCH_QUERIES[name]!r}")
        SEARCH_QUERIES[name] = sq
        return sq

    def run(self, api: Shodan) -> Iterator[ShodanResult]:
        return (ShodanResult(**result) for result in api.search_cursor(self.query))

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r}, query={self.query!r}, callback={self.callback!r})"


def get_keychain_api_key() -> Optional[str]:
    return keyring.get_password(KEYRING_NAME, "shodan_api_key")


def save_keychain_api_key(api_key: str):
    keyring.set_password(KEYRING_NAME, "shodan_api_key", api_key)


def get_api(api_key: Optional[str]) -> Shodan:
    keychain_key = get_keychain_api_key()
    if api_key is None:
        api_key = keychain_key
        if api_key is None:
            api_key = getpass(f"Shodan API KEY: ")
            if prompt("Would you like to save this API key to the system keychain for future use? [Yn] "):
                save_keychain_api_key(api_key)
    elif keychain_key is None:
        if prompt("Would you like to save this API key to the system keychain for future use? [Yn] "):
            save_keychain_api_key(api_key)
    elif api_key != keychain_key:
        print("This is a different API key than what is stored in the system keychain.")
        if prompt("Would you like to update the API key in the system keychain? [Yn] "):
            save_keychain_api_key(api_key)
    return Shodan(api_key)


class ShodanCommand:
    def __init_arguments__(self, parser: ArgumentParser):
        parser.add_argument("--api-key", "-k", type=str, default=None, help="Shodan API key. If omitted, a saved "
                            "API key in the system keychain will be used, if one exists. Otherwise the user will be "
                            "prompted to enter an API key.")


class ActiveNodes(ShodanCommand, Command):
    name = "active"
    help = "enumerate active nodes from Shodan"

    def __init_arguments__(self, parser: ArgumentParser):
        super().__init_arguments__(parser)
        parser.add_argument("QUERY", choices=SEARCH_QUERIES.keys())

    def run(self, args: Namespace):
        api = get_api(args.api_key)
        for result in SEARCH_QUERIES[args.QUERY].run(api):
            print(str(result))


class HostInfo(ShodanCommand, Command):
    name = "hostinfo"
    help = "get information about IP addresses from Shodan"

    def __init_arguments__(self, parser: ArgumentParser):
        super().__init_arguments__(parser)
        parser.add_argument("IP", nargs="+", type=str)

    def run(self, args: Namespace):
        api = get_api(args.api_key)
        for ip in args.IP:
            try:
                info = api.host(ip)
                for key, value in info.items():
                    print(f"{key!s}:\t{value!r}")
            except APIError as e:
                sys.stdout.flush()
                sys.stderr.write(str(e))
                sys.stderr.write("\n")
                sys.stderr.flush()
