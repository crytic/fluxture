import argparse

# We need to import these modules so that their commands/plugins auto-register:
from . import bitcoin, crawler

from .fluxture import add_command_subparsers


def main():
    parser = argparse.ArgumentParser(description="Fluxture: a peer-to-peer network crawler")

    add_command_subparsers(parser)

    args = parser.parse_args()

    args.func(args)


if __name__ == "__main__":
    main()
