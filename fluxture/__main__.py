import argparse

from .fluxture import add_command_subparsers


def main():
    parser = argparse.ArgumentParser(description="Fluxture: a peer-to-peer network crawler")

    add_command_subparsers(parser)

    args = parser.parse_args()

    return args.func(args)


if __name__ == "__main__":
    main()
