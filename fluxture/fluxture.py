from abc import abstractmethod, ABC, ABCMeta
from argparse import ArgumentParser, Namespace
from inspect import isabstract
from typing import Dict, Tuple, Type


PLUGINS: Dict[str, Type["Plugin"]] = {}
COMMANDS: Dict[str, Type["Command"]] = {}


class PluginMeta(ABCMeta):
    def __init__(cls, name, bases, clsdict):
        super().__init__(name, bases, clsdict)
        if not isabstract(cls) and name not in ("Plugin", "Command"):
            if "name" not in clsdict:
                raise TypeError(f"Fluxture plugin {name} does not define a name")
            elif clsdict["name"] in PLUGINS:
                raise TypeError(f"Cannot instaitiate class {cls.__name__} because a plugin named {name} already exists,"
                                f" implemented by class {PLUGINS[clsdict['name']]}")
            PLUGINS[clsdict["name"]] = cls
            if issubclass(cls, Command):
                if "help" not in clsdict:
                    raise TypeError(f"Fluxture command {name} does not define a help string")
                COMMANDS[clsdict["name"]] = cls


class Plugin(ABC, metaclass=PluginMeta):
    name: str


class Command(Plugin):
    help: str
    parent_parsers: Tuple[ArgumentParser, ...] = ()

    def __init__(self, argument_parser: ArgumentParser):
        self.__init_arguments__(argument_parser)

    def __init_arguments__(self, parser: ArgumentParser):
        pass

    @abstractmethod
    def run(self, args: Namespace):
        raise NotImplementedError()


def add_command_subparsers(parser: ArgumentParser):
    subparsers = parser.add_subparsers(title="command", description="valid fluxture commands",
                                       help="run `fluxture command --help` for help on a specific command")
    for name, command_type in COMMANDS.items():
        p = subparsers.add_parser(name, parents=command_type.parent_parsers, help=command_type.help)
        p.set_defaults(func=command_type(p).run)
