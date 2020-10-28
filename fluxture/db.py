import sqlite3
from typing import Any, cast, List, Optional, OrderedDict, Type, TypeVar, Union

from .serialization import Packable
from fluxture.struct import Struct, StructMeta

COLUMN_TYPES: List[Type[Any]] = [int, str, bytes, float, Packable]
FieldType = Union[bool, int, float, str, bytes, Packable]

T = TypeVar("T", bound=Packable)


def _col_modifier(ty: Type[T], key_name: str, value: FieldType = True) -> Type[T]:
    return cast(
        Type[T],
        type(f"{ty.__name__}{''.join(key.capitalize() for key in key_name.split('_'))}", (ty,), {key_name: value})
    )


def primary_key(ty: Type[T]) -> Type[T]:
    return _col_modifier(ty, "primary_key")


def unique(ty: Type[T]) -> Type[T]:
    return _col_modifier(ty, "unique")


def not_null(ty: Type[T]) -> Type[T]:
    return _col_modifier(ty, "not_null")


def default(ty: Type[T], default_value: FieldType) -> Type[T]:
    return _col_modifier(ty, "default", default_value)


class Model(Struct[FieldType]):
    @classmethod
    def validate_fields(cls, fields: OrderedDict[str, Type[FieldType]]):
        for field_name, field_type in fields.items():
            for valid_type in COLUMN_TYPES:
                if issubclass(field_type, valid_type):
                    break
            else:
                raise TypeError(f"Database field {field_name} of {cls.__name__} is type {field_type!r}, but "
                                f"must be one of {COLUMN_TYPES!r}")


class DatabaseConnection:
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self._con: Optional[sqlite3.Connection] = None
        self._entries: int = 0

    def cursor(self) -> sqlite3.Cursor:
        return self._con.cursor()

    def commit(self):
        self._con.commit()

    def rollback(self, *args, **kwargs):
        self._con.rollback(*args, **kwargs)

    def execute(self, sql: str, *parameters: COLUMN_TYPES):
        params = []
        for p in parameters:
            if issubclass(p, str) or issubclass(p, bytes) or issubclass(p, int) or issubclass(p, float):
                params.append(p)
            elif isinstance(p, Packable):
                params.append(p.pack())
            else:
                raise ValueError(f"Unsupported parameter type: {p!r}")
        self._con.execute(sql, params)

    def __enter__(self) -> "DatabaseConnection":
        if self._entries == 0 and self._con is None:
            self._con = sqlite3.connect(*self._args, **self._kwargs)
        self._entries += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._entries <= 1 and self._con is not None:
            self._con.close()
            self._con = None
        self._entries = max(self._entries - 1, 0)

    def __getattr__(self, item):
        if self._con is None:
            raise AttributeError(item)
        return getattr(self._con, item)


class Database(metaclass=StructMeta[Model]):
    def __init__(self, path: str = ":memory:"):
        self.path: str = path
        self.con = DatabaseConnection(self.path)
        if self.FIELDS:
            with self:
                for model in self.FIELDS.values():
                    self.create_table(model)

    @classmethod
    def validate_fields(cls, fields: OrderedDict[str, Type[FieldType]]):
        for field_name, field_type in fields.items():
            if not issubclass(field_type, Model):
                raise TypeError(f"Database table {field_name} of {cls.__name__} is type {field_type!r}, but "
                                "must be a subclass of `db.Model`")

    def __enter__(self) -> "Database":
        self.con.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            # no exception occurred
            self.con.commit()
        else:
            # an exception occurred
            self.con.rollback()
        self.con.__exit__(exc_type, exc_val, exc_tb)

    def create_table(self, model_type: Type[Model]):
        columns = []
        for field_name, field_type in model_type.FIELDS.items():
            if issubclass(field_type, int):
                data_type = "INTEGER"
            elif issubclass(field_type, float):
                data_type = "REAL"
            elif issubclass(field_type, str):
                data_type = "TEXT"
            elif issubclass(field_type, bytes) or isinstance(field_type, Packable):
                data_type = "BLOB"
            else:
                raise TypeError(f"Column {field_name} is of unsupported type {field_type!r}; it must be one of "
                                f"{COLUMN_TYPES}")
            modifiers = []
            if hasattr(field_type, "primary_key") and field_type.primary_key:
                modifiers.append(" PRIMARY KEY")
            if hasattr(field_type, "unique") and field_type.unique:
                modifiers.append(" UNIQUE")
            if hasattr(field_type, "not_null") and field_type.not_null:
                modifiers.append(" NOT NULL")
            if hasattr(field_type, "default"):
                if field_type.default is None:
                    modifiers.append(" DEFAULT NULL")
                else:
                    modifiers.append(f" DEFAULT {field_type.default}")
            columns.append(f"{field_name} {data_type}{''.join(modifiers)}")
        column_constraints = "\n    ".join(columns)
        if len(columns) > 1:
            column_constraints = "\n    " + column_constraints + "\n"
        with self:
            self.con.execute(f"CREATE TABLE IF NOT EXISTS {model_type.__name__} ({column_constraints});")
