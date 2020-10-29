import sqlite3
from typing import (
    Any, cast, Dict, Generic, Iterable, Iterator, List, Optional, OrderedDict, Tuple, Type, TypeVar, Union
)

from .serialization import Packable
from fluxture.struct import Struct, StructMeta

FieldType = Union[bool, int, float, str, bytes, Packable, "ForeignKey"]

T = TypeVar("T", bound=FieldType)


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


COLUMN_TYPES: List[Type[Any]] = [int, str, bytes, float, Packable]


class Model(Struct[FieldType]):
    non_serialized = "primary_key_name",
    primary_key_name: Optional[str] = None

    @classmethod
    def validate_fields(cls, fields: OrderedDict[str, Type[FieldType]]):
        primary_name = None
        for field_name, field_type in fields.items():
            if not issubclass(field_type, ForeignKey):
                for valid_type in COLUMN_TYPES:
                    if issubclass(field_type, valid_type):
                        break
                else:
                    raise TypeError(f"Database field {field_name} of {cls.__name__} is type {field_type!r}, but "
                                    f"must be one of {COLUMN_TYPES!r}")
            if hasattr(field_type, "primary_key") and field_type.primary_key:
                if primary_name is not None:
                    raise TypeError(f"A model can have at most one primary key, but both {primary_name} and "
                                    f"{field_name} were specified in {cls.__name__}")
                primary_name = field_name
        if fields:
            if primary_name is None:
                if len(fields) == 1:
                    # just make the sole field a primary key
                    primary_name, field_type = next(iter(fields.items()))
                    setattr(field_type, "primary_key", True)
                else:
                    raise TypeError(f"Table {cls.__name__} does not specify a primary key")
        cls.primary_key_name = primary_name

    def save(self, db: "Database"):
        db[self.__class__].append(self)

    def to_row(self) -> Iterator[FieldType]:
        return iter((getattr(self, f) for f in self.keys()))


M = TypeVar("M", bound=Model)


class DatabaseConnection:
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self._con: Optional[sqlite3.Connection] = None
        self._entries: int = 0
        if ("database" in self._kwargs and self._kwargs["database"] == ":memory:") or \
                (self._args and self._args[0] == ":memory:"):
            # if using an in-memory database, always stay connected
            self.__enter__()

    def cursor(self) -> sqlite3.Cursor:
        return self._con.cursor()

    def commit(self):
        self._con.commit()

    def rollback(self, *args, **kwargs):
        self._con.rollback(*args, **kwargs)

    def execute(self, sql: str, *parameters: COLUMN_TYPES):
        params = []
        for p in parameters:
            if isinstance(p, str) or isinstance(p, bytes) or isinstance(p, int) or isinstance(p, float):
                params.append(p)
            elif isinstance(p, ForeignKey):
                params.append(p.key)
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
        if exc_type is None:
            # no exception occurred
            self.commit()
        else:
            # an exception occurred
            self.rollback()
        if self._entries <= 1 and self._con is not None:
            self._con.close()
            self._con = None
        self._entries = max(self._entries - 1, 0)

    def __getattr__(self, item):
        if self._con is None:
            raise AttributeError(item)
        return getattr(self._con, item)


class Table(Generic[M]):
    def __init__(self, db: "Database", model_type: Type[M]):
        self.db: Database = db
        self.model_type: Type[M] = model_type
        self.name: str = model_type.__name__
        for field_type in self.model_type.FIELDS.values():
            if isinstance(field_type, ForeignKey):
                field_type.table = self

    def __iter__(self) -> Iterator[M]:
        yield from self.select()

    def select(
            self,
            distinct: bool = False,
            limit: Optional[int] = None,
            order_by: Optional[str] = None,
            **kwargs
    ) -> Iterator[M]:
        params = []
        where_clauses = []
        for col_name, value in kwargs.items():
            where_clauses.append(f"{col_name}=?")
            params.append(value)
        if where_clauses:
            clauses = [f"WHERE {' AND '.join(where_clauses)}"]
        else:
            clauses = []
        if order_by is not None:
            clauses.append(" ORDER BY ?")
            params.append(order_by)
        if limit is not None:
            clauses.append(" LIMIT ?")
            params.append(limit)
        clauses = "".join(clauses)
        if clauses:
            clauses = " " + clauses
        if distinct:
            distinct_clause = " DISTINCT"
        else:
            distinct_clause = ""
        with self.db:
            cur = self.db.con.cursor()
            try:
                for row in cur.execute(f"SELECT{distinct_clause} * from {self.name}{clauses}", params):
                    r = self.model_type(*row)
                    for field_name, field_type in self.model_type.FIELDS.items():
                        if issubclass(field_type, ForeignKey):
                            getattr(r, field_name).table = self
                    yield r
            finally:
                cur.close()

    def __len__(self):
        with self.db:
            cur = self.db.con.cursor()
            try:
                result = cur.execute(f"SELECT COUNT(*) from {self.name}")
                return result.fetchone()[0]
            finally:
                cur.close()

    def append(self, row: M):
        with self.db:
            self.db.con.execute(f"INSERT INTO {self.name} ({','.join(row.FIELDS.keys())}) VALUES "
                                f"({','.join(['?']*len(row.FIELDS))});", *row.to_row())

    def extend(self, rows: Iterable[M]):
        with self.db:
            rows = list(rows)
            if not rows:
                return
            row_data = [
                tuple(row.to_row()) for row in rows
            ]
            first_row = rows[0]
            self.db.con.executemany(f"INSERT INTO {self.name} ({','.join(first_row.FIELDS.keys())}) VALUES "
                                    f"({','.join(['?']*len(first_row.FIELDS))});", row_data)


class ForeignKey(Generic[M]):
    row_type: Type[M]
    foreign_col_name: str
    table: Optional[Table[M]] = None

    def __init__(self, key: Union[int, str, bytes, float], table: Optional[Table[M]] = None):
        self.key: Union[int, str, bytes, float] = key
        if table is not None:
            self.table = table
        self._row: Optional[M] = None

    def __class_getitem__(cls, arguments: Union[TypeVar, Type[M], Tuple[Type[M], str], Table[M]]) -> "ForeignKey[M]":
        if isinstance(arguments, TypeVar):
            return cls
        elif isinstance(arguments, tuple):
            model, row_name = arguments
        else:
            if isinstance(arguments, Table):
                if not hasattr(cls, "foreign_col_name") or not cls.foreign_col_name:
                    raise ValueError("A table can only be passed to a ForeignKey that already has a `foreign_col_name`")
                return cast(
                    ForeignKey[M],
                    type(f"{cls.__name__}{arguments.model_type.__name__.capitalize()}"
                         f"{cls.foreign_col_name.replace('_', '').capitalize()}", (cls,),
                         {
                             "table": arguments
                         })
                )
            model = arguments
            row_name = model.primary_key_name
        return cast(
            ForeignKey[M],
            type(f"{cls.__name__}{model.__name__.capitalize()}{row_name.replace('_', '').capitalize()}", (cls,), {
                "row_type": model,
                "foreign_col_name": row_name
            })
        )

    @classmethod
    def key_type(cls) -> Type[Union[int, float, str, bytes, Packable]]:
        return cls.row_type.FIELDS[cls.foreign_col_name]

    @property
    def row(self) -> M:
        if self._row is None:
            if self.table is None:
                raise ValueError(f"{self.__class__.__name__} must have a `table` set")
            foreign_table = self.table.db[self.row_type]
            self._row = next(iter(foreign_table.select(**{self.foreign_col_name: self.key})))
        return self._row

    def __getattr__(self, item):
        return getattr(self.row, item)

    def __eq__(self, other):
        if isinstance(other, ForeignKey):
            return self.key == other.key
        else:
            return self.row == other

    def __ne__(self, other):
        return not (self == other)

    def __lt__(self, other):
        if isinstance(other, ForeignKey):
            return self.key < other.key
        else:
            return self.row < other

    def __hash__(self):
        return hash(self.key)

    def __repr__(self):
        return f"{self.__class__.__name__}(key={self.key!r})"


class Database(metaclass=StructMeta[Model]):
    def __init__(self, path: str = ":memory:"):
        self.path: str = path
        self.con = DatabaseConnection(self.path)
        if self.FIELDS:
            with self:
                self.tables: Dict[Type[Model], Table[Model]] = {
                    model: self.create_table(model) for model in self.FIELDS.values()
                }
        else:
            self.tables = {}

    @classmethod
    def validate_fields(cls, fields: OrderedDict[str, Type[FieldType]]):
        for field_name, field_type in fields.items():
            if not issubclass(field_type, Model):
                raise TypeError(f"Database table {field_name} of {cls.__name__} is type {field_type!r}, but "
                                "must be a subclass of `db.Model`")

    def __getitem__(self, table_type: Type[M]) -> Table[M]:
        return self.tables[table_type]

    def __contains__(self, table_type: Type[M]):
        return table_type in self.tables

    def __len__(self):
        return len(self.tables)

    def __enter__(self) -> "Database":
        self.con.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.con.__exit__(exc_type, exc_val, exc_tb)

    def create_table(self, model_type: Type[Model]) -> Table[Model]:
        columns = []
        for field_name, field_type in model_type.FIELDS.items():
            if issubclass(field_type, ForeignKey):
                old_field_type = field_type
                field_type = field_type.key_type()
                setattr(field_type, "primary_key", getattr(old_field_type, "primary_key", False))
                if hasattr(old_field_type, "default"):
                    setattr(field_type, "default", old_field_type.default)
                setattr(field_type, "unique", getattr(old_field_type, "unique", False))
                setattr(field_type, "not_null", getattr(old_field_type, "not_null", False))
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
        column_constraints = ",\n    ".join(columns)
        if len(columns) > 1:
            column_constraints = "\n    " + column_constraints + "\n"
        with self:
            self.con.execute(f"CREATE TABLE IF NOT EXISTS {model_type.__name__} ({column_constraints});")
        return Table(self, model_type)
