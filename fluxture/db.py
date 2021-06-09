import sqlite3
from typing import (
    Any, cast, Dict, Generic, Iterable, Iterator, List, Optional, OrderedDict, Tuple, Type, TypeVar, Union
)

from .serialization import Packable
from .structures import Struct, StructMeta

FieldType = Union[bool, int, float, str, bytes, Packable, "ForeignKey"]

T = TypeVar("T", bound=FieldType)
D = TypeVar("D")


class AutoIncrement(int):
    initialized: bool = False

    def __new__(cls, *args):
        if args and (len(args) > 1 or not isinstance(args[0], AutoIncrement) or args[0].initialized):
            retval = int.__new__(cls, *args)
            setattr(retval, "initialized", True)
        else:
            retval = int.__new__(cls, 0)
            setattr(retval, "initialized", False)
        return retval

    def __repr__(self):
        if self.initialized:
            return f"{self.__class__.__name__}({int(self)})"
        else:
            return f"{self.__class__.__name__}()"


class RowId(int):
    initialized: bool = False

    def __new__(cls, *args):
        if args and (len(args) > 1 or not isinstance(args[0], RowId) or args[0].initialized):
            retval = int.__new__(cls, *args)
            setattr(retval, "initialized", True)
        else:
            retval = int.__new__(cls, 0)
            setattr(retval, "initialized", False)
        return retval

    def __repr__(self):
        if self.initialized:
            return f"{self.__class__.__name__}({int(self)})"
        else:
            return f"{self.__class__.__name__}()"

    def __eq__(self, other):
        return isinstance(other, RowId) and (not self.initialized or not other.initialized or int(self) == int(other))


class ColumnOptions:
    def __init__(
        self,
        primary_key: Optional[bool] = None,
        unique: Optional[bool] = None,
        not_null: Optional[bool] = None,
        default: Optional[FieldType] = None,
        auto_increment: Optional[bool] = None,
    ):
        self.primary_key: Optional[bool] = primary_key
        self.unique: Optional[bool] = unique
        self.not_null: Optional[bool] = not_null
        self.default: Optional[FieldType] = default
        self.auto_increment: Optional[bool] = auto_increment
        if self.auto_increment and not self.default:
            self.default = AutoIncrement()

    def is_set(self, option_name: str):
        a = getattr(self, option_name)
        return a is not None and not callable(a)

    def set_options(self) -> Iterator[str]:
        return iter(
            key_name for key_name in dir(self) if not key_name.startswith("_") and self.is_set(key_name)
        )

    def items(self) -> Iterator[Tuple[str, Any]]:
        return iter((key_name, getattr(self, key_name)) for key_name in self.set_options())

    def __or__(self, other: "ColumnOptions") -> "ColumnOptions":
        new_options = ColumnOptions(**dict(other.items()))
        for key_name, value in self.items():
            if not other.is_set(key_name):
                setattr(new_options, key_name, value)
        return new_options

    def __sub__(self, other: "ColumnOptions") -> "ColumnOptions":
        return ColumnOptions(**{
            key_name: value for key_name, value in self.items() if not other.is_set(key_name)
        })

    def type_suffix(self) -> str:
        return "".join(
            [
                f"{''.join(key.capitalize() for key in key_name.split('_'))}"
                f"{''.join(val.capitalize() for val in str(value).replace('(', '').replace(')', '').split(' '))}"
                for key_name, value in self.items()
            ]
        )

    def sql_modifiers(self) -> str:
        modifiers = []
        if self.primary_key:
            modifiers.append("PRIMARY KEY")
        if self.auto_increment:
            modifiers.append("AUTOINCREMENT")
        if self.unique:
            modifiers.append("UNIQUE")
        if self.not_null:
            modifiers.append("NOT NULL")
        if self.default is not None and not isinstance(self.default, AutoIncrement):
            modifiers.append(f"DEFAULT {self.default}")
        return " ".join(modifiers)

    def __repr__(self):
        args = [f"{key}={value!r}" for key, value in self.items()]
        return f"{self.__class__.__name__}({', '.join(args)})"


def column_options(
        ty: Type[T],
        options: ColumnOptions
) -> Type[T]:
    if hasattr(ty, "column_options") and ty.column_options is not None:
        options = ty.column_options | options
        type_suffix = (options - ty.column_options).type_suffix()
    else:
        type_suffix = options.type_suffix()
    return cast(
        Type[T],
        type(f"{ty.__name__}{type_suffix}", (ty,), {"column_options": options})
    )


def primary_key(ty: Type[T]) -> Type[T]:
    return column_options(ty, ColumnOptions(primary_key=True))


def unique(ty: Type[T]) -> Type[T]:
    return column_options(ty, ColumnOptions(unique=True))


def not_null(ty: Type[T]) -> Type[T]:
    return column_options(ty, ColumnOptions(not_null=True))


def default(ty: Type[T], default_value: FieldType) -> Type[T]:
    return column_options(ty, ColumnOptions(default=default_value))


COLUMN_TYPES: List[Type[Any]] = [int, str, bytes, float, Packable]


S = TypeVar("S")
D = TypeVar("D", bound="Database")


class Model(Struct[FieldType], Generic[D]):
    non_serialized = "primary_key_name", "_db", "rowid"
    primary_key_name: str = "rowid"
    _db: Optional[D] = None
    rowid: RowId

    @staticmethod
    def is_primary_key(cls) -> bool:
        return hasattr(cls, "column_options") and cls.column_options is not None \
               and cls.column_options.primary_key

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
            if Model.is_primary_key(field_type):
                if primary_name is not None:
                    raise TypeError(f"A model can have at most one primary key, but both {primary_name} and "
                                    f"{field_name} were specified in {cls.__name__}")
                primary_name = field_name
        if primary_name is not None:
            cls.primary_key_name = primary_name
        if "rowid" not in fields:
            fields["rowid"] = default(RowId, RowId())

    @property
    def in_db(self) -> bool:
        return self.rowid.initialized and self._db is not None

    def uninitialized_auto_increments(self) -> Iterator[Tuple[str, AutoIncrement]]:
        for key in self.keys():
            value = getattr(self, key)
            if isinstance(value, AutoIncrement) and not value.initialized:
                yield key, value

    @property
    def db(self) -> D:
        if self._db is None:
            raise ValueError(f"Model {self!r} has not yet been added to a database")
        return self._db

    @db.setter
    def db(self, db: D):
        if self._db is not None:
            if self._db != db:
                raise ValueError(f"Model {self!r} is already associated with a different database: {db!r}")
        else:
            self._db = db

    def to_row(self) -> Iterator[FieldType]:
        for key in self.keys():
            value = getattr(self, key)
            if (isinstance(value, AutoIncrement) or isinstance(value, RowId)) and not value.initialized:
                yield None
            else:
                yield getattr(self, key)


M = TypeVar("M", bound=Model)


def sql_format(
        param: Optional[FieldType], expected_type: Optional[Type[FieldType]] = None
) -> Optional[Union[str, bytes, int, float]]:
    if param is None:
        if expected_type is not None and hasattr(expected_type, "column_options") \
                and expected_type.column_options is not None and expected_type.column_options.not_null:
            raise ValueError(f"Field {expected_type!r} cannot be NULL")
        return None
    elif isinstance(param, Model) and expected_type is not None:
        if not issubclass(expected_type, ForeignKey):
            raise ValueError(f"Model {param!r} was expected to be of type {expected_type!r}")
        return getattr(param, expected_type.key)
    elif isinstance(param, int):
        return int(param)
    elif isinstance(param, float):
        return float(param)
    elif isinstance(param, str):
        return str(param)
    elif isinstance(param, bytes):
        return bytes(param)
    elif isinstance(param, ForeignKey):
        return sql_format(param.key, expected_type)
    elif isinstance(param, Packable):
        return param.pack()
    else:
        raise ValueError(f"Unsupported parameter type: {param!r}")


class DatabaseConnection(sqlite3.Connection):
    def __init__(self, *args, rollback_on_exception: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.rollback_on_exception: bool = rollback_on_exception

    def execute(self, sql: str, *parameters: COLUMN_TYPES) -> sqlite3.Cursor:
        params = [sql_format(p) for p in parameters]
        try:
            return super().execute(sql, params)
        except sqlite3.Error as e:
            raise ValueError(f"Error executing SQL {sql!r} with parameters {params!r}: {e!r}")

    def __enter__(self) -> "DatabaseConnection":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None or not self.rollback_on_exception:
            # no exception occurred
            self.commit()
        else:
            # an exception occurred
            self.rollback()


class Cursor(Generic[M]):
    def __init__(self, table: "Table[M]", sql: str, params: Iterable[Union[str, int, float, bytes, Packable]] = ()):
        self.table: Table[M] = table
        self.sql: str = sql
        self.params: List[Union[str, int, float, bytes]] = []
        for i, p in enumerate(params):
            if isinstance(p, str) or isinstance(p, int) or isinstance(p, float) or isinstance(p, bytes):
                self.params.append(p)
            elif isinstance(p, Packable):
                self.params.append(p.pack())
            else:
                raise ValueError(f"Unsupported SQL parameter #{i+1}, {p!r}, when running {sql!r}")
        self._item_iter: Optional[Iterator[M]] = None

    def __iter__(self) -> Iterator[M]:
        if self._item_iter is None:
            self._item_iter = self._iter()
        yield from self._item_iter

    def _iter(self) -> Iterator[M]:
        with self.table.db:
            cur = self.table.db.con.cursor()
            try:
                for row in cur.execute(self.sql, self.params):
                    r = self.table.model_type(*row)
                    r.db = self.table.db
                    for field_name, field_type in self.table.model_type.FIELDS.items():
                        if issubclass(field_type, ForeignKey):
                            getattr(r, field_name).table = self.table
                    yield r
            finally:
                cur.close()

    def fetchone(self) -> Optional[M]:
        try:
            return next(iter(self))
        except StopIteration:
            return None

    def fetchall(self) -> Iterator[M]:
        yield from iter(self)


class Table(Generic[M]):
    model_type: Optional[Type[M]] = None

    def __init__(self, db: "Database", name: str):
        if self.model_type is None:
            raise TypeError(f"A Table must be instantiated by subclassing it with a model: `Table[ModelType]`")
        self.db: Database = db
        self.model_type: Type[M] = self.model_type
        self.name: str = name
        for field_type in self.model_type.FIELDS.values():
            if isinstance(field_type, ForeignKey):
                field_type.table = self

    def __class_getitem__(cls, model_type: Type[M]) -> Type["Table[M]"]:
        if isinstance(model_type, TypeVar) or isinstance(model_type, str):
            return cls
        return cast(Type[Table[M]], type(f"{cls.__name__}{model_type.__name__}", (cls,), {"model_type": model_type}))

    def __iter__(self) -> Iterator[M]:
        yield from iter(self.select())

    def select(
            self,
            distinct: bool = False,
            limit: Optional[int] = None,
            order_by: Optional[str] = None,
            order_direction: str = "ASC",
            **kwargs
    ) -> Cursor[M]:
        params = []
        where_clauses = []
        for col_name, value in kwargs.items():
            if not isinstance(value, AutoIncrement) or value.initialized:
                where_clauses.append(f"{col_name}=?")
                params.append(value)
        if where_clauses:
            clauses = [f"WHERE {' AND '.join(where_clauses)}"]
        else:
            clauses = []
        if order_by is not None:
            clauses.append(f" ORDER BY ? {order_direction}")
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
        return Cursor(self, f"SELECT{distinct_clause} *, rowid from {self.name}{clauses}", params)

    def __len__(self):
        with self.db:
            cur = self.db.con.cursor()
            try:
                result = cur.execute(f"SELECT COUNT(*) from {self.name}")
                return result.fetchone()[0]
            finally:
                cur.close()

    def _finalize_added_row(self, row: M):
        to_update = [key for key, _ in row.uninitialized_auto_increments()]
        if to_update:
            try:
                obj_in_db = next(iter(self.select(**{"rowid": row.rowid})))
            except StopIteration:
                raise ValueError(f"Row {row} was expected to be in the database in table {self.name}, but was not")
            for key in to_update:
                setattr(row, key, getattr(obj_in_db, key))
        row.db = self.db

    def append(self, row: M):
        self.extend((row,))

    def extend(self, rows: Iterable[M]):
        with self.db:
            rows = list(rows)
            if not rows:
                return
            cur = self.db.con.cursor()
            try:
                # we have to add each row individually so we can set its rowid
                for row in rows:
                    if row.in_db:
                        raise ValueError(f"Row {row!r} is already in the database!")
                    result = cur.execute(f"INSERT INTO {self.name} ({','.join(self.model_type.FIELDS.keys())}) "
                                         f"VALUES ({','.join(['?']*len(self.model_type.FIELDS))})", tuple(
                                            sql_format(param, expected_type)
                                            for param, expected_type
                                            in zip(row.to_row(), self.model_type.FIELDS.values())
                                         ))
                    setattr(row, "rowid", RowId(result.lastrowid))
                    self._finalize_added_row(row)
            finally:
                cur.close()

    def update(self, row: M):
        if not row.in_db:
            raise ValueError(f"Row {row!r} is not yet in the database!")
        with self.db:
            set_argument = ",".join([f"{field_name} = ?" for field_name in self.model_type.FIELDS.keys() if field_name != "rowid"])
            new_values = tuple(
                sql_format(param, expected_type)
                for param, (field_name, expected_type)
                in zip(row.to_row(), self.model_type.FIELDS.items())
                if field_name != "rowid"
            )
            cur = self.db.con.cursor()
            try:
                cur.execute(f"UPDATE {self.name} SET {set_argument} WHERE rowid=?", new_values + (int(row.rowid),))
            finally:
                cur.close()


class ForeignKey(Generic[M]):
    row_type: Type[M]
    foreign_table_name: str
    foreign_col_name: str
    table: Optional[Table[M]] = None

    def __init__(self, key: Union[int, str, bytes, float, M], table: Optional[Table[M]] = None):
        self._row: Optional[M] = None
        if table is not None:
            self.table = table
        if isinstance(key, Model):
            if not hasattr(self, "foreign_col_name"):
                raise ValueError(f"Foreign key {self!r} has not yet been assigned to a table!")
            elif not isinstance(key, self.row_type):
                raise ValueError(f"Foreign key {self!r} was expeted to be passed a value of type {self.row_type!r} "
                                 f"but was instead passed {key!r}")
            self.key: Union[int, str, bytes, float] = getattr(key, self.foreign_col_name)
        else:
            self.key = key

    def __class_getitem__(
            cls,
            arguments: Union[TypeVar, Tuple[str, Type[M]], Tuple[str, Type[M], str], Table[M]]
    ) -> "ForeignKey[M]":
        if isinstance(arguments, TypeVar):
            return cls
        elif isinstance(arguments, Table):
            if not hasattr(cls, "foreign_table_name") or not cls.foreign_col_name:
                raise ValueError("A table can only be passed to a ForeignKey that already has a `foreign_table_name`")
            return cast(
                ForeignKey[M],
                type(f"{cls.__name__}{arguments.model_type.__name__.capitalize()}"
                     f"{cls.foreign_col_name.replace('_', '').capitalize()}", (cls,),
                     {
                         "table": arguments
                     })
            )
        else:
            if not isinstance(arguments, tuple) or not (2 <= len(arguments) <= 3) or \
                    not isinstance(arguments[0], str) or not issubclass(arguments[1], Model) or (
                        len(arguments) == 3 and not isinstance(arguments[2], str)
            ):
                raise TypeError(f"Invalid ForeignKey arguments: {list(arguments)!r}. Expected either two or three "
                                "arguments: (1) a string for the foreign table name; (2) the `Model` type for that "
                                "table; and, optionally, (3) the foreign column name. If (3) is omitted, the primary "
                                "key for the foreign table is used.")
            if len(arguments) == 3:
                table_name, model, row_name = arguments
            else:
                table_name, model = arguments
                row_name = model.primary_key_name
        return cast(
            ForeignKey[M],
            type(f"{cls.__name__}{model.__name__.capitalize()}{row_name.replace('_', '').capitalize()}", (cls,), {
                "row_type": model,
                "foreign_col_name": row_name,
                "foreign_table_name": table_name
            })
        )

    @classmethod
    def key_type(cls) -> Type[Union[int, float, str, bytes, Packable]]:
        foreign_type = cls.row_type.FIELDS[cls.foreign_col_name]
        if hasattr(cls, "column_options"):
            options = {"column_options": cls.column_options}
        else:
            options = {}
        return cast(
            Type[Union[int, float, str, bytes, Packable]],
            type(f"{foreign_type.__name__}ForeignKey", (foreign_type,), options)
        )

    @property
    def row(self) -> M:
        if self._row is None:
            if self.table is None:
                raise ValueError(f"{self.__class__.__name__} must have a `table` set")
            foreign_table = getattr(self.table.db, self.foreign_table_name)
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
    def __init__(self, path: str = ":memory:", rollback_on_exception: bool = False):
        self.path: str = path
        self.con = DatabaseConnection(self.path, rollback_on_exception=rollback_on_exception)
        self.tables: Dict[str, Table] = {}
        if self.FIELDS:
            with self:
                for table_name, table_type in self.FIELDS.items():
                    setattr(self, table_name, self.create_table(table_name, table_type))

    @classmethod
    def validate_fields(cls, fields: OrderedDict[str, Type[FieldType]]):
        for field_name, field_type in fields.items():
            if not issubclass(field_type, Table):
                raise TypeError(f"Database {cls!r} table `{field_name}` was expected to be of type `Table` but "
                                f"was instead {field_type!r}")

    def __enter__(self: D) -> D:
        # self.con.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # self.con.__exit__(exc_type, exc_val, exc_tb)
        self.con.commit()

    def create_table(self, table_name: str, table_type: Type[Table[M]]) -> Table[M]:
        columns = []
        table = table_type(self, table_name)
        model_type = table_type.model_type
        for field_name, field_type in model_type.FIELDS.items():
            if issubclass(field_type, RowId):
                continue
            elif issubclass(field_type, ForeignKey):
                old_field_type = field_type
                field_type = field_type.key_type()
                if hasattr(old_field_type, "column_options"):
                    setattr(field_type, "column_options", old_field_type.column_options)
                else:
                    setattr(field_type, "column_options", None)
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
            if hasattr(field_type, "column_options") and field_type.column_options is not None:
                modifiers = field_type.column_options.sql_modifiers()
                if modifiers:
                    modifiers = f" {modifiers}"
            else:
                modifiers = ""
            columns.append(f"{field_name} {data_type}{modifiers}")
        column_constraints = ",\n    ".join(columns)
        if len(columns) > 1:
            column_constraints = "\n    " + column_constraints + "\n"
        with self:
            self.con.execute(f"CREATE TABLE IF NOT EXISTS {table.name} ({column_constraints});")
        self.tables[table_name] = table
        return table
