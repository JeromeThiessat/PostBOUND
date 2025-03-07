"""Models all supported SQL expressions.

Predicates and clauses that are build on top of these expressions are located in separate modules.

See the package description for more details on how these concepts are related. The`SqlExpression` provides a
high-level introduction into the structure of different expressions.
"""
from __future__ import annotations

import abc
import enum
import numbers
import typing
from collections.abc import Iterable, Sequence
from typing import Union, Optional

from postbound.qal import base, qal
from postbound.util import collections as collection_utils

T = typing.TypeVar("T")
"""Typed expressions use this generic type variable."""


class MathematicalSqlOperators(enum.Enum):
    """The supported mathematical operators."""
    Add = "+"
    Subtract = "-"
    Multiply = "*"
    Divide = "/"
    Modulo = "%"
    Negate = "-"


class LogicalSqlOperators(enum.Enum):
    """The supported unary and binary operators.

    Notice that the predicates which make heavy use of these operators are specified in the `predicates` module.
    """
    Equal = "="
    NotEqual = "<>"
    Less = "<"
    LessEqual = "<="
    Greater = ">"
    GreaterEqual = ">="
    Like = "LIKE"
    NotLike = "NOT LIKE"
    ILike = "ILIKE"
    NotILike = "NOT ILIKE"
    In = "IN"
    Exists = "IS NULL"
    Missing = "IS NOT NULL"
    Between = "BETWEEN"


UnarySqlOperators: frozenset[LogicalSqlOperators] = frozenset({LogicalSqlOperators.Exists, LogicalSqlOperators.Missing})
"""The `LogicalSqlOperators` that can be used as unary operators."""


class LogicalSqlCompoundOperators(enum.Enum):
    """The supported compound operators.

    Notice that predicates which make heavy use of these operators are specified in the `predicates` module.
    """
    And = "AND"
    Or = "OR"
    Not = "NOT"


SqlOperator = Union[MathematicalSqlOperators, LogicalSqlOperators, LogicalSqlCompoundOperators]
"""Captures all different kinds of operators in one type."""


class SqlExpression(abc.ABC):
    """Base class for all expressions.

    Expressions form one of the central building blocks of representing a SQL query in the QAL. They specify how values
    from different columns are modified and combined, thereby forming larger (hierarchical) structures.

    Expressions can be inserted in many different places in a SQL query. For example, a ``SELECT`` clause produces
    columns such as in ``SELECT R.a FROM R``, but it can also modify the column values slightly, such as in
    ``SELECT R.a + 42 FROM R``. To account for all  these different situations, the `SqlExpression` is intended to form
    hierarchical trees and chains of expressions. In the first case, a `ColumnExpression` is used, whereas a
    `MathematicalExpression` can model the second case. Whereas column expressions represent leaves in the expression
    tree, mathematical expressions are intermediate nodes.

    As a more advanced example, a complicated expressions such as `my_udf(R.a::interval + 42)` which consists of a
    user-defined function, a value cast and a mathematical operation is represented the following way:
    `FunctionExpression(MathematicalExpression(CastExpression(ColumnExpression), StaticValueExpression))`. The methods
    provided by all expression instances enable a more convenient use and access to the expression hierarchies.

    The different kinds of expressions are represented using different subclasses of the `SqlExpression` interface.
    This really is an abstract interface, not a usable expression. All inheriting expression have to provide their own
    `__eq__` method and re-use the `__hash__` method provided by the base expression. Remember to explicitly set this
    up! The concrete hash value is constant since the clause itself is immutable. It is up to the implementing class to
    make sure that the equality/hash consistency is enforced.

    Parameters
    ----------
    hash_val : int
        The hash of the concrete expression object
    """

    def __init__(self, hash_val: int):
        self._hash_val = hash_val

    def tables(self) -> set[base.TableReference]:
        """Provides all tables that are accessed by this expression.

        Returns
        -------
        set[base.TableReference]
            All tables. This includes virtual tables if such tables are present in the expression.
        """
        return {column.table for column in self.columns() if column.is_bound()}

    @abc.abstractmethod
    def columns(self) -> set[base.ColumnReference]:
        """Provides all columns that are referenced by this expression.

        Returns
        -------
        set[base.ColumnReference]
            The columns
        """
        raise NotImplementedError

    @abc.abstractmethod
    def itercolumns(self) -> Iterable[base.ColumnReference]:
        """Provides all columns that are referenced by this expression.

        If a column is referenced multiple times, it is also returned multiple times.

        Returns
        -------
        Iterable[base.ColumnReference]
            All columns in exactly the order in which they are used.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def iterchildren(self) -> Iterable[SqlExpression]:
        """Provides unified access to all child expressions of the concrete expression type.

        For *leaf* expressions such as static values, the iterable will not contain any elements. Otherwise, all
        *direct* children will be returned. For example, a mathematical expression could return both the left, as well
        as the right operand. This allows for access to nested expressions in a recursive manner.

        Returns
        -------
        Iterable[SqlExpression]
            The expressions
        """
        raise NotImplementedError

    def __hash__(self) -> int:
        return self._hash_val

    @abc.abstractmethod
    def __eq__(self, other) -> bool:
        raise NotImplementedError

    def __repr__(self) -> str:
        return str(self)

    @abc.abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError


class StaticValueExpression(SqlExpression, typing.Generic[T]):
    """An expression that wraps a literal/static value.

    This is one of the leaf expressions that does not contain any further child expressions.

    Parameters
    ----------
    value : T
        The value that is wrapped by the expression

    Examples
    --------
    Consider the following SQL query: ``SELECT * FROM R WHERE R.a = 42``. In this case the comparison value of 42 will
    be represented as a static value expression. The reference to the column ``R.a`` cannot be a static value since its
    values depend on the actual column values. Hence, a `ColumnExpression` is used for it.
    """

    def __init__(self, value: T) -> None:
        self._value = value
        super().__init__(hash(value))

    @property
    def value(self) -> T:
        """Get the value.

        Returns
        -------
        T
            The value, duh!
        """
        return self._value

    def columns(self) -> set[base.ColumnReference]:
        return set()

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return []

    def iterchildren(self) -> Iterable[SqlExpression]:
        return []

    __hash__ = SqlExpression.__hash__

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.value == other.value

    def __str__(self) -> str:
        return f"{self.value}" if isinstance(self.value, numbers.Number) else f"'{self.value}'"


class CastExpression(SqlExpression):
    """An expression that casts the type of another nested expression.

    Note that PostBOUND itself does not know about the semantics of the actual types or casts. Eventual errors due to
    illegal casts are only caught at runtime by the actual database system.

    Parameters
    ----------
    expression : SqlExpression
        The expression that is casted to a different type.
    target_type : str
        The type to which the expression should be converted to. This cannot be empty.

    Raises
    ------
    ValueError
        If the `target_type` is empty.
    """

    def __init__(self, expression: SqlExpression, target_type: str) -> None:
        if not expression or not target_type:
            raise ValueError("Expression and target type are required")
        self._casted_expression = expression
        self._target_type = target_type

        hash_val = hash((self._casted_expression, self._target_type))
        super().__init__(hash_val)

    @property
    def casted_expression(self) -> SqlExpression:
        """Get the expression that is being casted.

        Returns
        -------
        SqlExpression
            The expression
        """
        return self._casted_expression

    @property
    def target_type(self) -> str:
        """Get the type to which to cast to.

        Returns
        -------
        str
            The desired type. This is never empty.
        """
        return self._target_type

    def columns(self) -> set[base.ColumnReference]:
        return self.casted_expression.columns()

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return self.casted_expression.itercolumns()

    def iterchildren(self) -> Iterable[SqlExpression]:
        return [self.casted_expression]

    __hash__ = SqlExpression.__hash__

    def __eq__(self, other) -> bool:
        return (isinstance(other, type(self))
                and self.casted_expression == other.casted_expression
                and self.target_type == other.target_type)

    def __str__(self) -> str:
        return f"CAST({self.casted_expression} AS {self.target_type})"


class MathematicalExpression(SqlExpression):
    """A mathematical expression computes a result value based on a mathematical formula.

    The formula is based on an arbitrary expression, an operator and potentially a number of additional
    expressions/arguments.

    The precise representation of mathematical expressions is not tightly standardized by PostBOUND and there will be
    multiple ways to represent the same expression.

    For example, the expression ``R.a + S.b + 42`` could be modeled as a single expression object with ``R.a`` as first
    argument and the sequence ``S.b, 42`` as second arguments. At the same time, the mathematical expression can also
    be used to represent logical expressions such as ``R.a < 42`` or ``S.b IN (1, 2, 3)``. However, this should be used
    sparingly since logical expressions can be considered as predicates which are handled in the dedicated `predicates`
    module. Moving logical expressions into a mathematical expression object can break correct functionality in that
    module (e.g. determining joins and filters in a query).

    Parameters
    ----------
    operator : SqlOperator
        The operator that is used to combine the arguments.
    first_argument : SqlExpression
        The first argument. For unary expressions, this can also be the only argument
    second_argument : SqlExpression | Sequence[SqlExpression] | None, optional
        Additional arguments. For the most common case of a binary expression, this will be exactly one argument.
        Defaults to ``None`` to accomodate for unary expressions.
    """

    def __init__(self, operator: SqlOperator, first_argument: SqlExpression,
                 second_argument: SqlExpression | Sequence[SqlExpression] | None = None) -> None:
        if not operator or not first_argument:
            raise ValueError("Operator and first argument are required!")
        self._operator = operator
        self._first_arg = first_argument
        self._second_arg: SqlExpression | tuple[SqlExpression] | None = (tuple(second_argument)
                                                                         if isinstance(second_argument, Sequence)
                                                                         else second_argument)

        if isinstance(self._second_arg, tuple) and len(self._second_arg) == 1:
            self._second_arg = self._second_arg[0]

        hash_val = hash((self._operator, self._first_arg, self._second_arg))
        super().__init__(hash_val)

    @property
    def operator(self) -> SqlOperator:
        """Get the operation to combine the input value(s).

        Returns
        -------
        SqlOperator
            The operator
        """
        return self._operator

    @property
    def first_arg(self) -> SqlExpression:
        """Get the first argument to the operator. This is always specified.

        Returns
        -------
        SqlExpression
            The argument
        """
        return self._first_arg

    @property
    def second_arg(self) -> SqlExpression | Sequence[SqlExpression] | None:
        """Get the second argument to the operator.

        Depending on the operator, this can be a single expression (the most common case), but also a sequence of
        expressions (e.g. sum of multiple values) or no value at all (e.g. negation).

        Returns
        -------
        SqlExpression | Sequence[SqlExpression] | None
            The argument(s)
        """
        return self._second_arg

    def columns(self) -> set[base.ColumnReference]:
        all_columns = set(self.first_arg.columns())
        if isinstance(self.second_arg, list):
            for expression in self.second_arg:
                all_columns |= expression.columns()
        elif isinstance(self.second_arg, SqlExpression):
            all_columns |= self.second_arg.columns()
        return all_columns

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        first_columns = list(self.first_arg.itercolumns())
        if not self.second_arg:
            return first_columns
        second_columns = (collection_utils.flatten(sub_arg.itercolumns() for sub_arg in self.second_arg)
                          if isinstance(self.second_arg, tuple) else list(self.second_arg.itercolumns()))
        return first_columns + second_columns

    def iterchildren(self) -> Iterable[SqlExpression]:
        return [self.first_arg, self.second_arg]

    __hash__ = SqlExpression.__hash__

    def __eq__(self, other) -> bool:
        return (isinstance(other, type(self))
                and self.operator == other.operator
                and self.first_arg == other.first_arg
                and self.second_arg == other.second_arg)

    def __str__(self) -> str:
        operator_str = self.operator.value
        if self.operator == MathematicalSqlOperators.Negate:
            return f"{operator_str}{self.first_arg}"
        if isinstance(self.second_arg, tuple):
            all_args = [self.first_arg] + list(self.second_arg)
            return operator_str.join(str(arg) for arg in all_args)
        return f"{self.first_arg} {operator_str} {self.second_arg}"


class ColumnExpression(SqlExpression):
    """A column expression wraps the reference to a column.

    This is a leaf expression, i.e. a column expression cannot have any more child expressions. It corresponds directly
    to an access to the values of the wrapped column with no modifications.

    Parameters
    ----------
    column : base.ColumnReference
        The column being wrapped
    """

    def __init__(self, column: base.ColumnReference) -> None:
        if column is None:
            raise ValueError("Column cannot be none")
        self._column = column
        super().__init__(hash(self._column))

    @property
    def column(self) -> base.ColumnReference:
        """Get the column that is wrapped by this expression.

        Returns
        -------
        base.ColumnReference
            The column
        """
        return self._column

    def columns(self) -> set[base.ColumnReference]:
        return {self.column}

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return [self.column]

    def iterchildren(self) -> Iterable[SqlExpression]:
        return []

    __hash__ = SqlExpression.__hash__

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.column == other.column

    def __str__(self) -> str:
        return str(self.column)


AggregateFunctions = {"COUNT", "SUM", "MIN", "MAX", "AVG"}
"""All aggregate functions specified in standard SQL."""


class FunctionExpression(SqlExpression):
    """The function expression indicates a call to an arbitrary function.

    The actual function might be one of the standard SQL functions, an aggregation function or a user-defined one.
    PostBOUND treats them all the same and it is up to the user to differentiate e.g. between UDFs and aggregations if
    this distinction is important. This can be easily achieved by introducing additional subclasses of the function
    expression and updating the queries to use the new function expressions where appropriate. The `transform` module
    provides utilities to make such updates easy.

    Parameters
    ----------
    function : str
        The name of the function that should be called. Cannot be empty.
    arguments : Optional[Sequence[SqlExpression]], optional
        The parameters that should be passed to the function. Can be ``None`` if the function does not take or does not
        need any arguments (e.g. ``CURRENT_TIME()``)
    distinct : bool, optional
        Whether the (aggregation) function should only operate on distinct column values and hence a duplicate
        elimination needs to be performed before passing the argument values (e.g. ``COUNT(DISTINCT *)``). Defaults to
        ``False``

    Raises
    ------
    ValueError
        If `function` is empty

    See Also
    --------
    postbound.qal.transform.replace_expressions
    """
    def __init__(self, function: str, arguments: Optional[Sequence[SqlExpression]] = None, *,
                 distinct: bool = False) -> None:
        if not function:
            raise ValueError("Function is required")
        self._function = function.upper()
        self._arguments: tuple[SqlExpression] = () if arguments is None else tuple(arguments)
        self._distinct = distinct

        hash_val = hash((self._function, self._distinct, self._arguments))
        super().__init__(hash_val)

    @property
    def function(self) -> str:
        """Get the function name.

        Returns
        -------
        str
            The function name. Will never be empty
        """
        return self._function

    @property
    def arguments(self) -> Sequence[SqlExpression]:
        """Get all arguments that are supplied to the function.

        Returns
        -------
        Sequence[SqlExpression]
            The arguments. Can be empty if no arguments are passed (but will never be ``None``).
        """
        return self._arguments

    @property
    def distinct(self) -> bool:
        """Get whether the function should only operate on distinct values.

        Whether this makes any sense for the function at hand is entirely dependend on the specific function and not
        enfored by PostBOUND. The runtime DBS has to check this.

        Generally speaking, this argument is intended for aggregation functions.

        Returns
        -------
        bool
            Whether a duplicate elimination has to be performed on the function arguments
        """
        return self._distinct

    def is_aggregate(self) -> bool:
        """Checks, whether the function is a well-known SQL aggregation function.

        Only standard functions are considered (e.g. no CORR for computing correlations).

        Returns
        -------
        bool
            Whether the function is a known aggregate function.
        """
        return self._function.upper() in AggregateFunctions

    def columns(self) -> set[base.ColumnReference]:
        all_columns = set()
        for arg in self.arguments:
            all_columns |= arg.columns()
        return all_columns

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        all_columns = []
        for arg in self.arguments:
            all_columns.extend(arg.itercolumns())
        return all_columns

    def iterchildren(self) -> Iterable[SqlExpression]:
        return list(self.arguments)

    __hash__ = SqlExpression.__hash__

    def __eq__(self, other) -> bool:
        return (isinstance(other, type(self))
                and self.function == other.function
                and self.arguments == other.arguments
                and self.distinct == other.distinct)

    def __str__(self) -> str:
        args_str = ", ".join(str(arg) for arg in self._arguments)
        distinct_str = "DISTINCT " if self._distinct else ""
        return f"{self._function}({distinct_str}{args_str})"


class SubqueryExpression(SqlExpression):
    """A subquery expression wraps an arbitrary subquery.

    This expression can be used in two different contexts: as table source to produce a virtual temporary table for
    reference in the query (see the `clauses` module), or as a part of a predicate. In the latter scenario the
    subqueries' results are transient for the rest of the query. Therefore, this expression only represents the
    subquery part but no name under which the query result can be accessed. This is added by the different parts of the
    `clauses` module (e.g. `WithQuery` or `SubqueryTableSource`).

    This is a leaf expression, i.e. a subquery expression cannot have any more child expressions. However, the subquery itself
    likely consists of additional expressions.

    Parameters
    ----------
    subquery : qal.SqlQuery
        The subquery that forms this expression

    """

    def __init__(self, subquery: qal.SqlQuery) -> None:
        self._query = subquery
        super().__init__(hash(subquery))

    @property
    def query(self) -> qal.SqlQuery:
        """The (sub)query that is wrapped by this expression.

        Returns
        -------
        qal.SqlQuery
            The query
        """
        return self._query

    def columns(self) -> set[base.ColumnReference]:
        return self._query.columns()

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return self._query.itercolumns()

    def iterchildren(self) -> Iterable[SqlExpression]:
        return []

    def tables(self) -> set[base.TableReference]:
        return self._query.tables()

    __hash__ = SqlExpression.__hash__

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self._query == other._query

    def __str__(self) -> str:
        query_str = str(self._query).removesuffix(";")
        return f"({query_str})"


class StarExpression(SqlExpression):
    """A special expression that is only used in ``SELECT`` clauses to select all columns."""

    def __init__(self) -> None:
        super().__init__(hash("*"))

    def columns(self) -> set[base.ColumnReference]:
        return set()

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return []

    def iterchildren(self) -> Iterable[SqlExpression]:
        return []

    __hash__ = SqlExpression.__hash__

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self))

    def __str__(self) -> str:
        return "*"


def as_expression(value: object) -> SqlExpression:
    """Transforms the given value into the most appropriate `SqlExpression` instance.

    This is a heuristic utility method that applies the following rules:

    - `ColumnReference` becomes `ColumnExpression`
    - `SqlQuery` becomes `SubqueryExpression`
    - the star-string ``*`` becomes a `StarExpression`

    All other values become a `StaticValueExpression`.

    Parameters
    ----------
    value : object
        The object to be transformed into an expression

    Returns
    -------
    SqlExpression
        The most appropriate expression object according to the transformation rules
    """
    if isinstance(value, SqlExpression):
        return value

    if isinstance(value, base.ColumnReference):
        return ColumnExpression(value)
    elif isinstance(value, qal.SqlQuery):
        return SubqueryExpression(value)

    if value == "*":
        return StarExpression()
    return StaticValueExpression(value)
