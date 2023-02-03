"""`transform` provides utilities to generate SQL queries from other queries."""

from __future__ import annotations

import copy
import typing
from typing import Iterable

from postbound.db import db
from postbound.qal import qal, base, clauses

_Q = typing.TypeVar("_Q", bound=qal.SqlQuery)


def implicit_to_explicit(source_query: qal.ImplicitSqlQuery) -> qal.ExplicitSqlQuery:
    pass


def explicit_to_implicit(source_query: qal.ExplicitSqlQuery) -> qal.ImplicitSqlQuery:
    pass


def extract_query_fragment(source_query: qal.SqlQuery,
                           referenced_tables: Iterable[base.TableReference]) -> qal.SqlQuery:
    pass


def as_count_star_query(source_query: qal.SqlQuery) -> qal.SqlQuery:
    count_star = clauses.BaseProjection.count_star()
    target_query = copy.copy(source_query)
    target_query.select_clause = count_star
    return target_query


def rename_table(source_query: qal.SqlQuery, from_table: base.TableReference, target_table: base.TableReference, *,
                 prefix_column_names: bool = False) -> qal.SqlQuery:
    target_query = copy.deepcopy(source_query)

    def _update_column_name(col: base.ColumnReference):
        if col.table == from_table:
            col.table = target_table
        if prefix_column_names and col.table == target_table:
            col.name = f"{from_table.alias}_{col.name}"

    for column in target_query.select_clause.itercolumns():
        _update_column_name(column)

    for table in target_query.tables():
        if table == from_table:
            table.full_name = target_table.full_name
            table.alias = target_table.alias

    if source_query.predicates():
        for column in target_query.predicates().root().itercolumns():
            _update_column_name(column)

    if source_query.groupby_clause:
        for expression in target_query.groupby_clause.group_columns:
            for column in expression.itercolumns():
                _update_column_name(column)

    if source_query.having_clause:
        for column in target_query.having_clause.condition.itercolumns():
            _update_column_name(column)

    if source_query.orderby_clause:
        for expression in target_query.orderby_clause.expressions:
            for column in expression.column.itercolumns():
                _update_column_name(column)

    return target_query


def bind_columns(query: qal.SqlQuery, *, with_schema: bool = True, db_schema: db.DatabaseSchema | None = None) -> None:
    """Queries the table metadata to obtain additional information about the referenced columns.

    The retrieved information includes type information for all columns and the tables that contain the columns.
    """
    alias_map = {table.alias: table for table in query.tables() if table.alias}
    unbound_tables = [table for table in query.tables() if not table.alias]
    unbound_columns = []

    def _update_column_binding(col: base.ColumnReference) -> None:
        if not col.table:
            unbound_columns.append(col)
        elif not col.table.full_name and col.table.alias in alias_map:
            col.table.full_name = alias_map[col.table.alias].full_name
        elif col.table and not col.table.full_name:
            col.table.full_name = col.table.alias
            col.table.alias = ""

    for column in query.select_clause.itercolumns():
        _update_column_binding(column)

    if query.predicates():
        for column in query.predicates().root().itercolumns():
            _update_column_binding(column)

    column_output_names = query.select_clause.output_names()

    if query.groupby_clause:
        for expression in query.groupby_clause.group_columns:
            for column in expression.itercolumns():
                _update_column_binding(column)
                if column.name in column_output_names:
                    column.redirect = column_output_names[column.name]

    if query.having_clause:
        for column in query.having_clause.condition.itercolumns():
            _update_column_binding(column)
            if column.name in column_output_names:
                column.redirect = column_output_names[column.name]

    if query.orderby_clause:
        for expression in query.orderby_clause.expressions:
            for column in expression.column.itercolumns():
                _update_column_binding(column)
                if column.name in column_output_names:
                    column.redirect = column_output_names[column.name]

    if with_schema:
        db_schema = db_schema if db_schema else db.DatabasePool.get_instance().current_database()
        for column in unbound_columns:
            column.table = db_schema.lookup_column(column, unbound_tables)
