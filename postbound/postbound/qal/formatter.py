"""Provides logic to generate pretty strings for query objects."""
from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Optional

from postbound.qal import qal, clauses, expressions as expr, predicates as preds, transform
from postbound.util import errors

FormatIndentDepth = 2
"""The default amount of whitespace that is used to indent specific parts of the SQL query."""


def _increase_indentation(content: str, indentation: int = FormatIndentDepth) -> str:
    """Prefixes all lines in a string by a given amount of whitespace.

    This breaks the input string into separate lines and adds whitespace equal to the desired amount at the start of
    each line.

    Parameters
    ----------
    content : str
        The string that should be prefixed/whose indentation should be increased.
    indentation : Optional[int], optional
        The amount of whitespace that should be added, by default `FormatIndentDepth`

    Returns
    -------
    str
        The indented string
    """
    indent_prefix = "\n" + " " * indentation
    return " " * indentation + indent_prefix.join(content.split("\n"))


class FormattingSubqueryExpression(expr.SubqueryExpression):
    """Wraps subquery expressions to ensure that they are also pretty-printed and aligned properly.

    This class acts as a decorator around the actual subquery. It can be used entirely as a replacement of the original
    query.

    Parameters
    ----------
    original_expression : expr.SubqueryExpression
        The actual subquery.
    inline_hint_block : bool
        Whether potential hint blocks of the subquery should be printed as preceding blocks or as inline blocks (see
        `format_quick` for details)
    indentation : int
        The current amount of indentation that should be used for the subquery. While pretty-printing, additional
        indentation levels can be inserted for specific parts of the query.
    """
    def __init__(self, original_expression: expr.SubqueryExpression, inline_hint_block: bool,
                 indentation: int) -> None:
        super().__init__(original_expression.query)
        self._inline_hint_block = inline_hint_block
        self._indentation = indentation

    def __str__(self) -> str:
        formatted = format_quick(self.query, inline_hint_block=self._inline_hint_block)
        prefix = " " * self._indentation
        if "\n" not in formatted:
            return prefix + formatted

        indented_lines = [""] + [prefix + line for line in formatted.split("\n")] + [""]
        return "\n".join(indented_lines)


class FormattingLimitClause(clauses.Limit):
    """Wraps the `Limit` clause to enable pretty printing of its different parts (limit and offset).

    This class acts as a decorator around the actual clause. It can be used entirely as a replacement of the original
    clause.

    Parameters
    ----------
    original_clause : clauses.Limit
        The clause to wrap
    """

    def __init__(self, original_clause: clauses.Limit) -> None:
        super().__init__(limit=original_clause.limit, offset=original_clause.offset)

    def __str__(self) -> str:
        if self.offset and self.limit:
            return f"OFFSET {self.offset} ROWS\nFETCH FIRST {self.limit} ROWS ONLY"
        elif self.offset:
            return f"OFFSET {self.offset} ROWS"
        elif self.limit:
            return f"FETCH FIRST {self.limit} ROWS ONLY"
        raise errors.InvariantViolationError("Either limit or offset must be specified for Limit clause")


def _quick_format_cte(cte_clause: clauses.CommonTableExpression) -> list[str]:
    """Formatting logic for Common Table Expressions

    Parameters
    ----------
    cte_clause : clauses.CommonTableExpression
        The clause to format

    Returns
    -------
    list[str]
        The pretty-printed parts of the CTE, indented as necessary.
    """
    if len(cte_clause.queries) == 1:
        cte_query = cte_clause.queries[0]
        cte_header = f"WITH {cte_query.target_name} AS ("
        cte_content = _increase_indentation(format_quick(cte_query.query).removesuffix(";"))
        cte_footer = ")"
        return [cte_header, cte_content, cte_footer]

    first_cte, *remaining_ctes = cte_clause.queries
    first_content = _increase_indentation(format_quick(first_cte.query)).removesuffix(";")
    formatted_parts: list[str] = [f"WITH {first_cte.target_name} AS (", first_content]
    for next_cte in remaining_ctes:
        current_header = f"), {next_cte.target_name} AS ("
        cte_content = _increase_indentation(format_quick(next_cte.query).removesuffix(";"))

        formatted_parts.append(current_header)
        formatted_parts.append(cte_content)

    formatted_parts.append(")")
    return formatted_parts


def _quick_format_select(select_clause: clauses.Select, *,
                         inlined_hint_block: Optional[clauses.Hint] = None) -> list[str]:
    """Quick and dirty formatting logic for ``SELECT`` clauses.

    Up to 3 targets are put on the same line, otherwise each target is put on a separate line.

    Parameters
    ----------
    select_clause : clauses.Select
        The clause to format
    inlined_hint_block : Optional[clauses.Hint], optional
        A hint block that should be inserted after the ``SELECT`` statement. Defaults to ``None`` which indicates that
        no block should be inserted that way

    Returns
    -------
    list[str]
        The pretty-printed parts of the clause, indented as necessary.
    """
    hint_text = f"{inlined_hint_block} " if inlined_hint_block else ""
    if len(select_clause.targets) > 3:
        first_target, *remaining_targets = select_clause.targets
        formatted_targets = [f"SELECT {hint_text}{first_target}"
                             if select_clause.projection_type == clauses.SelectType.Select
                             else f"SELECT DISTINCT {hint_text}{first_target}"]
        formatted_targets += [((" " * FormatIndentDepth) + str(target)) for target in remaining_targets]
        for i in range(len(formatted_targets) - 1):
            formatted_targets[i] += ","
        return formatted_targets
    else:
        distinct_text = "DISTINCT " if select_clause.projection_type == clauses.SelectType.SelectDistinct else ""
        targets_text = ", ".join(str(target) for target in select_clause.targets)
        return [f"SELECT {distinct_text}{hint_text}{targets_text}"]


def _quick_format_implicit_from(from_clause: clauses.ImplicitFromClause) -> list[str]:
    """Quick and dirty formatting logic for implicit ``FROM`` clauses.

    Up to 3 tables are put on the same line, otherwise each table is put on its own line.

    Parameters
    ----------
    from_clause : clauses.ImplicitFromClause
        The clause to format

    Returns
    -------
    list[str]
        The pretty-printed parts of the clause, indented as necessary.
    """
    tables = list(from_clause.itertables())
    if not tables:
        return []
    elif len(tables) > 3:
        first_table, *remaining_tables = tables
        formatted_tables = [f"FROM {first_table}"]
        formatted_tables += [((" " * FormatIndentDepth) + str(tab)) for tab in remaining_tables]
        for i in range(len(formatted_tables) - 1):
            formatted_tables[i] += ","
        return formatted_tables
    else:
        tables_str = ", ".join(str(tab) for tab in tables)
        return [f"FROM {tables_str}"]


def _quick_format_explicit_from(from_clause: clauses.ExplicitFromClause) -> list[str]:
    """Quick and dirty formatting logic for explicit ``FROM`` clauses.

    This function just puts each ``JOIN ON`` statement on a separate line.

    Parameters
    ----------
    from_clause : clauses.ExplicitFromClause
        The clause to format

    Returns
    -------
    list[str]
        The pretty-printed parts of the clause, indented as necessary.
    """
    pretty_base = [f"FROM {from_clause.base_table}"]
    pretty_joins = [((" " * FormatIndentDepth) + str(join)) for join in from_clause.joined_tables]
    return pretty_base + pretty_joins


def _quick_format_predicate(predicate: preds.AbstractPredicate) -> list[str]:
    """Quick and dirty formatting logic for arbitrary (i.e. also compound) predicates.

    ``AND`` conditions are put on separate lines, everything else is put on one line.

    Parameters
    ----------
    predicate : preds.AbstractPredicate
        The predicate to format

    Returns
    -------
    list[str]
        The pretty-printed parts of the predicate, indented as necessary.
    """
    if not isinstance(predicate, preds.CompoundPredicate):
        return [str(predicate)]
    compound_pred: preds.CompoundPredicate = predicate
    if compound_pred.operation == expr.LogicalSqlCompoundOperators.And:
        first_child, *remaining_children = compound_pred.children
        return [str(first_child)] + ["AND " + str(child) for child in remaining_children]
    return [str(compound_pred)]


def _quick_format_where(where_clause: clauses.Where) -> list[str]:
    """Quick and dirty formatting logic for ``WHERE`` clauses.

    This function just puts each part of an ``AND`` condition on a separate line and leaves the parts of ``OR``
    conditions, negations or base predicates on the same line.

    Parameters
    ----------
    where_clause : clauses.Where
        The clause to format

    Returns
    -------
    list[str]
        The pretty-printed parts of the clause, indented as necessary.
    """
    first_pred, *additional_preds = _quick_format_predicate(where_clause.predicate)
    return [f"WHERE {first_pred}"] + [((" " * FormatIndentDepth) + str(pred)) for pred in additional_preds]


def _quick_format_limit(limit_clause: clauses.Limit) -> list[str]:
    """Quick and dirty formatting logic for ``FETCH FIRST`` / ``LIMIT`` clauses.

    This produces output that is equivalent to the SQL standard's syntax to denote limit clauses and splits the limit
    and offset parts onto separate lines.

    Parameters
    ----------
    limit_clause : clauses.Limit
        The clause to format

    Returns
    -------
    list[str]
        The pretty-printed parts of the clause, indented as necessary.
    """
    pass


def _subquery_replacement(expression: expr.SqlExpression, *, inline_hints: bool,
                          indentation: int) -> expr.SqlExpression:
    """Handler method for `transform.replace_expressions` to apply our custom `FormattingSubqueryExpression`.

    Parameters
    ----------
    expression : expr.SqlExpression
        The expression to replace.
    inline_hints : bool
        Whether potential hint blocks should be inserted as part of the ``SELECT`` clause rather than before the
        actual query.
    indentation : int
        The amount of indentation to use for the subquery

    Returns
    -------
    expr.SqlExpression
        The original SQL expression if the `expression` is not a `SubqueryExpression`. Otherwise, the expression is
        wrapped in a `FormattingSubqueryExpression`.
    """
    if not isinstance(expression, expr.SubqueryExpression):
        return expression
    return FormattingSubqueryExpression(expression, inline_hints, indentation)


def format_quick(query: qal.SqlQuery, *, inline_hint_block: bool = False,
                 custom_formatter: Optional[Callable[[qal.SqlQuery], qal.SqlQuery]] = None) -> str:
    """Applies a quick formatting heuristic to structure the given query.

    The query will be structured as follows:

    - all clauses start at a new line
    - long clauses with multiple parts (e.g. ``SELECT`` clause, ``FROM`` clause) are split along multiple intended
      lines
    - the predicate in the ``WHERE`` clause is split on multiple lines along the different parts of a conjunctive
      predicate

    All other clauses are written on a single line (e.g. ``GROUP BY`` clause).

    Parameters
    ----------
    query : qal.SqlQuery
        The query to format
    inline_hint_block : bool, optional
        Whether to insert a potential hint block in the ``SELECT`` clause (i.e. *inline* it), or leave it as a
        block preceding the actual query. Defaults to ``False`` which indicates that the clause should be printed
        before the actual query.
    custom_formatter : Callable[[qal.SqlQuery], qal.SqlQuery], optional
        A post-processing formatting service to apply to the SQL query after all preparatory steps have been performed,
        but *before* the actual formatting is started. This can be used to inject custom clause or expression
        formatting rules that are necessary to adhere to specific SQL syntax deviations for a database system. Defaults
        to ``None`` which skips this step.

    Returns
    -------
    str
        A pretty string representation of the query.
    """
    pretty_query_parts = []
    inlined_hint_block = None
    subquery_update = functools.partial(_subquery_replacement, inline_hints=inline_hint_block,
                                        indentation=FormatIndentDepth)
    query = transform.replace_expressions(query, subquery_update)
    if query.limit_clause is not None:
        query = transform.replace_clause(query, FormattingLimitClause(query.limit_clause))

    if custom_formatter is not None:
        query = custom_formatter(query)

    for clause in query.clauses():
        if inline_hint_block and isinstance(clause, clauses.Hint):
            inlined_hint_block = clause
            continue

        if isinstance(clause, clauses.CommonTableExpression):
            pretty_query_parts.extend(_quick_format_cte(clause))
        elif isinstance(clause, clauses.Select):
            pretty_query_parts.extend(_quick_format_select(clause, inlined_hint_block=inlined_hint_block))
        elif isinstance(clause, clauses.ImplicitFromClause):
            pretty_query_parts.extend(_quick_format_implicit_from(clause))
        elif isinstance(clause, clauses.ExplicitFromClause):
            pretty_query_parts.extend(_quick_format_explicit_from(clause))
        elif isinstance(clause, clauses.Where):
            pretty_query_parts.extend(_quick_format_where(clause))
        else:
            pretty_query_parts.append(str(clause))

    pretty_query_parts[-1] += ";"
    return "\n".join(pretty_query_parts)
