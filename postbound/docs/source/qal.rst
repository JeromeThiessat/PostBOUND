Query abstraction layer
=======================

Lots of PostBOUND's functionality needs to access information about SQL queries or modify them. Therefore, PostBOUND introduces
a standardized way of representing queries and information about them. This is done in the so-called *query abstraction layer*
or *qal* for short. The qal provides models for expressions, clauses and entire queries that cover a large variety of SQL
queries. Notice that PostBOUND focuses on read-only SELECT queries. Therefore, no INSERT or UPDATE queries can be represented
in the qal.

The entire abstraction is designed around immutable read-only objects. Therefore, queries cannot be easily modified and new
query instances have to be created instead. To make this process easier, the `transform` module provides a number utilities
that cover the most common modification cases. The immutability was introduced for two reasons: at the one hand immutable
objects enable efficient calculation and caching of hash values. This is especially useful since many access patterns for
specific parts of the input queries (such as join predicates) are part of hot loops in many optimization scenarios. On the
other hand, immutable objects prevent suprising behaviour if queries are modified by other code, thereby eliminating a huge
class of potential bugs by design.

Other than the representation and transformation tools, the qal also provides a parser to create query instances from plain
text. See the `postbound.qal.parser` module for details. Finally, the `formatter` module adds some functionality to
pretty-print queries based on a couple of heuristics.
