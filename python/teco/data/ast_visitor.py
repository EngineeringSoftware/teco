from typing import Generic, TypeVar

from teco.data.structures import AST

TContext = TypeVar("TContext")
TReturn = TypeVar("TReturn")


class Visitor(Generic[TContext, TReturn]):
    """
    Base visitor class for AST.
    Since we save AST's type information as string rather than using different node type, the way to modify the visiting behavior for a node type is to define a new method called "visit_<node_type>" in the derived class.
    """

    def visit(self, node: AST, context: TContext = None) -> TReturn:
        method = "visit_" + node.ast_type
        if hasattr(self, method):
            return getattr(self, method)(node, context)
        else:
            return self.default_visit(node, context)

    def default_visit(self, node: AST, context: TContext = None) -> TReturn:
        if node.children is not None:
            return [self.visit(child, context) for child in node.children]
        else:
            return None
