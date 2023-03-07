package org.teco.joint;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import com.github.javaparser.JavaToken;
import com.github.javaparser.Position;
import com.github.javaparser.Range;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.comments.BlockComment;
import com.github.javaparser.ast.comments.LineComment;
import com.github.javaparser.ast.stmt.AssertStmt;
import com.github.javaparser.ast.stmt.BreakStmt;
import com.github.javaparser.ast.stmt.ContinueStmt;
import com.github.javaparser.ast.stmt.DoStmt;
import com.github.javaparser.ast.stmt.ExplicitConstructorInvocationStmt;
import com.github.javaparser.ast.stmt.ExpressionStmt;
import com.github.javaparser.ast.stmt.ForEachStmt;
import com.github.javaparser.ast.stmt.ForStmt;
import com.github.javaparser.ast.stmt.IfStmt;
import com.github.javaparser.ast.stmt.LocalClassDeclarationStmt;
import com.github.javaparser.ast.stmt.ReturnStmt;
import com.github.javaparser.ast.stmt.Statement;
import com.github.javaparser.ast.stmt.SwitchStmt;
import com.github.javaparser.ast.stmt.SynchronizedStmt;
import com.github.javaparser.ast.stmt.ThrowStmt;
import com.github.javaparser.ast.stmt.TryStmt;
import com.github.javaparser.ast.stmt.WhileStmt;
import com.github.javaparser.ast.visitor.GenericVisitorWithDefaults;

public class ExtractASTVisitor extends GenericVisitorWithDefaults<AST, ExtractASTVisitor.Context> {

    public static final String TERMINAL = "Terminal";

    public static class Context {
        boolean subStmtLevel = false;
    }

    @Override
    public AST defaultAction(Node n, Context ctx) {
        AST ast = new AST();
        ast.astType = n.getClass().getSimpleName();
        ast.setLineno(n.getRange().get().begin.line, n.getRange().get().end.line);

        // collect contained tokens
        List<JavaToken> javaTokens = new LinkedList<>();
        n.getTokenRange().orElseThrow(RuntimeException::new).forEach(javaTokens::add);

        // collect child nodes
        Map<Range, Node> range2ChildNode = new HashMap<>();
        for (Node c : n.getChildNodes()) {
            // skip the child nodes that do not map to any tokens, e.g., UnknownType nodes in lambda expressions
            if (c.getRange().isPresent()) {
                range2ChildNode.put(c.getRange().get(), c);
            }
        }

        if (javaTokens.size() == 0) {
            // should not happen
            return null;
        }

        if (javaTokens.size() == 1 && range2ChildNode.isEmpty()) {
            // node without children: treat this node as terminal node
            JavaToken t = javaTokens.get(0);
            ast.tok = t.getText();
            ast.tokKind = t.getCategory().toString();
        } else {
            // node with children
            ast.children = new LinkedList<>();

            Position curPos = null;
            for (JavaToken t : javaTokens) {
                Range tRange = t.getRange().orElseThrow(RuntimeException::new);
                if (curPos != null && !tRange.isAfter(curPos)) {
                    continue;
                }

                // find if this token belongs to any child node
                Range cRange = null;
                for (Range r : range2ChildNode.keySet()) {
                    if (r.contains(tRange)) {
                        cRange = r;
                        break;
                    }
                }

                if (cRange != null) {
                    // visit child node (skip comment)
                    Node c = range2ChildNode.get(cRange);
                    if (!(c instanceof BlockComment || c instanceof LineComment)) {
                        ast.children.add(c.accept(this, ctx));
                    }
                    curPos = cRange.end;
                    range2ChildNode.remove(cRange);
                } else {
                    // skip ws and comment tokens
                    if (t.getCategory().isWhitespaceOrComment()) {
                        continue;
                    }

                    // add current token (which does not belong to any child) as terminal
                    AST terminal = new AST();
                    terminal.astType = TERMINAL;
                    terminal.setLineno(t.getRange().get().begin.line, t.getRange().get().end.line);
                    terminal.tok = t.getText();
                    terminal.tokKind = t.getCategory().toString();
                    ast.children.add(terminal);
                    curPos = tRange.end;
                }
            }
        }

        return ast;
    }

    // terminal statements (i.e., simple statements considered as leaf nodes)

    public AST visitTerminalStmt(final Statement n, final Context ctx) {
        if (!ctx.subStmtLevel) {
            ctx.subStmtLevel = true;
            AST ast = defaultAction(n, ctx);
            ctx.subStmtLevel = false;
            return ast;
        } else {
            return defaultAction(n, ctx);
        }
    }

    @Override
    public AST visit(final AssertStmt n, final Context ctx) {
        return visitTerminalStmt(n, ctx);
    }

    @Override
    public AST visit(final ExplicitConstructorInvocationStmt n, final Context ctx) {
        return visitTerminalStmt(n, ctx);
    }

    @Override
    public AST visit(final ExpressionStmt n, final Context ctx) {
        return visitTerminalStmt(n, ctx);
    }

    @Override
    public AST visit(final ReturnStmt n, final Context ctx) {
        return visitTerminalStmt(n, ctx);
    }

    @Override
    public AST visit(final ThrowStmt n, final Context ctx) {
        return visitTerminalStmt(n, ctx);
    }

    @Override
    public AST visit(final LocalClassDeclarationStmt n, final Context ctx) {
        return visitTerminalStmt(n, ctx);
    }

    @Override
    public AST visit(final BreakStmt n, final Context ctx) {
        return visitTerminalStmt(n, ctx);
    }

    @Override
    public AST visit(final ContinueStmt n, final Context ctx) {
        return visitTerminalStmt(n, ctx);
    }

    // complex statements (for control flows)

    @Override
    public AST visit(final DoStmt n, final Context ctx) {
        return defaultAction(n, ctx);
    }

    @Override
    public AST visit(final ForEachStmt n, final Context ctx) {
        return defaultAction(n, ctx);
    }

    @Override
    public AST visit(final ForStmt n, final Context ctx) {
        return defaultAction(n, ctx);
    }

    @Override
    public AST visit(final IfStmt n, final Context ctx) {
        return defaultAction(n, ctx);
    }

    @Override
    public AST visit(final SwitchStmt n, final Context ctx) {
        return defaultAction(n, ctx);
    }

    @Override
    public AST visit(final SynchronizedStmt n, final Context ctx) {
        return defaultAction(n, ctx);
    }

    @Override
    public AST visit(final TryStmt n, final Context ctx) {
        return defaultAction(n, ctx);
    }

    @Override
    public AST visit(final WhileStmt n, final Context ctx) {
        return defaultAction(n, ctx);
    }
}
