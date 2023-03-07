package org.teco;

import java.io.FileWriter;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import com.github.javaparser.ast.Modifier;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.NodeList;
import com.github.javaparser.ast.body.BodyDeclaration;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.ConstructorDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.body.Parameter;
import com.github.javaparser.ast.body.ReceiverParameter;
import com.github.javaparser.ast.body.VariableDeclarator;
import com.github.javaparser.ast.comments.Comment;
import com.github.javaparser.ast.expr.AnnotationExpr;
import com.github.javaparser.ast.expr.AssignExpr;
import com.github.javaparser.ast.expr.BooleanLiteralExpr;
import com.github.javaparser.ast.expr.ClassExpr;
import com.github.javaparser.ast.expr.Expression;
import com.github.javaparser.ast.expr.IntegerLiteralExpr;
import com.github.javaparser.ast.expr.MemberValuePair;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.ast.expr.NameExpr;
import com.github.javaparser.ast.expr.NormalAnnotationExpr;
import com.github.javaparser.ast.expr.SimpleName;
import com.github.javaparser.ast.expr.StringLiteralExpr;
import com.github.javaparser.ast.expr.VariableDeclarationExpr;
import com.github.javaparser.ast.stmt.AssertStmt;
import com.github.javaparser.ast.stmt.BlockStmt;
import com.github.javaparser.ast.stmt.CatchClause;
import com.github.javaparser.ast.stmt.ExpressionStmt;
import com.github.javaparser.ast.stmt.Statement;
import com.github.javaparser.ast.stmt.TryStmt;
import com.github.javaparser.ast.type.ClassOrInterfaceType;
import com.github.javaparser.ast.type.ReferenceType;
import com.github.javaparser.ast.type.Type;
import com.github.javaparser.ast.type.TypeParameter;
import com.github.javaparser.ast.visitor.ModifierVisitor;
import com.github.javaparser.ast.visitor.Visitable;

public class AdHocRunnerGeneratorModifier
    extends ModifierVisitor<AdHocRunnerGeneratorModifier.Context> {

    public static final String HELPER_CLS_NAME = Helper.class.getName();
    public static final String HELPER_LOG_VAR_DEPTH = HELPER_CLS_NAME + "." + "logVarDepth";

    public static class Context {
        public String logPath;
        public String className;
        public String methodName;
        public boolean inTheClass = false;
        public boolean inTheMethod = false;
        public int stmtCnt = 0;
        public List<String> locals = new LinkedList<>();
        {
            locals.add("this");
        }
        public Set<String> localsUninitialized = new HashSet<>();
        public List<String> setupMethods = new LinkedList<>();
        public List<String> teardownMethods = new LinkedList<>();
        public String expectedException = null;

        public String getModifiedClassName() {
            return "teco_" + className;
        }
    }

    @Override
    public Visitable visit(ClassOrInterfaceType n, Context arg) {
        if (n.getNameAsString().equals(arg.className)) {
            n.setName(arg.getModifiedClassName());
        }

        return super.visit(n, arg);
    }

    @Override
    public Visitable visit(ExpressionStmt n, Context arg) {
        if (arg.inTheMethod) {
            Expression expression = (Expression) n.getExpression().accept(this, arg);
            Comment comment = n.getComment().map(s -> (Comment) s.accept(this, arg)).orElse(null);
            if (expression == null)
                return null;
            if (expression.isVariableDeclarationExpr()) {
                VariableDeclarationExpr varDeclExpr = (VariableDeclarationExpr) expression;
                for (VariableDeclarator varDecl : varDeclExpr.getVariables()) {
                    String varName = varDecl.getNameAsString();
                    arg.locals.add(varName);
                    if (!varDecl.getInitializer().isPresent()) {
                        arg.localsUninitialized.add(varName);
                    }
                }
            } else if (expression.isAssignExpr()) {
                AssignExpr agnExpr = (AssignExpr) expression;
                String target = agnExpr.getTarget().toString();
                arg.localsUninitialized.remove(target);
            }
            n.setExpression(expression);
            n.setComment(comment);
            return n;
        } else {
            return super.visit(n, arg);
        }
    }

    @Override
    public Visitable visit(ClassOrInterfaceDeclaration n, Context arg) {
        if (n.getNameAsString().equals(arg.className)) {
            arg.inTheClass = true;
            debug("original class:\n-----\n" + n.toString() + "\n-----\n", arg.logPath);
            // visit children
            NodeList<AnnotationExpr> annotations = modifyList(n.getAnnotations(), arg);
            NodeList<Modifier> modifiers = modifyList(n.getModifiers(), arg);
            NodeList<ClassOrInterfaceType> extendedTypes = modifyList(n.getExtendedTypes(), arg);
            NodeList<ClassOrInterfaceType> implementedTypes =
                modifyList(n.getImplementedTypes(), arg);
            NodeList<TypeParameter> typeParameters = modifyList(n.getTypeParameters(), arg);
            NodeList<BodyDeclaration<?>> members = modifyList(n.getMembers(), arg);
            SimpleName name = new SimpleName(arg.getModifiedClassName());
            Comment comment = n.getComment().map(s -> (Comment) s.accept(this, arg)).orElse(null);

            // add a main method
            members.add(genMethodMain(arg));

            n.setAnnotations(annotations);
            n.setModifiers(modifiers);
            n.setExtendedTypes(extendedTypes);
            n.setImplementedTypes(implementedTypes);
            n.setTypeParameters(typeParameters);
            n.setMembers(members);
            n.setName(name);
            n.setComment(comment);
            arg.inTheClass = false;
            debug("modified class:\n-----\n" + n.toString() + "\n-----\n", arg.logPath);
            return n;
        } else {
            return super.visit(n, arg);
        }
    }

    @Override
    public Visitable visit(ConstructorDeclaration n, Context arg) {
        if (arg.inTheClass) {
            ConstructorDeclaration ret = (ConstructorDeclaration) super.visit(n, arg);
            if (ret.getNameAsString().equals(arg.className)) {
                ret.setName(arg.getModifiedClassName());
            }
            return ret;
        } else {
            return super.visit(n, arg);
        }
    }

    @Override
    public Visitable visit(MethodDeclaration n, Context arg) {
        if (arg.inTheClass) {
            if (n.getNameAsString().equals(arg.methodName)) {
                arg.inTheMethod = true;
                debug("original method:\n-----\n" + n.toString() + "\n-----\n", arg.logPath);
                NodeList<AnnotationExpr> annotations = modifyList(n.getAnnotations(), arg);
                NodeList<Modifier> modifiers = modifyList(n.getModifiers(), arg);
                Type type = (Type) n.getType().accept(this, arg);
                SimpleName name = (SimpleName) n.getName().accept(this, arg);
                NodeList<Parameter> parameters = modifyList(n.getParameters(), arg);
                ReceiverParameter receiverParameter = n.getReceiverParameter()
                    .map(s -> (ReceiverParameter) s.accept(this, arg)).orElse(null);
                NodeList<ReferenceType> thrownExceptions = modifyList(n.getThrownExceptions(), arg);
                NodeList<TypeParameter> typeParameters = modifyList(n.getTypeParameters(), arg);
                Comment comment =
                    n.getComment().map(s -> (Comment) s.accept(this, arg)).orElse(null);
                if (type == null || name == null)
                    return null;

                // detect expected exception
                for (AnnotationExpr annoExpr : annotations) {
                    if (annoExpr instanceof NormalAnnotationExpr
                        && annoExpr.getNameAsString().equals("Test")) {
                        for (MemberValuePair memberValuePair : ((NormalAnnotationExpr) annoExpr)
                            .getPairs()) {
                            if (memberValuePair.getNameAsString().equals("expected")) {
                                Expression value = memberValuePair.getValue();
                                if (value instanceof ClassExpr) {
                                    arg.expectedException =
                                        ((ClassExpr) value).getType().toString();
                                }
                            }
                        }
                    }
                }

                NodeList<Statement> bodyStmts =
                    n.getBody().orElseThrow(RuntimeException::new).getStatements();
                NodeList<Statement> newBodyStmts = new NodeList<>();
                for (int stmtNo = 0; stmtNo <= bodyStmts.size(); ++stmtNo) {
                    // add logging before each stmt and after last stmt
                    for (String var : arg.locals) {
                        if (arg.localsUninitialized.contains(var)) {
                            continue;
                        }

                        // Helper.logVarDepth("var", var, logPath, stmtNo, 0);
                        newBodyStmts.add(
                            new ExpressionStmt(
                                new MethodCallExpr(
                                    HELPER_LOG_VAR_DEPTH, new StringLiteralExpr(var),
                                    new NameExpr(var), new StringLiteralExpr(arg.logPath),
                                    new IntegerLiteralExpr(String.valueOf(stmtNo)),
                                    new IntegerLiteralExpr(String.valueOf(0)))));
                    }

                    if (stmtNo == bodyStmts.size()) {
                        // already after the last stmt, nothing to do
                    } else {
                        Statement stmt = bodyStmts.get(stmtNo);
                        newBodyStmts.add((Statement) stmt.accept(this, arg));
                    }
                }

                n.setAnnotations(annotations);
                n.setModifiers(modifiers);
                n.setBody(new BlockStmt(newBodyStmts));
                n.setType(type);
                n.setName(name);
                n.setParameters(parameters);
                n.setReceiverParameter(receiverParameter);
                n.setThrownExceptions(thrownExceptions);
                n.setTypeParameters(typeParameters);
                n.setComment(comment);
                debug("modified method: \n-----\n" + n.toString() + "\n-----\n", arg.logPath);
                arg.inTheMethod = false;
                return n;
            } else if (n.getNameAsString().equals("main")) {
                // remove any existing main method
                return null;
            } else {
                Set<String> annotations = n.getAnnotations().stream()
                    .map(AnnotationExpr::getNameAsString).collect(Collectors.toSet());
                if (annotations.contains("Before") || annotations.contains("BeforeEach")
                    || annotations.contains("BeforeClass") || annotations.contains("BeforeAll")) {
                    // detect & keep setup methods
                    arg.setupMethods.add(n.getNameAsString());
                    return super.visit(n, arg);
                } else if (annotations.contains("After") || annotations.contains("AfterEach")
                    || annotations.contains("AfterClass") || annotations.contains("AfterAll")) {
                    // detect & keep teardown methods
                    arg.teardownMethods.add(n.getNameAsString());
                    return super.visit(n, arg);
                } else if (annotations.contains("Test")) {
                    // remove other test methods
                    return null;
                } else {
                    // keep all other methods (could be utility methods)
                    return super.visit(n, arg);
                }
            }
        } else {
            return super.visit(n, arg);
        }
    }

    protected MethodDeclaration genMethodMain(Context arg) {
        MethodDeclaration method = new MethodDeclaration();
        // public static void main(String... args) throws Exception {
        method.setName("main");
        method.setType(new ClassOrInterfaceType(null, "void"));
        method
            .setModifiers(NodeList.nodeList(Modifier.publicModifier(), Modifier.staticModifier()));
        method.setParameters(
            NodeList.nodeList(new Parameter(new ClassOrInterfaceType(null, "String..."), "args")));
        method.setThrownExceptions(NodeList.nodeList(new ClassOrInterfaceType(null, "Exception")));
        debug("main: signature ok", arg.logPath);

        NodeList<Statement> body = new NodeList<>();
        //   TestClass instance = TestClass.class.newInstance();
        // not using new TestClass() which always causes compilation error when the test class's constructor has arguments (e.g., for parameterized tests); there is a chance that JUnit 4 runner can run this test without using this main method
        body.add(
            new ExpressionStmt(
                new VariableDeclarationExpr(
                    new VariableDeclarator(
                        new ClassOrInterfaceType(null, arg.getModifiedClassName()), "instance",
                        new MethodCallExpr(
                            new ClassExpr(
                                new ClassOrInterfaceType(null, arg.getModifiedClassName())),
                            "newInstance")))));

        // execute all setup methods
        for (String setupMethod : arg.setupMethods) {
            //   instance.setup();
            body.add(new ExpressionStmt(new MethodCallExpr(new NameExpr("instance"), setupMethod)));
        }

        // execute test method
        if (arg.expectedException == null) {
            //   instance.test();
            body.add(
                new ExpressionStmt(new MethodCallExpr(new NameExpr("instance"), arg.methodName)));
        } else {
            //   try {
            TryStmt tryStmt = new TryStmt();
            NodeList<Statement> tryBody = new NodeList<>();
            //     instance.test();
            tryBody.add(
                new ExpressionStmt(new MethodCallExpr(new NameExpr("instance"), arg.methodName)));
            //     assert false, "expected expectedException, but none was thrown";
            tryBody.add(
                new AssertStmt(
                    new BooleanLiteralExpr(false), new StringLiteralExpr(
                        "expected " + arg.expectedException + ", but none was thrown")));
            //   }
            tryStmt.setTryBlock(new BlockStmt(tryBody));
            //   catch (expectedException e) { }
            CatchClause catchClause = new CatchClause(
                new Parameter(new ClassOrInterfaceType(null, arg.expectedException), "e"),
                new BlockStmt());
            tryStmt.setCatchClauses(NodeList.nodeList(catchClause));

            body.add(tryStmt);
        }

        //   execute all teardown methods
        for (String teardownMethod : arg.teardownMethods) {
            //   instance.teardown();
            body.add(
                new ExpressionStmt(new MethodCallExpr(new NameExpr("instance"), teardownMethod)));
        }

        // }
        method.setBody(new BlockStmt(body));
        debug("generated main method: \n-----\n" + method.toString() + "\n-----\n", arg.logPath);
        return method;
    }

    @SuppressWarnings("unchecked")
    private <N extends Node> NodeList<N> modifyList(NodeList<N> list, Context arg) {
        return (NodeList<N>) list.accept(this, arg);
    }

    // private Expression chainedPlus(Expression... exprs) {
    //     if (exprs.length == 0) {
    //         return null;
    //     } else if (exprs.length == 1) {
    //         return exprs[0];
    //     } else {
    //         Expression ret = new BinaryExpr(exprs[0], exprs[1], BinaryExpr.Operator.PLUS);
    //         for (int i = 2; i < exprs.length; ++i) {
    //             ret = new BinaryExpr(ret, exprs[i], BinaryExpr.Operator.PLUS);
    //         }
    //         return ret;
    //     }
    // }

    private void debug(String msg, String logPath) {
        try (FileWriter writer = new FileWriter(logPath + "/debug.log", true)) {
            writer.write(msg);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException(e);
        }
    }
}
