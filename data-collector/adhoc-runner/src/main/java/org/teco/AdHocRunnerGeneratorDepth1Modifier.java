package org.teco;

import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import org.apache.commons.lang3.tuple.Pair;
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
import com.github.javaparser.ast.expr.BinaryExpr;
import com.github.javaparser.ast.expr.BooleanLiteralExpr;
import com.github.javaparser.ast.expr.CastExpr;
import com.github.javaparser.ast.expr.ClassExpr;
import com.github.javaparser.ast.expr.Expression;
import com.github.javaparser.ast.expr.InstanceOfExpr;
import com.github.javaparser.ast.expr.IntegerLiteralExpr;
import com.github.javaparser.ast.expr.MemberValuePair;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.ast.expr.NameExpr;
import com.github.javaparser.ast.expr.NormalAnnotationExpr;
import com.github.javaparser.ast.expr.NullLiteralExpr;
import com.github.javaparser.ast.expr.ObjectCreationExpr;
import com.github.javaparser.ast.expr.SimpleName;
import com.github.javaparser.ast.expr.StringLiteralExpr;
import com.github.javaparser.ast.expr.VariableDeclarationExpr;
import com.github.javaparser.ast.stmt.AssertStmt;
import com.github.javaparser.ast.stmt.BlockStmt;
import com.github.javaparser.ast.stmt.CatchClause;
import com.github.javaparser.ast.stmt.ContinueStmt;
import com.github.javaparser.ast.stmt.ExpressionStmt;
import com.github.javaparser.ast.stmt.ForEachStmt;
import com.github.javaparser.ast.stmt.IfStmt;
import com.github.javaparser.ast.stmt.ReturnStmt;
import com.github.javaparser.ast.stmt.Statement;
import com.github.javaparser.ast.stmt.ThrowStmt;
import com.github.javaparser.ast.stmt.TryStmt;
import com.github.javaparser.ast.type.ClassOrInterfaceType;
import com.github.javaparser.ast.type.ReferenceType;
import com.github.javaparser.ast.type.Type;
import com.github.javaparser.ast.type.TypeParameter;
import com.github.javaparser.ast.visitor.ModifierVisitor;
import com.github.javaparser.ast.visitor.Visitable;

public class AdHocRunnerGeneratorDepth1Modifier
    extends ModifierVisitor<AdHocRunnerGeneratorDepth1Modifier.Context> {

    public static class Context {
        public String logPath;
        public String className;
        public String methodName;
        public boolean inTheClass = false;
        public boolean inTheMethod = false;
        public int stmtCnt = 0;
        public List<Pair<String, String>> localsPrimitive = new LinkedList<>();
        public List<String> localsOther = new LinkedList<>();
        public Set<String> localsUninitialized = new HashSet<>();
        public List<String> setupMethods = new LinkedList<>();
        public List<String> teardownMethods = new LinkedList<>();
        public String expectedException = null;

        public String getModifiedClassName() {
            return "teco_" + className;
        }
    }

    protected static final Set<String> PRIMITIVE_TYPES;
    static {
        Set<String> primitiveTypesInit = new HashSet<>();
        primitiveTypesInit.add("boolean");
        primitiveTypesInit.add("byte");
        primitiveTypesInit.add("char");
        primitiveTypesInit.add("short");
        primitiveTypesInit.add("int");
        primitiveTypesInit.add("long");
        primitiveTypesInit.add("float");
        primitiveTypesInit.add("double");
        primitiveTypesInit.add("String");
        PRIMITIVE_TYPES = Collections.unmodifiableSet(primitiveTypesInit);
    }

    protected static final String LOG_TO_FILE_NAME = "teco_logToFile";
    protected static final String VALUE_OF_NAME = "teco_valueOf";
    protected static final String LOG_VALUES_DEPTH_1_NAME = "teco_logValuesDepth1";

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
                    String typeName = varDecl.getType().toString();
                    if (PRIMITIVE_TYPES.contains(typeName)) {
                        arg.localsPrimitive.add(Pair.of(varName, typeName));
                    } else {
                        arg.localsOther.add(varName);
                    }
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

            // add util methods
            for (MethodDeclaration method : genMethodsValueOf(arg)) {
                members.add(method);
            }
            members.add(genMethodValueOfObject(arg));
            members.add(genMethodLogValuesDepth1(arg));
            members.add(genMethodLogToFile(arg));

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
                for (int stmtCnt = 0; stmtCnt <= bodyStmts.size(); ++stmtCnt) {
                    // add logToFile before each stmt
                    if (arg.localsPrimitive.size() > 0) {
                        List<Expression> msgExprs = new LinkedList<>();
                        for (Pair<String, String> pair : arg.localsPrimitive) {
                            String local = pair.getLeft();

                            // skip uninitialized locals
                            if (arg.localsUninitialized.contains(local)) {
                                continue;
                            }

                            // "local " + valueOf(local) + "\n"
                            msgExprs.add(new StringLiteralExpr(local + " "));
                            msgExprs.add(new MethodCallExpr(VALUE_OF_NAME, new NameExpr(local)));
                            msgExprs.add(new StringLiteralExpr("\\n"));
                        }
                        Expression msgExpr = chainedPlus(msgExprs.toArray(new Expression[0]));
                        if (msgExpr != null) {
                            newBodyStmts.add(
                                new ExpressionStmt(
                                    new MethodCallExpr(
                                        LOG_TO_FILE_NAME,
                                        new IntegerLiteralExpr(String.valueOf(stmtCnt)),
                                        new StringLiteralExpr("primitive-values"), msgExpr)));
                        }
                    }
                    if (arg.localsOther.size() > 0) {
                        for (String local : arg.localsOther) {
                            // skip uninitialized locals
                            if (arg.localsUninitialized.contains(local)) {
                                continue;
                            }

                            // logValuesDepth1(stmtNo, "local", local);
                            newBodyStmts.add(
                                new ExpressionStmt(
                                    new MethodCallExpr(
                                        LOG_VALUES_DEPTH_1_NAME,
                                        new IntegerLiteralExpr(String.valueOf(stmtCnt)),
                                        new StringLiteralExpr(local), new NameExpr(local))));
                        }
                    }

                    if (stmtCnt == bodyStmts.size()) {
                        // already after the last stmt, nothing to do
                    } else {
                        Statement stmt = bodyStmts.get(stmtCnt);
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
            } else if (n.getNameAsString().equals(LOG_TO_FILE_NAME)) {
                // remove any existing method that happens to be named logToFile
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

    protected List<MethodDeclaration> genMethodsValueOf(Context arg) {
        List<MethodDeclaration> methods = new LinkedList<>();
        for (String type : PRIMITIVE_TYPES) {
            MethodDeclaration method = new MethodDeclaration();
            // public static String valueOf(<type> value) {
            method.setName(VALUE_OF_NAME);
            method.setType(new ClassOrInterfaceType(null, "String"));
            method.setModifiers(
                NodeList.nodeList(Modifier.publicModifier(), Modifier.staticModifier()));
            method.setParameters(
                NodeList.nodeList(new Parameter(new ClassOrInterfaceType(null, type), "value")));

            List<Expression> msgExprs = new LinkedList<>();
            if (type.equals("String")) {
                // "\"" + value.replace("\\", "\\\\").replace("\n", "\\n").replace("\"", "\\\"") + "\""
                msgExprs.add(new StringLiteralExpr("\\\""));
                MethodCallExpr replaced = new MethodCallExpr(
                    new NameExpr("value"), "replace", NodeList.nodeList(
                        new StringLiteralExpr("\\\\"), new StringLiteralExpr("\\\\\\\\")));
                replaced = new MethodCallExpr(
                    replaced, "replace", NodeList
                        .nodeList(new StringLiteralExpr("\\n"), new StringLiteralExpr("\\\\n")));
                replaced = new MethodCallExpr(
                    replaced, "replace", NodeList.nodeList(
                        new StringLiteralExpr("\\\""), new StringLiteralExpr("\\\\\\\"")));
                msgExprs.add(replaced);
                msgExprs.add(new StringLiteralExpr("\\\""));
            } else if (type.equals("char")) {
                // "'" + value + "'"
                msgExprs.add(new StringLiteralExpr("'"));
                msgExprs.add(new NameExpr("value"));
                msgExprs.add(new StringLiteralExpr("'"));

            } else {
                // String.valueOf(value);
                msgExprs.add(new MethodCallExpr("String.valueOf", new NameExpr("value")));

            }
            // return ^;
            method.setBody(
                new BlockStmt(
                    NodeList.nodeList(
                        new ReturnStmt(chainedPlus(msgExprs.toArray(new Expression[0]))))));

            methods.add(method);
        }
        return methods;
    }

    protected MethodDeclaration genMethodValueOfObject(Context arg) {
        MethodDeclaration method = new MethodDeclaration();
        // public static String valueOf(Object value) {
        method.setName(VALUE_OF_NAME);
        method.setType(new ClassOrInterfaceType(null, "String"));
        method
            .setModifiers(NodeList.nodeList(Modifier.publicModifier(), Modifier.staticModifier()));
        method.setParameters(
            NodeList.nodeList(new Parameter(new ClassOrInterfaceType(null, "Object"), "value")));
        NodeList<Statement> stmts = new NodeList<>();

        //   if (value instanceof Boolean) { return valueOf((boolean) value); }
        stmts.add(
            new IfStmt(
                new InstanceOfExpr(
                    new NameExpr("value"), new ClassOrInterfaceType(null, "Boolean")),
                new ReturnStmt(
                    new MethodCallExpr(
                        VALUE_OF_NAME,
                        new CastExpr(
                            new ClassOrInterfaceType(null, "boolean"), new NameExpr("value")))),
                null));
        //   if (value instanceof Byte) { return valueOf((byte) value); }
        stmts.add(
            new IfStmt(
                new InstanceOfExpr(new NameExpr("value"), new ClassOrInterfaceType(null, "Byte")),
                new ReturnStmt(
                    new MethodCallExpr(
                        VALUE_OF_NAME,
                        new CastExpr(
                            new ClassOrInterfaceType(null, "byte"), new NameExpr("value")))),
                null));
        //   if (value instanceof Character) { return valueOf((char) value); }
        stmts.add(
            new IfStmt(
                new InstanceOfExpr(
                    new NameExpr("value"), new ClassOrInterfaceType(null, "Character")),
                new ReturnStmt(
                    new MethodCallExpr(
                        VALUE_OF_NAME,
                        new CastExpr(
                            new ClassOrInterfaceType(null, "char"), new NameExpr("value")))),
                null));
        //   if (value instanceof Short) { return valueOf((short) value); }
        stmts.add(
            new IfStmt(
                new InstanceOfExpr(new NameExpr("value"), new ClassOrInterfaceType(null, "Short")),
                new ReturnStmt(
                    new MethodCallExpr(
                        VALUE_OF_NAME,
                        new CastExpr(
                            new ClassOrInterfaceType(null, "short"), new NameExpr("value")))),
                null));
        //   if (value instanceof Integer) { return valueOf((int) value); }
        stmts
            .add(
                new IfStmt(
                    new InstanceOfExpr(
                        new NameExpr("value"), new ClassOrInterfaceType(null, "Integer")),
                    new ReturnStmt(
                        new MethodCallExpr(
                            VALUE_OF_NAME,
                            new CastExpr(
                                new ClassOrInterfaceType(null, "int"), new NameExpr("value")))),
                    null));
        //   if (value instanceof Long) { return valueOf((long) value); }
        stmts.add(
            new IfStmt(
                new InstanceOfExpr(new NameExpr("value"), new ClassOrInterfaceType(null, "Long")),
                new ReturnStmt(
                    new MethodCallExpr(
                        VALUE_OF_NAME,
                        new CastExpr(
                            new ClassOrInterfaceType(null, "long"), new NameExpr("value")))),
                null));
        //   if (value instanceof Float) { return valueOf((float) value); }
        stmts.add(
            new IfStmt(
                new InstanceOfExpr(new NameExpr("value"), new ClassOrInterfaceType(null, "Float")),
                new ReturnStmt(
                    new MethodCallExpr(
                        VALUE_OF_NAME,
                        new CastExpr(
                            new ClassOrInterfaceType(null, "float"), new NameExpr("value")))),
                null));
        //   if (value instanceof Double) { return valueOf((double) value); }
        stmts.add(
            new IfStmt(
                new InstanceOfExpr(new NameExpr("value"), new ClassOrInterfaceType(null, "Double")),
                new ReturnStmt(
                    new MethodCallExpr(
                        VALUE_OF_NAME,
                        new CastExpr(
                            new ClassOrInterfaceType(null, "double"), new NameExpr("value")))),
                null));
        //   if (value instanceof String) { return valueOf((String) value); }
        stmts.add(
            new IfStmt(
                new InstanceOfExpr(new NameExpr("value"), new ClassOrInterfaceType(null, "String")),
                new ReturnStmt(
                    new MethodCallExpr(
                        VALUE_OF_NAME,
                        new CastExpr(
                            new ClassOrInterfaceType(null, "String"), new NameExpr("value")))),
                null));
        //   throw new RuntimeException("not a primitive type");
        stmts.add(
            new ThrowStmt(
                new ObjectCreationExpr(
                    null, new ClassOrInterfaceType(null, "RuntimeException"),
                    NodeList.nodeList(new StringLiteralExpr("not a primitive type")))));
        // }
        method.setBody(new BlockStmt(stmts));

        return method;
    }

    protected MethodDeclaration genMethodLogValuesDepth1(Context arg) {
        MethodDeclaration method = new MethodDeclaration();
        // public static void logValuesDepth1(int stmtNo, String name, Object obj) {
        method.setName(LOG_VALUES_DEPTH_1_NAME);
        method.setType(new ClassOrInterfaceType(null, "void"));
        method
            .setModifiers(NodeList.nodeList(Modifier.publicModifier(), Modifier.staticModifier()));
        method.setParameters(
            NodeList.nodeList(
                new Parameter(new ClassOrInterfaceType(null, "int"), "stmtNo"),
                new Parameter(new ClassOrInterfaceType(null, "String"), "name"),
                new Parameter(new ClassOrInterfaceType(null, "Object"), "obj")));

        NodeList<Statement> stmts = new NodeList<>();

        //   if (obj == null) { return; }
        stmts.add(
            new IfStmt(
                new BinaryExpr(
                    new NameExpr("obj"), new NullLiteralExpr(), BinaryExpr.Operator.EQUALS),
                new ReturnStmt(), null));

        //   Class<?> clz = obj.getClass();
        stmts.add(
            new ExpressionStmt(
                new VariableDeclarationExpr(
                    new VariableDeclarator(
                        new ClassOrInterfaceType(null, "Class<?>"), "clz",
                        new MethodCallExpr(new NameExpr("obj"), "getClass")))));

        //   for (java.lang.reflect.Field fld : clz.getDeclaredFields()) {
        ForEachStmt forEachStmt = new ForEachStmt();
        forEachStmt.setVariable(
            new VariableDeclarationExpr(
                new ClassOrInterfaceType(null, "java.lang.reflect.Field"), "fld"));
        forEachStmt.setIterable(new MethodCallExpr(new NameExpr("clz"), "getDeclaredFields"));
        NodeList<Statement> forEachStmtBody = new NodeList<>();

        //     Class<?> fldType = fld.getType();
        forEachStmtBody.add(
            new ExpressionStmt(
                new VariableDeclarationExpr(
                    new VariableDeclarator(
                        new ClassOrInterfaceType(null, "Class<?>"), "fldType",
                        new MethodCallExpr(new NameExpr("fld"), "getType")))));

        //     if (fldType.isPrimitive() || fldType.getSimpleName().equals("String")) {
        IfStmt ifStmt = new IfStmt();
        ifStmt.setCondition(
            new BinaryExpr(
                new MethodCallExpr(new NameExpr("fldType"), "isPrimitive"),
                new MethodCallExpr(
                    new MethodCallExpr(new NameExpr("fldType"), "getSimpleName"), "equals",
                    NodeList.nodeList(new StringLiteralExpr("String"))),
                BinaryExpr.Operator.OR));
        NodeList<Statement> ifStmtThen = new NodeList<>();

        //       try {
        TryStmt tryStmt = new TryStmt();
        NodeList<Statement> tryStmtBody = new NodeList<>();

        //         fld.setAccessible(true);
        tryStmtBody.add(
            new ExpressionStmt(
                new MethodCallExpr(
                    new NameExpr("fld"), "setAccessible",
                    NodeList.nodeList(new BooleanLiteralExpr(true)))));
        //         Object value = fld.get(obj);
        tryStmtBody.add(
            new ExpressionStmt(
                new VariableDeclarationExpr(
                    new VariableDeclarator(
                        new ClassOrInterfaceType(null, "Object"), "value", new MethodCallExpr(
                            new NameExpr("fld"), "get", NodeList.nodeList(new NameExpr("obj")))))));
        //         if (value == null) { continue; }
        tryStmtBody.add(
            new IfStmt(
                new BinaryExpr(
                    new NameExpr("value"), new NullLiteralExpr(), BinaryExpr.Operator.EQUALS),
                new ContinueStmt(), null));
        //         logToFile(stmtNo, "depth1-values", name + "." + fld.getName() + " " + valueOf(value) + "\n");
        List<Expression> msgExprs = new LinkedList<>();
        msgExprs.add(new NameExpr("name"));
        msgExprs.add(new StringLiteralExpr("."));
        msgExprs.add(new MethodCallExpr(new NameExpr("fld"), "getName"));
        msgExprs.add(new StringLiteralExpr(" "));
        msgExprs.add(new MethodCallExpr(VALUE_OF_NAME, new NameExpr("value")));
        msgExprs.add(new StringLiteralExpr("\n"));
        tryStmtBody.add(
            new ExpressionStmt(
                new MethodCallExpr(
                    LOG_TO_FILE_NAME, new NameExpr("stmtNo"),
                    new StringLiteralExpr("depth1-values"),
                    chainedPlus(msgExprs.toArray(new Expression[0])))));
        //         }
        tryStmt.setTryBlock(new BlockStmt(tryStmtBody));

        //         catch (Exception e) { }
        tryStmt.setCatchClauses(
            NodeList.nodeList(
                new CatchClause(
                    new Parameter(new ClassOrInterfaceType(null, "Exception"), "e"),
                    new BlockStmt())));

        //       }
        ifStmtThen.add(tryStmt);

        //     } // if
        ifStmt.setThenStmt(new BlockStmt(ifStmtThen));
        forEachStmtBody.add(ifStmt);
        //   } // for each
        forEachStmt.setBody(new BlockStmt(forEachStmtBody));
        stmts.add(forEachStmt);

        // }
        method.setBody(new BlockStmt(stmts));

        return method;
    }

    protected MethodDeclaration genMethodLogToFile(Context arg) {
        MethodDeclaration method = new MethodDeclaration();
        // public static void logToFile(int stmtNo, String prefix, String msg) {
        method.setName(LOG_TO_FILE_NAME);
        method.setType(new ClassOrInterfaceType(null, "void"));
        method
            .setModifiers(NodeList.nodeList(Modifier.publicModifier(), Modifier.staticModifier()));
        method.setParameters(
            NodeList.nodeList(
                new Parameter(new ClassOrInterfaceType(null, "int"), "stmtNo"),
                new Parameter(new ClassOrInterfaceType(null, "String"), "prefix"),
                new Parameter(new ClassOrInterfaceType(null, "String"), "msg")));

        //   try {
        TryStmt wrapper = new TryStmt();
        NodeList<Statement> body = new NodeList<>();
        //     java.io.FileWriter writer = new java.io.FileWriter("arg.logPath/" + prefix + "-" + stmtNo, true);
        body.add(
            new ExpressionStmt(
                new VariableDeclarationExpr(
                    new VariableDeclarator(
                        new ClassOrInterfaceType(null, "java.io.FileWriter"), "writer",
                        new ObjectCreationExpr(
                            null, new ClassOrInterfaceType(null, "java.io.FileWriter"),
                            NodeList.nodeList(
                                chainedPlus(
                                    new StringLiteralExpr(arg.logPath + "/"),
                                    new NameExpr("prefix"), new StringLiteralExpr("-"),
                                    new NameExpr("stmtNo")),
                                new BooleanLiteralExpr(true)))))));
        //     writer.write(msg);
        body.add(new ExpressionStmt(new MethodCallExpr("writer.write", new NameExpr("msg"))));
        //     writer.close();
        body.add(new ExpressionStmt(new MethodCallExpr("writer.close")));
        //   }
        wrapper.setTryBlock(new BlockStmt(body));

        //   catch (Exception e) {
        //     e.printStackTrace();
        //     System.exit(1);
        //   }
        CatchClause catcher = new CatchClause(
            new Parameter(new ClassOrInterfaceType(null, "Exception"), "e"),
            new BlockStmt(
                NodeList.nodeList(
                    new ExpressionStmt(new MethodCallExpr("e.printStackTrace")), new ExpressionStmt(
                        new MethodCallExpr("System.exit", new IntegerLiteralExpr("1"))))));
        wrapper.setCatchClauses(NodeList.nodeList(catcher));

        // }
        method.setBody(new BlockStmt(NodeList.nodeList(wrapper)));
        debug(
            "generated logToFile method: \n-----\n" + method.toString() + "\n-----\n", arg.logPath);
        return method;
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

    private Expression chainedPlus(Expression... exprs) {
        if (exprs.length == 0) {
            return null;
        } else if (exprs.length == 1) {
            return exprs[0];
        } else {
            Expression ret = new BinaryExpr(exprs[0], exprs[1], BinaryExpr.Operator.PLUS);
            for (int i = 2; i < exprs.length; ++i) {
                ret = new BinaryExpr(ret, exprs[i], BinaryExpr.Operator.PLUS);
            }
            return ret;
        }
    }

    private void debug(String msg, String logPath) {
        // try (FileWriter writer = new FileWriter(logPath + "/debug.log", true)) {
        //     writer.write(msg);
        // } catch (Exception e) {
        //     e.printStackTrace();
        //     throw new RuntimeException(e);
        // }
    }
}
