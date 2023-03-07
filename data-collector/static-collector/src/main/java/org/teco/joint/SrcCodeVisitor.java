package org.teco.joint;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.apache.commons.collections4.map.LazyMap;
import org.apache.commons.lang3.tuple.Triple;
import org.objectweb.asm.tree.InsnList;
import org.objectweb.asm.tree.MethodNode;
import org.teco.AllCollectors;
import org.teco.util.BytecodeEncoder;
import org.teco.util.BytecodeUtils;
import org.teco.util.TypeResolver;
import com.github.javaparser.ast.NodeList;
import com.github.javaparser.ast.body.AnnotationDeclaration;
import com.github.javaparser.ast.body.CallableDeclaration;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.ConstructorDeclaration;
import com.github.javaparser.ast.body.EnumDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.body.Parameter;
import com.github.javaparser.ast.expr.ObjectCreationExpr;
import com.github.javaparser.ast.nodeTypes.NodeWithTypeParameters;
import com.github.javaparser.ast.type.ClassOrInterfaceType;
import com.github.javaparser.ast.type.TypeParameter;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import com.github.javaparser.printer.configuration.DefaultConfigurationOption;
import com.github.javaparser.printer.configuration.DefaultPrinterConfiguration;
import com.github.javaparser.printer.configuration.PrinterConfiguration;
import me.xdrop.fuzzywuzzy.FuzzySearch;

public class SrcCodeVisitor extends VoidVisitorAdapter<SrcCodeVisitor.Context> {

    ExtractASTVisitor astVisitor = new ExtractASTVisitor();

    public static class Context implements Cloneable {
        /** package name (could be empty) */
        String pName = null;

        /** class name */
        String cName = null;

        /** fully qualified class name = package name (if present) . class name */
        String fqCName = null;

        /** 
         * anonymous class count tracker
         * should be freshed for each new class context
         */
        int anonymousClassCount = 1;

        /**
         * local class count tracker
         * not deep cloned: should be refreshed for each new class context
         */
        LazyMap<String, Integer> localClassCount = LazyMap.lazyMap(new HashMap<>(), () -> 1);

        /**
         * additional constructor parameters tracker
         * deep cloned
         */
        List<String> extraInitParams = new LinkedList<>();

        /**
         * type parameters mapping tracker
         * deep cloned
         */
        Map<String, String> typeParams = new HashMap<>();

        /** class id */
        int cid = -1;

        /** class structure */
        ClassStructure cs = null;

        /** 
         * signature to method id map in current class 
         * not deep cloned: should be unmodifiable
         */
        Map<String, Integer> sign2mid = null;

        @Override
        public Context clone() {
            try {
                Context cloned = (Context) super.clone();
                cloned.typeParams = new HashMap<>(typeParams);
                cloned.extraInitParams = new LinkedList<>(extraInitParams);
                return cloned;
            } catch (CloneNotSupportedException e) {
                throw new RuntimeException(e);
            }
        }
    }

    private static PrinterConfiguration METHOD_PPRINT_CONFIG = new DefaultPrinterConfiguration();
    static {
        METHOD_PPRINT_CONFIG.removeOption(
            new DefaultConfigurationOption(DefaultPrinterConfiguration.ConfigOption.PRINT_COMMENTS))
            .removeOption(
                new DefaultConfigurationOption(
                    DefaultPrinterConfiguration.ConfigOption.PRINT_JAVADOC));
    }

    @Override
    public void visit(ClassOrInterfaceDeclaration n, Context ctx) {
        String name = null;
        if (n.isLocalClassDeclaration()) {
            // local class is named as OuterClass$%dInnerClass
            int cnt = ctx.localClassCount.get(n.getNameAsString());
            name = cnt + n.getNameAsString();
            ctx.localClassCount.put(n.getNameAsString(), cnt + 1);
        } else {
            name = n.getNameAsString();
        }

        List<String> extraInitParams = null;
        if (ctx.cName != null && !n.isStatic()) {
            // non-static inner class has extra init parameter OuterClass
            extraInitParams = Arrays.asList(ctx.fqCName);
        }

        ctx = getNewContextForTypeDeclaration(name, extraInitParams, ctx);
        registerTypeParameters(n, ctx);
        // AllCollectors.warning("visit(ClassOrInterfaceDeclaration) " + ctx.fqCName);
        super.visit(n, ctx);
    }

    @Override
    public void visit(EnumDeclaration n, Context ctx) {
        // enum constructor has additional parameters String,int
        super.visit(
            n, getNewContextForTypeDeclaration(
                n.getNameAsString(), Arrays.asList("java.lang.String", "int"), ctx));
    }

    @Override
    public void visit(AnnotationDeclaration n, Context ctx) {
        super.visit(n, getNewContextForTypeDeclaration(n.getNameAsString(), ctx));
    }

    @Override
    public void visit(ObjectCreationExpr n, Context ctx) {
        n.getAnonymousClassBody().ifPresent(l -> {
            Context newCtx =
                getNewContextForTypeDeclaration(String.valueOf(ctx.anonymousClassCount), ctx);
            ++ctx.anonymousClassCount;
            l.forEach(v -> v.accept(this, newCtx));
        });

        n.getArguments().forEach(p -> p.accept(this, ctx));
        n.getScope().ifPresent(l -> l.accept(this, ctx));
        n.getType().accept(this, ctx);
        n.getTypeArguments().ifPresent(l -> l.forEach(v -> v.accept(this, ctx)));
        n.getComment().ifPresent(l -> l.accept(this, ctx));
    }

    protected Context getNewContextForTypeDeclaration(String name, Context ctx) {
        return getNewContextForTypeDeclaration(name, null, ctx);
    }

    protected Context getNewContextForTypeDeclaration(String name, List<String> extraInitParams,
        Context ctx) {
        String newCName = "";
        if (ctx.cName != null) {
            // inner class
            newCName += ctx.cName + "$";
        }

        newCName += name;

        Context newCtx = ctx.clone();
        newCtx.cName = newCName;
        newCtx.anonymousClassCount = 1;
        newCtx.localClassCount = LazyMap.lazyMap(new HashMap<>(), () -> 1);
        if (extraInitParams != null) {
            newCtx.extraInitParams.addAll(extraInitParams);
        }

        if (newCtx.pName.isEmpty()) {
            newCtx.fqCName = newCName;
        } else {
            newCtx.fqCName = newCtx.pName + "." + newCName;
        }

        // try to match with existing class structure from byte code
        newCtx.cid = JointCollector.name2cid.getOrDefault(newCtx.fqCName, -2);
        // AllCollectors.warning("Class " + newCtx.fqCName + " has id " + newCtx.cid);
        if (newCtx.cid >= 0 && JointCollector.cid2sign2mid.containsKey(newCtx.cid)) {
            newCtx.cs = JointCollector.classes.get(newCtx.cid);
            newCtx.sign2mid =
                Collections.unmodifiableMap(JointCollector.cid2sign2mid.get(newCtx.cid));
        } else {
            newCtx.cs = null;
            newCtx.sign2mid = null;
            AllCollectors.warning(
                "Cannot find byte code for src code class " + newCtx.fqCName
                    + "\n  fuzzy matching: "
                    + FuzzySearch.extractTop(newCtx.fqCName, JointCollector.name2cid.keySet(), 10));
        }

        return newCtx;
    }

    protected void registerTypeParameters(NodeWithTypeParameters<?> n, Context ctx) {
        NodeList<TypeParameter> typeParams = n.getTypeParameters();
        if (typeParams != null && !typeParams.isEmpty()) {
            for (TypeParameter typeParameter : typeParams) {
                NodeList<ClassOrInterfaceType> typeBound = typeParameter.getTypeBound();
                String mapTo = "java.lang.Object";
                if (typeBound != null && !typeBound.isEmpty()) {
                    mapTo = TypeResolver.resolveType(typeBound.get(0), ctx.typeParams);
                }
                ctx.typeParams.put(typeParameter.getNameAsString(), mapTo);
            }
        }
    }

    @Override
    public void visit(MethodDeclaration n, Context ctx) {
        if (!n.getTypeParameters().isEmpty()) {
            ctx = ctx.clone();
            registerTypeParameters(n, ctx);
        }

        visitCallableDeclaration(n, n.getNameAsString(), null, ctx);
        super.visit(n, ctx);
    }

    @Override
    public void visit(ConstructorDeclaration n, Context ctx) {
        if (!n.getTypeParameters().isEmpty()) {
            ctx = ctx.clone();
            registerTypeParameters(n, ctx);
        }

        visitCallableDeclaration(n, "<init>", ctx.extraInitParams, ctx);
        super.visit(n, ctx);
    }

    private void visitCallableDeclaration(CallableDeclaration<?> n, String name,
        List<String> extraParams, Context ctx) {
        // AllCollectors.warning("visitCallableDeclaration " + ctx.fqCName + " " + name);
        if (ctx.sign2mid == null) {
            // didn't find a matching class in byte code
            return;
        }

        // code (literal)
        String code = n.toString(METHOD_PPRINT_CONFIG);

        // refrain from collecting more data if the size of source code is huge
        if (code.length() > 100_000 || code.split("\n").length > 1_000) {
            AllCollectors.warning("Skipping a large method: " + ctx.fqCName + "." + name);
            return;
        }

        // resolve the return type and parameter types
        String resolvedRtype = null;
        if (n instanceof MethodDeclaration) {
            resolvedRtype =
                TypeResolver.resolveType(((MethodDeclaration) n).getType(), ctx.typeParams);
        } else {
            resolvedRtype = "void";
        }

        List<String> resolvedPtypes = new LinkedList<>();
        if (extraParams != null) {
            resolvedPtypes.addAll(extraParams);
        }
        for (Parameter param : n.getParameters()) {
            String ptype = TypeResolver.resolveType(param.getType(), ctx.typeParams);
            if (param.isVarArgs()) {
                ptype = ptype + "[]";
            }
            resolvedPtypes.add(ptype);
        }

        // try to match with existing method structure from byte code
        String sign = name + "(" + String.join(",", resolvedPtypes) + ")" + resolvedRtype;
        int mid = ctx.sign2mid.getOrDefault(sign, -2);
        if (mid < 0 && sign.contains("$")) {
            // try to match again after erasing the local class numbering (because JavaParser doesn't reason about it properly)
            for (Map.Entry<String, Integer> entry : ctx.sign2mid.entrySet()) {
                if (sign.equals(entry.getKey().replaceAll("\\$\\d+", "\\$"))) {
                    mid = entry.getValue();
                    break;
                }
            }
        }
        if (mid < 0) {
            AllCollectors.warning(
                "Cannot find byte code for src code method " + ctx.fqCName + " " + sign
                    + "\n  fuzzy matching: "
                    + FuzzySearch.extractTop(sign, ctx.sign2mid.keySet(), 10));
            return;
        }

        MethodStructure ms = JointCollector.methods.get(mid);
        MethodNode mn = JointCollector.methodNodes.get(mid);

        // collect AST for the method structure
        ms.code = code;

        ms.ast = n.accept(astVisitor, new ExtractASTVisitor.Context());

        BytecodeEncoder bcEncoder = new BytecodeEncoder();
        bcEncoder.getMethodInsns = this::getMethodInsns;
        bcEncoder.resolveFieldInAccessMethod = false;
        bcEncoder.resolveInvokeDynamic = false;
        ms.bytecode = bcEncoder.encode(mn.instructions);
        ms.bytecode.coalesce();
    }

    public Optional<InsnList> getMethodInsns(Triple<String, String, String> mspec) {
        String owner = mspec.getLeft();
        String name = mspec.getMiddle();
        String desc = mspec.getRight();

        int cid = JointCollector.name2cid.getOrDefault(BytecodeUtils.i2qName(owner), -2);
        if (cid < 0 || !JointCollector.cid2sign2mid.containsKey(cid)) {
            return Optional.empty();
        }
        int mid = JointCollector.cid2sign2mid.get(cid)
            .getOrDefault(name + BytecodeUtils.i2qMethodDesc(desc), -2);
        if (mid < 0 || !JointCollector.methodNodes.containsKey(mid)) {
            return Optional.empty();
        }
        MethodNode mn = JointCollector.methodNodes.get(mid);
        return Optional.of(mn.instructions);
    }
}
