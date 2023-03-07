package org.teco.joint;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.objectweb.asm.ClassReader;
import org.objectweb.asm.tree.ClassNode;
import org.objectweb.asm.tree.MethodNode;
import org.teco.AllCollectors;
import org.teco.joint.ClassStructure.Scope;
import org.teco.util.ASMUtils;
import org.teco.util.BytecodeUtils;
import org.teco.util.ClassFileFinder;
import org.teco.util.MethodBytecode;
import org.teco.util.ParallelUtils;
import org.teco.util.TypeResolver;
import com.github.javaparser.ParseResult;
import com.github.javaparser.ParserConfiguration;
import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.utils.SourceRoot;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;
import com.google.gson.stream.JsonWriter;

public class JointCollector {

    public static Gson GSON;
    static {
        // Prepare gson serializer/deserializer
        GsonBuilder gsonBuilder = new GsonBuilder().disableHtmlEscaping().serializeNulls()
            .registerTypeAdapter(ClassStructure.class, ClassStructure.sSerializer)
            .registerTypeAdapter(ClassStructure.class, ClassStructure.sDeserializer)
            .registerTypeAdapter(MethodStructure.class, MethodStructure.sSerializer)
            .registerTypeAdapter(MethodStructure.class, MethodStructure.sDeserializer)
            .registerTypeAdapter(FieldStructure.class, FieldStructure.sSerializer)
            .registerTypeAdapter(FieldStructure.class, FieldStructure.sDeserializer)
            .registerTypeAdapter(AST.class, AST.sSerializer)
            .registerTypeAdapter(MethodBytecode.class, MethodBytecode.sSerializer)
            .registerTypeAdapter(CallGraph.class, CallGraph.sSerializer);

        GSON = gsonBuilder.create();
    }

    static List<ClassStructure> classes = new LinkedList<>();
    static int baseCid = 0;
    static List<MethodStructure> methods = new LinkedList<>();
    static int baseMid = 0;
    static List<FieldStructure> fields = new LinkedList<>();
    static int baseFid = 0;
    static CallGraph cg = null;

    // class nodes and method nodes for app/test classes
    static Map<Integer, ClassNode> classNodes = new HashMap<>();
    static Map<Integer, MethodNode> methodNodes = new HashMap<>();

    // indexed class nodes (FQ name -> class id)
    static Map<String, Integer> name2cid = new HashMap<>();
    // indexed method nodes (class id -> method signature "name retType(paramTypes)" -> method id)
    static Map<Integer, Map<String, Integer>> cid2sign2mid = new HashMap<>();

    public static void collectJRE() {
        // scan all class files (jre), and create class/method/field structures
        ClassFileFinder.findClasses(AllCollectors.sConfig.jreClassPath, (className, classFile) -> {
            if (className.startsWith("java.") || className.startsWith("javax.")
                || className.startsWith("org.")) {
                // only scan java.*, javax.*, org.*
                scanClass(classFile, Scope.JRE);
            }
        });

        for (int cid : classNodes.keySet()) {
            ByteCodeAnalyzer.ofCid(cid).fillClassRelations();
        }

        // save all structures
        save();
    }

    public static void collect() {
        // enable javaparser logging to stdout/err
        // com.github.javaparser.utils.Log.setAdapter(new com.github.javaparser.utils.Log.StandardOutStandardErrorAdapter());

        // turn off javaparser validation
        StaticJavaParser.getConfiguration().setLanguageLevel(ParserConfiguration.LanguageLevel.RAW);

        // long time = System.currentTimeMillis();

        // load jre data
        loadJREData();

        // AllCollectors.debug("[PROFILING] loadJRE " + (System.currentTimeMillis() - time));
        // time = System.currentTimeMillis();

        // scan all class files (app/test/lib), and create class/method/field structures
        ClassFileFinder
            .findClassesParallel(AllCollectors.sConfig.testClassPath, (className, classFile) -> {
                scanClass(classFile, Scope.TEST);
            });
        ClassFileFinder
            .findClassesParallel(AllCollectors.sConfig.appClassPath, (className, classFile) -> {
                scanClass(classFile, Scope.APP);
            });
        ClassFileFinder.findClassesParallel(
            AllCollectors.sConfig.dependencyClassPath, (className, classFile) -> {
                scanClass(classFile, Scope.LIB);
            });

        // AllCollectors.debug("[PROFILING] scanClass " + (System.currentTimeMillis() - time));
        // time = System.currentTimeMillis();

        ParallelUtils.parallelForEach(
            classNodes.keySet(), cid -> ByteCodeAnalyzer.ofCid(cid).fillClassRelations());

        // AllCollectors
        // .debug("[PROFILING] fillClassRelations " + (System.currentTimeMillis() - time));
        // time = System.currentTimeMillis();

        // now we can delete lib class nodes
        Set<Integer> cids = new HashSet<>(classNodes.keySet());
        for (int cid : cids) {
            ClassStructure cs = classes.get(cid);
            if (cs.scope == Scope.LIB) {
                classNodes.remove(cid);
            }
        }

        // AllCollectors
        // .debug("[PROFILING] removeLibClassNodes " + (System.currentTimeMillis() - time));
        // time = System.currentTimeMillis();

        // scan all source files (app/test), and fill in code & AST of methods
        TypeResolver.setup();
        for (String srcRoot : AllCollectors.sConfig.testSrcPath.split(File.pathSeparator)) {
            scanSrcRoot(srcRoot);
        }
        for (String srcRoot : AllCollectors.sConfig.appSrcPath.split(File.pathSeparator)) {
            scanSrcRoot(srcRoot);
        }

        // AllCollectors.debug("[PROFILING] scanSrc " + (System.currentTimeMillis() - time));
        // time = System.currentTimeMillis();

        // collect call graph (for detecting focal methods)
        CGAnalyzer cgAnalyzer = new CGAnalyzer();
        cgAnalyzer.analyze();
        cg = cgAnalyzer.cg;

        // AllCollectors.debug("[PROFILING] analyzeCG " + (System.currentTimeMillis() - time));
        // time = System.currentTimeMillis();

        // save all structures, that are new to this project
        save();

        // AllCollectors.debug("[PROFILING] save " + (System.currentTimeMillis() - time));
        // time = System.currentTimeMillis();
    }

    private static void loadJREData() {
        assert baseCid == 0;
        assert baseMid == 0;
        assert baseFid == 0;
        assert classes.size() == 0;
        assert methods.size() == 0;
        assert fields.size() == 0;

        try (BufferedReader br = Files.newBufferedReader(getJREDataPath("class"))) {
            classes.addAll(GSON.fromJson(br, new TypeToken<List<ClassStructure>>() {}.getType()));
            baseCid = classes.size();
            classes.forEach(c -> name2cid.put(c.name, c.id));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        try (BufferedReader br = Files.newBufferedReader(getJREDataPath("method"))) {
            methods.addAll(GSON.fromJson(br, new TypeToken<List<MethodStructure>>() {}.getType()));
            baseMid = methods.size();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        try (BufferedReader br = Files.newBufferedReader(getJREDataPath("field"))) {
            fields.addAll(GSON.fromJson(br, new TypeToken<List<FieldStructure>>() {}.getType()));
            baseFid = fields.size();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static void scanClass(byte[] classFile, Scope scope) {
        // read to class node
        ClassReader cr = new ClassReader(classFile);
        ClassNode cn = new ClassNode();
        cr.accept(cn, 0);

        if ((scope == Scope.LIB && ASMUtils.isPrivate(cn.access))
            || (scope == Scope.JRE && !ASMUtils.isPublic(cn.access))) {
            // skip private classes for lib; non-public classes for jre
            return;
        }

        String name = BytecodeUtils.i2qName(cn.name);
        if (name2cid.containsKey(name)) {
            // this method already exists earilier in the classpath
            return;
        }

        // create class structure
        ClassStructure cs = new ClassStructure();
        addClassStructure(cs);
        cs.scope = scope;

        // cache class node
        synchronized (classNodes) {
            classNodes.put(cs.id, cn);
        }

        // scan this class
        ByteCodeAnalyzer analyzer = new ByteCodeAnalyzer(cs, cn);
        analyzer.scan();
    }

    static int addClassStructure(ClassStructure cs) {
        synchronized (classes) {
            int id = classes.size();
            cs.id = id;
            classes.add(cs);
        }
        return cs.id;
    }

    static int addMethodStructure(MethodStructure ms) {
        synchronized (methods) {
            int id = methods.size();
            ms.id = id;
            methods.add(ms);
        }
        return ms.id;
    }

    static int addFieldStructure(FieldStructure fs) {
        synchronized (fields) {
            int id = fields.size();
            fs.id = id;
            fields.add(fs);
        }
        return fs.id;
    }

    static void scanSrcRoot(String srcRoot) {
        Path srcRootPath = Paths.get(srcRoot);
        if (!Files.isDirectory(srcRootPath)) {
            AllCollectors.warning("Src root doesn't exist: " + srcRoot);
            return;
        }

        try {
            SourceRoot sourceRoot =
                new SourceRoot(srcRootPath, StaticJavaParser.getConfiguration());
            sourceRoot.parse("", new ScanSrcRootCallback());
            // sourceRoot.parseParallelized(new ScanSrcRootCallback());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

    }

    static class ScanSrcRootCallback implements SourceRoot.Callback {

        @Override
        public Result process(Path localPath, Path absolutePath,
            ParseResult<CompilationUnit> result) {
            if (result.isSuccessful()) {
                CompilationUnit cu = result.getResult().get();
                SrcCodeVisitor.Context ctx = new SrcCodeVisitor.Context();
                if (localPath.getParent() == null) {
                    ctx.pName = "";
                } else {
                    ctx.pName = localPath.getParent().toString().replace(File.separator, ".");
                }
                cu.accept(new SrcCodeVisitor(), ctx);
            } else {
                AllCollectors.warning(
                    "Parsing failed for: " + localPath + ", reason: " + result.getProblems());
            }
            return Result.DONT_SAVE;
        }
    }

    private static final String OUTPUT_PREFIX = "joint";

    private static Path getJREDataPath(String section) {
        return Paths.get(AllCollectors.sConfig.jreDataPath)
            .resolve(OUTPUT_PREFIX + "." + section + ".json");
    }

    private static Path getOutputPath(String section) {
        return Paths.get(AllCollectors.sConfig.outputDir)
            .resolve(OUTPUT_PREFIX + "." + section + ".json");
    }

    public static void save() {
        try (JsonWriter jw = GSON.newJsonWriter(
            new BufferedWriter(
                new OutputStreamWriter(
                    new FileOutputStream(getOutputPath("class").toFile()), "utf-8")))) {
            GSON.toJson(
                classes.subList(baseCid, classes.size()),
                new TypeToken<List<ClassStructure>>() {}.getType(), jw);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        try (JsonWriter jw = GSON.newJsonWriter(
            new BufferedWriter(
                new OutputStreamWriter(
                    new FileOutputStream(getOutputPath("method").toFile()), "utf-8")))) {
            GSON.toJson(
                methods.subList(baseMid, methods.size()),
                new TypeToken<List<ClassStructure>>() {}.getType(), jw);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        try (JsonWriter jw = GSON.newJsonWriter(
            new BufferedWriter(
                new OutputStreamWriter(
                    new FileOutputStream(getOutputPath("field").toFile()), "utf-8")))) {
            GSON.toJson(
                fields.subList(baseFid, fields.size()),
                new TypeToken<List<ClassStructure>>() {}.getType(), jw);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        if (cg != null) {
            try (JsonWriter jw = GSON.newJsonWriter(
                new BufferedWriter(
                    new OutputStreamWriter(
                        new FileOutputStream(getOutputPath("cg").toFile()), "utf-8")))) {
                GSON.toJson(cg, CallGraph.class, jw);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }
}
