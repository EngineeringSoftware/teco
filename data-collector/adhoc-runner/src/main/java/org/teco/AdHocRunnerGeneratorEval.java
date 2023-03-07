package org.teco;

import java.io.FileWriter;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import com.facebook.nailgun.NGContext;
import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;

public class AdHocRunnerGeneratorEval {

    public void generate(String srcPath, String className, String methodName, String outPath,
        int predicting) throws Exception {
        AdHocRunnerGeneratorEvalModifier.Context context =
            new AdHocRunnerGeneratorEvalModifier.Context();
        context.className = className;
        context.methodName = methodName;
        context.predicting = predicting;

        AdHocRunnerGeneratorEvalModifier visitor = new AdHocRunnerGeneratorEvalModifier();
        CompilationUnit cu = StaticJavaParser.parse(Paths.get(srcPath));
        cu = (CompilationUnit) cu.accept(visitor, context);
        FileWriter writer = new FileWriter(outPath);
        writer.write(cu.toString());
        writer.close();
    }

    // nailgun entry point

    static AdHocRunnerGeneratorEval instance = new AdHocRunnerGeneratorEval();

    public static void nailMain(NGContext context) throws Exception {
        String[] args = context.getArgs();
        if (args.length == 0) {
            throw new RuntimeException("Usage(nailgun): AcHocRunnerGenerator action args");
        }

        String action = args[0];
        List<String> otherArgs = Arrays.asList(args).subList(1, args.length);

        try {
            switch (action) {
                case "generate":
                    instance.generate(
                        otherArgs.get(0), otherArgs.get(1), otherArgs.get(2), otherArgs.get(3),
                        Integer.valueOf(otherArgs.get(4)));
                    return;
            }
        } catch (Exception e) {
            System.err.println("Exception during " + action + ": " + e.getClass());
            e.printStackTrace();
            throw e;
        }
        throw new RuntimeException("unknown command: " + context.getCommand());
    }

    public static void main(String... args) throws Exception {
        instance.generate(args[0], args[1], args[2], args[3], Integer.valueOf(args[4]));
    }
}
