package org.teco.util;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.function.Consumer;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import java.util.stream.Collectors;
import org.teco.AllCollectors;
import com.google.common.io.ByteStreams;

public class ClassFileFinder {

    @FunctionalInterface
    public interface Visitor {
        /**
         * Visits one class file with class name.
         */
        void visit(String className, byte[] classFile);
    }

    public static void findClasses(String classpath, Visitor visitor) {
        findClasses(classpath, visitor, false);
    }

    public static void findClassesParallel(String classpath, Visitor visitor) {
        findClasses(classpath, visitor, true);
    }

    public static void findClasses(String classpath, Visitor visitor, boolean parallel) {
        List<String> cpComponents = Arrays.asList(classpath.split(File.pathSeparator));

        Consumer<String> function = cpComponent -> {
            File cpComponentFile = Paths.get(cpComponent).toFile();
            if (cpComponentFile.isDirectory()) {
                findClassesBinRoot(cpComponentFile, visitor, parallel);
            } else if (cpComponentFile.isFile()) {
                findClassesJar(cpComponentFile, visitor, parallel);
            } else {
                AllCollectors.warning("classpath " + cpComponent + " doesn't exist");
            }
        };

        if (parallel) {
            ParallelUtils.parallelForEach(cpComponents, function);
        } else {
            cpComponents.forEach(function);
        }
    }

    public static void findClassesBinRoot(File binRoot, Visitor visitor, boolean parallel) {
        Path binRootPath = binRoot.toPath();
        try {
            List<Path> allClassFiles = Files.walk(binRootPath)
                .filter(
                    p -> p.toFile().isFile()
                        && p.getFileName().toString().toLowerCase().endsWith(".class"))
                .sorted().collect(Collectors.toList());

            Consumer<Path> function = classFilePath -> {
                try {
                    String className = binRootPath.relativize(classFilePath).toString();
                    className = className.substring(0, className.lastIndexOf(".class"));
                    className = className.replace(File.separator, ".");
                    byte[] classFileContent = Files.readAllBytes(classFilePath);
                    visitor.visit(className, classFileContent);
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            };

            if (parallel) {
                ParallelUtils.parallelForEach(allClassFiles, function);
            } else {
                allClassFiles.forEach(function);
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static void findClassesJar(File jarFile, Visitor visitor, boolean parallel) {
        try (JarFile jar = new JarFile(jarFile)) {
            Consumer<JarEntry> function = je -> {
                String name = je.getName();
                int extIndex = name.lastIndexOf(".class");
                if (extIndex > 0) {
                    String className = name.substring(0, extIndex).replace(File.separator, ".");
                    byte[] classFileContent = null;
                    try (InputStream is = jar.getInputStream(je)) {
                        classFileContent = ByteStreams.toByteArray(is);
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                    visitor.visit(className, classFileContent);
                }
            };

            if (parallel) {
                ParallelUtils.parallelForEach(Collections.list(jar.entries()), function);
            } else {
                jar.stream().forEach(function);
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
