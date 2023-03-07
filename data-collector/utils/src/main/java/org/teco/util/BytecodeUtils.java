package org.teco.util;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.stream.Collectors;
import org.apache.commons.lang3.tuple.Triple;
import org.objectweb.asm.Type;

public class BytecodeUtils {

    public static String i2qName(String internalName) {
        return internalName.replace('/', '.');
    }

    public static String q2iName(String qualifiedName) {
        return qualifiedName.replace('.', '/');
    }

    public static String i2qMethodDesc(String desc) {
        Type mType = Type.getType(desc);
        List<String> ptypes = new LinkedList<>();
        for (Type t : mType.getArgumentTypes()) {
            ptypes.add(t.getClassName());
        }
        String retType = mType.getReturnType().getClassName();
        return "(" + String.join(",", ptypes) + ")" + retType;
    }

    public static String q2iClassDesc(String desc) {
        switch (desc) {
            case "void":
                return "V";
            case "boolean":
                return "Z";
            case "byte":
                return "B";
            case "char":
                return "C";
            case "short":
                return "S";
            case "int":
                return "I";
            case "long":
                return "J";
            case "float":
                return "F";
            case "double":
                return "D";
        }
        if (desc.endsWith("[]")) {
            return "[" + q2iClassDesc(desc.substring(0, desc.length() - 2));
        } else {
            return "L" + desc.replace(".", "/") + ";";
        }
    }

    public static String q2iMethodDesc(String desc) {
        int leftPara = desc.indexOf('(');
        if (leftPara != 0) {
            throw new IllegalArgumentException("Invalid method descriptor: " + desc);
        }
        int rightPara = desc.indexOf(')');
        if (rightPara == -1) {
            throw new IllegalArgumentException("Invalid method descriptor: " + desc);
        }
        String ptypesStr = "";
        if (leftPara + 1 < rightPara) {
            List<String> ptypes = Arrays.stream(desc.substring(leftPara + 1, rightPara).split(","))
                .map(BytecodeUtils::q2iClassDesc).collect(Collectors.toList());
            ptypesStr = String.join("", ptypes);
        }

        String rtype = q2iClassDesc(desc.substring(rightPara + 1));
        return "(" + ptypesStr + ")" + rtype;
    }

    public static String assembleMethodId(String classQName, String methodName, String methodDesc) {
        return q2iName(classQName) + '.' + methodName
            + methodDesc.substring(0, methodDesc.indexOf(')') + 1);
    }

    public static Triple<String, String, Type[]> splitMethodId(String methodId) {
        String name = methodId.substring(0, methodId.indexOf('('));
        String paramTypes = methodId.substring(methodId.indexOf('('));

        String className = name.substring(0, name.lastIndexOf('.'));
        String methodName = name.substring(name.lastIndexOf('.') + 1);

        // Hack: add a "V" to the end of method id to get a descriptor
        Type methodType = Type.getMethodType(paramTypes + "V");
        return Triple.of(className, methodName, methodType.getArgumentTypes());
    }


    public static Path DEBUG_PATH = Paths.get(System.getProperty("user.dir")).resolve("debug");

    /**
     * Saves the class file buffer for debugging. Saved to {@link #DEBUG_PATH}.
     *
     * @param className the name of class
     * @param classfileBuffer the class file buffer
     */
    public static void saveClassfileBufferForDebugging(String className, byte[] classfileBuffer) {
        saveClassfileBufferForDebugging(className, classfileBuffer, DEBUG_PATH);
    }

    /**
     * Saves the class file buffer for debugging.
     *
     * @param className the name of class
     * @param classfileBuffer the class file buffer
     * @param debugPath the path for saving the class file; if null, will use {@link #DEBUG_PATH}
     */
    public static void saveClassfileBufferForDebugging(String className, byte[] classfileBuffer,
        Path debugPath) {
        if (debugPath == null) {
            debugPath = DEBUG_PATH;
        }

        try {
            if (debugPath.toFile().isDirectory() || debugPath.toFile().mkdir()) {
                java.io.DataOutputStream tmpout = new java.io.DataOutputStream(
                    new java.io.FileOutputStream(debugPath.resolve(className + ".class").toFile()));
                tmpout.write(classfileBuffer, 0, classfileBuffer.length);
                tmpout.close();
            }
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }
}
