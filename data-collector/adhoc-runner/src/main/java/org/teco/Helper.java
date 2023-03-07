package org.teco;

import java.io.FileWriter;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.Set;

public class Helper {

    public static final int MAX_DEPTH = 2;
    public static final String OBJ_VALUE_NOTNULL = "NOTNULL";
    public static final String OBJ_VALUE_NULL = "NULL";
    public static final int MAX_STR_LENGTH = 100;

    public static void logToFile(String logPath, int stmtNo, String prefix, String msg) {
        try {
            FileWriter writer = new FileWriter(logPath + "/" + prefix + "-" + stmtNo, true);
            writer.write(msg);
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    public static void logVarDepth(String name, Object var, String logPath, int stmtNo, int depth) {
        String type;
        Class<?> clz = null;
        boolean isCompound = false;
        if (var == null) {
            type = "null";
        } else {
            clz = var.getClass();
            switch (clz.getName()) {
                case "java.lang.Boolean":
                    type = "boolean";
                    break;
                case "java.lang.Byte":
                    type = "byte";
                    break;
                case "java.lang.Character":
                    type = "char";
                    break;
                case "java.lang.Double":
                    type = "double";
                    break;
                case "java.lang.Float":
                    type = "float";
                    break;
                case "java.lang.Integer":
                    type = "int";
                    break;
                case "java.lang.Long":
                    type = "long";
                    break;
                case "java.lang.Short":
                    type = "short";
                    break;
                case "java.lang.String":
                    type = "String";
                    break;
                default:
                    type = var.getClass().getSimpleName();
                    isCompound = true;
                    break;
            }
        }

        if (!isCompound) {
            // simple or null type: directly log the value
            logToFile(logPath, stmtNo, "typevalue", name + " " + type + " " + valueOf(var) + "\n");
        } else {
            // not-null compound type: log "not-null", then visit children if depth allowing
            logToFile(
                logPath, stmtNo, "typevalue", name + " " + type + " " + OBJ_VALUE_NOTNULL + "\n");
            if (depth < MAX_DEPTH) {
                // visit children
                // TODO: array won't have fields, need to handle separately
                Set<String> seenFldNames = new java.util.HashSet<>();
                while (clz != Object.class) {
                    for (Field fld : clz.getDeclaredFields()) {
                        String fldName = fld.getName();
                        switch (fldName) {
                            case "this$0":
                            case "serialVersionUID":
                            case "$assertionsEnabled":
                            case "$assertionsDisabled":
                                // skip some useless fields
                                continue;
                        }
                        if (Modifier.isStatic(fld.getModifiers())) {
                            // skip static fields
                            continue;
                        }
                        if (seenFldNames.contains(fld.getName())) {
                            continue;
                        }

                        try {
                            fld.setAccessible(true);
                            logVarDepth(
                                name + "." + fldName, fld.get(var), logPath, stmtNo, depth + 1);
                            seenFldNames.add(fldName);
                        } catch (SecurityException | IllegalAccessException e) {
                            continue;
                        }
                    }
                    // also consider fields defined in super class (but skip the fields with the same names)
                    clz = clz.getSuperclass();
                }
            }
        }
    }

    public static String valueOf(Object obj) {
        if (obj == null) {
            return OBJ_VALUE_NULL;
        } else if (obj instanceof String) {
            String s = "\"" + ((String) obj).replace("\\", "\\\\").replace("\n", "\\n")
                .replace("\r", "\\r").replace("\"", "\\\"") + "\"";
            if (s.length() > MAX_STR_LENGTH) {
                // a very long string will be truncated to: "blabla..."
                s = s.substring(0, MAX_STR_LENGTH - 4) + "..." + s.substring(s.length() - 1);
            }
            return s;
        } else if (obj instanceof Boolean) {
            return String.valueOf((Boolean) obj);
        } else if (obj instanceof Byte) {
            return String.valueOf((Byte) obj);
        } else if (obj instanceof Character) {
            char c = (Character) obj;
            if (c == '\'') {
                return "'\\''";
            } else if (c == '\n') {
                return "'\\n'";
            } else if (c == '\r') {
                return "'\\r'";
            } else {
                return "'" + c + "'";
            }
        } else if (obj instanceof Double) {
            return String.valueOf((Double) obj);
        } else if (obj instanceof Float) {
            return String.valueOf((Float) obj);
        } else if (obj instanceof Integer) {
            return String.valueOf((Integer) obj);
        } else if (obj instanceof Long) {
            return String.valueOf((Long) obj);
        } else if (obj instanceof Short) {
            return String.valueOf((Short) obj);
        } else {
            return OBJ_VALUE_NOTNULL;
        }
    }

}
