package org.teco.util;

import java.lang.reflect.Method;
import org.objectweb.asm.Type;

public class Names {

    static {
        try {
            METHOD_BOOLEAN_VALUE_OF = Boolean.class.getDeclaredMethod("valueOf", boolean.class);
            METHOD_BYTE_VALUE_OF = Byte.class.getDeclaredMethod("valueOf", byte.class);
            METHOD_CHARACTER_VALUE_OF = Character.class.getDeclaredMethod("valueOf", char.class);
            METHOD_DOUBLE_VALUE_OF = Double.class.getDeclaredMethod("valueOf", double.class);
            METHOD_FLOAT_VALUE_OF = Float.class.getDeclaredMethod("valueOf", float.class);
            METHOD_INTEGER_VALUE_OF = Integer.class.getDeclaredMethod("valueOf", int.class);
            METHOD_LONG_VALUE_OF = Long.class.getDeclaredMethod("valueOf", long.class);
            METHOD_SHORT_VALUE_OF = Short.class.getDeclaredMethod("valueOf", short.class);

            METHOD_BOOLEAN_BOOLEAN_VALUE = Boolean.class.getDeclaredMethod("booleanValue");
        } catch (NoSuchMethodException e) {
            throw new RuntimeException(e);
        }
    }

    public static final Method METHOD_BOOLEAN_VALUE_OF;
    public static final String METHOD_NAME_BOOLEAN_VALUE_OF = METHOD_BOOLEAN_VALUE_OF.getName();
    public static final String METHOD_DESC_BOOLEAN_VALUE_OF =
        Type.getMethodDescriptor(METHOD_BOOLEAN_VALUE_OF);

    public static final Method METHOD_BYTE_VALUE_OF;
    public static final String METHOD_NAME_BYTE_VALUE_OF = METHOD_BYTE_VALUE_OF.getName();
    public static final String METHOD_DESC_BYTE_VALUE_OF =
        Type.getMethodDescriptor(METHOD_BYTE_VALUE_OF);

    public static final Method METHOD_CHARACTER_VALUE_OF;
    public static final String METHOD_NAME_CHARACTER_VALUE_OF = METHOD_CHARACTER_VALUE_OF.getName();
    public static final String METHOD_DESC_CHARACTER_VALUE_OF =
        Type.getMethodDescriptor(METHOD_CHARACTER_VALUE_OF);

    public static final Method METHOD_DOUBLE_VALUE_OF;
    public static final String METHOD_NAME_DOUBLE_VALUE_OF = METHOD_DOUBLE_VALUE_OF.getName();
    public static final String METHOD_DESC_DOUBLE_VALUE_OF =
        Type.getMethodDescriptor(METHOD_DOUBLE_VALUE_OF);

    public static final Method METHOD_FLOAT_VALUE_OF;
    public static final String METHOD_NAME_FLOAT_VALUE_OF = METHOD_FLOAT_VALUE_OF.getName();
    public static final String METHOD_DESC_FLOAT_VALUE_OF =
        Type.getMethodDescriptor(METHOD_FLOAT_VALUE_OF);

    public static final Method METHOD_INTEGER_VALUE_OF;
    public static final String METHOD_NAME_INTEGER_VALUE_OF = METHOD_INTEGER_VALUE_OF.getName();
    public static final String METHOD_DESC_INTEGER_VALUE_OF =
        Type.getMethodDescriptor(METHOD_INTEGER_VALUE_OF);

    public static final Method METHOD_LONG_VALUE_OF;
    public static final String METHOD_NAME_LONG_VALUE_OF = METHOD_LONG_VALUE_OF.getName();
    public static final String METHOD_DESC_LONG_VALUE_OF =
        Type.getMethodDescriptor(METHOD_LONG_VALUE_OF);

    public static final Method METHOD_SHORT_VALUE_OF;
    public static final String METHOD_NAME_SHORT_VALUE_OF = METHOD_SHORT_VALUE_OF.getName();
    public static final String METHOD_DESC_SHORT_VALUE_OF =
        Type.getMethodDescriptor(METHOD_SHORT_VALUE_OF);

    public static final Method METHOD_BOOLEAN_BOOLEAN_VALUE;
    public static final String METHOD_NAME_BOOLEAN_BOOLEAN_VALUE =
        METHOD_BOOLEAN_BOOLEAN_VALUE.getName();
    public static final String METHOD_DESC_BOOLEAN_BOOLEAN_VALUE =
        Type.getMethodDescriptor(METHOD_BOOLEAN_BOOLEAN_VALUE);
}
