package org.teco.util;

import org.objectweb.asm.Opcodes;

public class ASMUtils {

    public static int ASM_API = Opcodes.ASM7;

    public static boolean isPrivate(int access) {
        return (access & Opcodes.ACC_PRIVATE) != 0;
    }

    public static boolean isPublic(int access) {
        return (access & Opcodes.ACC_PUBLIC) != 0;
    }

    public static boolean isProtected(int access) {
        return (access & Opcodes.ACC_PROTECTED) != 0;
    }

    public static boolean isPackage(int access) {
        return (access & Opcodes.ACC_PUBLIC) == 0 && (access & Opcodes.ACC_PROTECTED) == 0
            && (access & Opcodes.ACC_PRIVATE) == 0;
    }
}
