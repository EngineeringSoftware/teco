package org.teco.bcverifier;

import java.util.List;
import org.objectweb.asm.ConstantDynamic;
import org.objectweb.asm.Handle;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.Type;
import org.objectweb.asm.tree.AbstractInsnNode;
import org.objectweb.asm.tree.FieldInsnNode;
import org.objectweb.asm.tree.IntInsnNode;
import org.objectweb.asm.tree.InvokeDynamicInsnNode;
import org.objectweb.asm.tree.LdcInsnNode;
import org.objectweb.asm.tree.MethodInsnNode;
import org.objectweb.asm.tree.MultiANewArrayInsnNode;
import org.objectweb.asm.tree.TypeInsnNode;
import org.objectweb.asm.tree.analysis.AnalyzerException;
import org.objectweb.asm.tree.analysis.BasicValue;
import org.objectweb.asm.tree.analysis.Interpreter;
import org.objectweb.asm.tree.analysis.Value;

/**
 * Similar to {@link org.objectweb.asm.tree.analysis.BasicVerifier}, but report the accurate type of reference types if possible.
 * But unlike {@link org.objectweb.asm.tree.analysis.SimpleVerifier}, we don't reason about the relations between classes, thus do not require loading those classes.
 */
public class BasicVerifierEx extends Interpreter<BasicValue> implements Opcodes {

    /**
     * Constructs a new {@link BasicVerifierEx}. <i>Subclasses must not use this constructor</i>.
     * Instead, they must use the {@link #BasicVerifierEx(int)} version.
     */
    public BasicVerifierEx() {
        this(/* latest api = */ ASM9);
        if (getClass() != BasicVerifierEx.class) {
            throw new IllegalStateException();
        }
    }

    /**
     * Constructs a new {@link BasicVerfierEx}.
     *
     * @param api the ASM API version supported by this verifier. Must be one of {@link
     *     org.objectweb.asm.Opcodes#ASM4}, {@link org.objectweb.asm.Opcodes#ASM5}, {@link
     *     org.objectweb.asm.Opcodes#ASM6}, {@link org.objectweb.asm.Opcodes#ASM7}, {@link
     *     org.objectweb.asm.Opcodes#ASM8} or or {@link org.objectweb.asm.Opcodes#ASM9}.
     */
    protected BasicVerifierEx(final int api) {
        super(api);
    }

    /**
    * Special type used for the {@literal null} literal. This is an object reference type with
    * descriptor 'Lnull;'.
    */
    public static final Type NULL_TYPE = Type.getObjectType("null");

    @Override
    public BasicValue newValue(final Type type) {
        if (type == null) {
            return BasicValue.UNINITIALIZED_VALUE;
        }
        switch (type.getSort()) {
            case Type.VOID:
                return null;
            case Type.BOOLEAN:
            case Type.CHAR:
            case Type.BYTE:
            case Type.SHORT:
            case Type.INT:
                return BasicValue.INT_VALUE;
            case Type.FLOAT:
                return BasicValue.FLOAT_VALUE;
            case Type.LONG:
                return BasicValue.LONG_VALUE;
            case Type.DOUBLE:
                return BasicValue.DOUBLE_VALUE;
            case Type.ARRAY:
            case Type.OBJECT:
                return new BasicValue(type);
            default:
                throw new AssertionError();
        }
    }

    @Override
    public BasicValue newOperation(final AbstractInsnNode insn) throws AnalyzerException {
        switch (insn.getOpcode()) {
            case ACONST_NULL:
                return newValue(NULL_TYPE);
            case ICONST_M1:
            case ICONST_0:
            case ICONST_1:
            case ICONST_2:
            case ICONST_3:
            case ICONST_4:
            case ICONST_5:
                return BasicValue.INT_VALUE;
            case LCONST_0:
            case LCONST_1:
                return BasicValue.LONG_VALUE;
            case FCONST_0:
            case FCONST_1:
            case FCONST_2:
                return BasicValue.FLOAT_VALUE;
            case DCONST_0:
            case DCONST_1:
                return BasicValue.DOUBLE_VALUE;
            case BIPUSH:
            case SIPUSH:
                return BasicValue.INT_VALUE;
            case LDC:
                Object value = ((LdcInsnNode) insn).cst;
                if (value instanceof Integer) {
                    return BasicValue.INT_VALUE;
                } else if (value instanceof Float) {
                    return BasicValue.FLOAT_VALUE;
                } else if (value instanceof Long) {
                    return BasicValue.LONG_VALUE;
                } else if (value instanceof Double) {
                    return BasicValue.DOUBLE_VALUE;
                } else if (value instanceof String) {
                    return newValue(Type.getObjectType("java/lang/String"));
                } else if (value instanceof Type) {
                    int sort = ((Type) value).getSort();
                    if (sort == Type.OBJECT || sort == Type.ARRAY) {
                        return newValue(Type.getObjectType("java/lang/Class"));
                    } else if (sort == Type.METHOD) {
                        return newValue(Type.getObjectType("java/lang/invoke/MethodType"));
                    } else {
                        throw new AnalyzerException(insn, "Illegal LDC value " + value);
                    }
                } else if (value instanceof Handle) {
                    return newValue(Type.getObjectType("java/lang/invoke/MethodHandle"));
                } else if (value instanceof ConstantDynamic) {
                    return newValue(Type.getType(((ConstantDynamic) value).getDescriptor()));
                } else {
                    throw new AnalyzerException(insn, "Illegal LDC value " + value);
                }
            case JSR:
                return BasicValue.RETURNADDRESS_VALUE;
            case GETSTATIC:
                return newValue(Type.getType(((FieldInsnNode) insn).desc));
            case NEW:
                return newValue(Type.getObjectType(((TypeInsnNode) insn).desc));
            default:
                throw new AssertionError();
        }
    }

    public BasicValue _unaryOperation(final AbstractInsnNode insn, final BasicValue value)
        throws AnalyzerException {
        switch (insn.getOpcode()) {
            case INEG:
            case IINC:
            case L2I:
            case F2I:
            case D2I:
            case I2B:
            case I2C:
            case I2S:
                return BasicValue.INT_VALUE;
            case FNEG:
            case I2F:
            case L2F:
            case D2F:
                return BasicValue.FLOAT_VALUE;
            case LNEG:
            case I2L:
            case F2L:
            case D2L:
                return BasicValue.LONG_VALUE;
            case DNEG:
            case I2D:
            case L2D:
            case F2D:
                return BasicValue.DOUBLE_VALUE;
            case IFEQ:
            case IFNE:
            case IFLT:
            case IFGE:
            case IFGT:
            case IFLE:
            case TABLESWITCH:
            case LOOKUPSWITCH:
            case IRETURN:
            case LRETURN:
            case FRETURN:
            case DRETURN:
            case ARETURN:
            case PUTSTATIC:
                return null;
            case GETFIELD:
                return newValue(Type.getType(((FieldInsnNode) insn).desc));
            case NEWARRAY:
                switch (((IntInsnNode) insn).operand) {
                    case T_BOOLEAN:
                        return newValue(Type.getType("[Z"));
                    case T_CHAR:
                        return newValue(Type.getType("[C"));
                    case T_BYTE:
                        return newValue(Type.getType("[B"));
                    case T_SHORT:
                        return newValue(Type.getType("[S"));
                    case T_INT:
                        return newValue(Type.getType("[I"));
                    case T_FLOAT:
                        return newValue(Type.getType("[F"));
                    case T_DOUBLE:
                        return newValue(Type.getType("[D"));
                    case T_LONG:
                        return newValue(Type.getType("[J"));
                    default:
                        break;
                }
                throw new AnalyzerException(insn, "Invalid array type");
            case ANEWARRAY:
                return newValue(Type.getType("[" + Type.getObjectType(((TypeInsnNode) insn).desc)));
            case ARRAYLENGTH:
                return BasicValue.INT_VALUE;
            case ATHROW:
                return null;
            case CHECKCAST:
                return newValue(Type.getObjectType(((TypeInsnNode) insn).desc));
            case INSTANCEOF:
                return BasicValue.INT_VALUE;
            case MONITORENTER:
            case MONITOREXIT:
            case IFNULL:
            case IFNONNULL:
                return null;
            default:
                throw new AssertionError();
        }
    }

    public BasicValue _binaryOperation(final AbstractInsnNode insn, final BasicValue value1,
        final BasicValue value2) throws AnalyzerException {
        switch (insn.getOpcode()) {
            case IALOAD:
            case BALOAD:
            case CALOAD:
            case SALOAD:
            case IADD:
            case ISUB:
            case IMUL:
            case IDIV:
            case IREM:
            case ISHL:
            case ISHR:
            case IUSHR:
            case IAND:
            case IOR:
            case IXOR:
                return BasicValue.INT_VALUE;
            case FALOAD:
            case FADD:
            case FSUB:
            case FMUL:
            case FDIV:
            case FREM:
                return BasicValue.FLOAT_VALUE;
            case LALOAD:
            case LADD:
            case LSUB:
            case LMUL:
            case LDIV:
            case LREM:
            case LSHL:
            case LSHR:
            case LUSHR:
            case LAND:
            case LOR:
            case LXOR:
                return BasicValue.LONG_VALUE;
            case DALOAD:
            case DADD:
            case DSUB:
            case DMUL:
            case DDIV:
            case DREM:
                return BasicValue.DOUBLE_VALUE;
            case AALOAD:
                return BasicValue.REFERENCE_VALUE;
            case LCMP:
            case FCMPL:
            case FCMPG:
            case DCMPL:
            case DCMPG:
                return BasicValue.INT_VALUE;
            case IF_ICMPEQ:
            case IF_ICMPNE:
            case IF_ICMPLT:
            case IF_ICMPGE:
            case IF_ICMPGT:
            case IF_ICMPLE:
            case IF_ACMPEQ:
            case IF_ACMPNE:
            case PUTFIELD:
                return null;
            default:
                throw new AssertionError();
        }
    }

    public BasicValue _naryOperation(final AbstractInsnNode insn,
        final List<? extends BasicValue> values) throws AnalyzerException {
        int opcode = insn.getOpcode();
        if (opcode == MULTIANEWARRAY) {
            return newValue(Type.getType(((MultiANewArrayInsnNode) insn).desc));
        } else if (opcode == INVOKEDYNAMIC) {
            return newValue(Type.getReturnType(((InvokeDynamicInsnNode) insn).desc));
        } else {
            return newValue(Type.getReturnType(((MethodInsnNode) insn).desc));
        }
    }

    @Override
    public BasicValue merge(final BasicValue value1, final BasicValue value2) {
        if (!value1.equals(value2)) {
            Type type1 = value1.getType();
            Type type2 = value2.getType();
            if (type1.equals(NULL_TYPE)) {
                return value2;
            }
            if (type2.equals(NULL_TYPE)) {
                return value1;
            }
            if (value1.isReference() && value2.isReference()) {
                // approximate the common supertype as Object
                return newValue(Type.getObjectType("java/lang/Object"));
            }
            return BasicValue.UNINITIALIZED_VALUE;
        }
        return value1;
    }

    @Override
    public BasicValue copyOperation(final AbstractInsnNode insn, final BasicValue value)
        throws AnalyzerException {
        Value expected;
        switch (insn.getOpcode()) {
            case ILOAD:
            case ISTORE:
                expected = BasicValue.INT_VALUE;
                break;
            case FLOAD:
            case FSTORE:
                expected = BasicValue.FLOAT_VALUE;
                break;
            case LLOAD:
            case LSTORE:
                expected = BasicValue.LONG_VALUE;
                break;
            case DLOAD:
            case DSTORE:
                expected = BasicValue.DOUBLE_VALUE;
                break;
            case ALOAD:
                if (!value.isReference()) {
                    throw new AnalyzerException(insn, null, "an object reference", value);
                }
                return value;
            case ASTORE:
                if (!value.isReference() && !BasicValue.RETURNADDRESS_VALUE.equals(value)) {
                    throw new AnalyzerException(
                        insn, null, "an object reference or a return address", value);
                }
                return value;
            default:
                return value;
        }
        if (!expected.equals(value)) {
            throw new AnalyzerException(insn, null, expected, value);
        }
        return value;
    }

    @Override
    public BasicValue unaryOperation(final AbstractInsnNode insn, final BasicValue value)
        throws AnalyzerException {
        BasicValue expected;
        switch (insn.getOpcode()) {
            case INEG:
            case IINC:
            case I2F:
            case I2L:
            case I2D:
            case I2B:
            case I2C:
            case I2S:
            case IFEQ:
            case IFNE:
            case IFLT:
            case IFGE:
            case IFGT:
            case IFLE:
            case TABLESWITCH:
            case LOOKUPSWITCH:
            case IRETURN:
            case NEWARRAY:
            case ANEWARRAY:
                expected = BasicValue.INT_VALUE;
                break;
            case FNEG:
            case F2I:
            case F2L:
            case F2D:
            case FRETURN:
                expected = BasicValue.FLOAT_VALUE;
                break;
            case LNEG:
            case L2I:
            case L2F:
            case L2D:
            case LRETURN:
                expected = BasicValue.LONG_VALUE;
                break;
            case DNEG:
            case D2I:
            case D2F:
            case D2L:
            case DRETURN:
                expected = BasicValue.DOUBLE_VALUE;
                break;
            case GETFIELD:
                expected = newValue(Type.getObjectType(((FieldInsnNode) insn).owner));
                break;
            case ARRAYLENGTH:
                if (!isArrayValue(value)) {
                    throw new AnalyzerException(insn, null, "an array reference", value);
                }
                return _unaryOperation(insn, value);
            case CHECKCAST:
            case ARETURN:
            case ATHROW:
            case INSTANCEOF:
            case MONITORENTER:
            case MONITOREXIT:
            case IFNULL:
            case IFNONNULL:
                if (!value.isReference()) {
                    throw new AnalyzerException(insn, null, "an object reference", value);
                }
                return _unaryOperation(insn, value);
            case PUTSTATIC:
                expected = newValue(Type.getType(((FieldInsnNode) insn).desc));
                break;
            default:
                throw new AssertionError();
        }
        if (!isSubTypeOf(value, expected)) {
            throw new AnalyzerException(insn, null, expected, value);
        }
        return _unaryOperation(insn, value);
    }

    @Override
    public BasicValue binaryOperation(final AbstractInsnNode insn, final BasicValue value1,
        final BasicValue value2) throws AnalyzerException {
        BasicValue expected1;
        BasicValue expected2;
        switch (insn.getOpcode()) {
            case IALOAD:
                expected1 = newValue(Type.getType("[I"));
                expected2 = BasicValue.INT_VALUE;
                break;
            case BALOAD:
                if (isSubTypeOf(value1, newValue(Type.getType("[Z")))) {
                    expected1 = newValue(Type.getType("[Z"));
                } else {
                    expected1 = newValue(Type.getType("[B"));
                }
                expected2 = BasicValue.INT_VALUE;
                break;
            case CALOAD:
                expected1 = newValue(Type.getType("[C"));
                expected2 = BasicValue.INT_VALUE;
                break;
            case SALOAD:
                expected1 = newValue(Type.getType("[S"));
                expected2 = BasicValue.INT_VALUE;
                break;
            case LALOAD:
                expected1 = newValue(Type.getType("[J"));
                expected2 = BasicValue.INT_VALUE;
                break;
            case FALOAD:
                expected1 = newValue(Type.getType("[F"));
                expected2 = BasicValue.INT_VALUE;
                break;
            case DALOAD:
                expected1 = newValue(Type.getType("[D"));
                expected2 = BasicValue.INT_VALUE;
                break;
            case AALOAD:
                expected1 = newValue(Type.getType("[Ljava/lang/Object;"));
                expected2 = BasicValue.INT_VALUE;
                break;
            case IADD:
            case ISUB:
            case IMUL:
            case IDIV:
            case IREM:
            case ISHL:
            case ISHR:
            case IUSHR:
            case IAND:
            case IOR:
            case IXOR:
            case IF_ICMPEQ:
            case IF_ICMPNE:
            case IF_ICMPLT:
            case IF_ICMPGE:
            case IF_ICMPGT:
            case IF_ICMPLE:
                expected1 = BasicValue.INT_VALUE;
                expected2 = BasicValue.INT_VALUE;
                break;
            case FADD:
            case FSUB:
            case FMUL:
            case FDIV:
            case FREM:
            case FCMPL:
            case FCMPG:
                expected1 = BasicValue.FLOAT_VALUE;
                expected2 = BasicValue.FLOAT_VALUE;
                break;
            case LADD:
            case LSUB:
            case LMUL:
            case LDIV:
            case LREM:
            case LAND:
            case LOR:
            case LXOR:
            case LCMP:
                expected1 = BasicValue.LONG_VALUE;
                expected2 = BasicValue.LONG_VALUE;
                break;
            case LSHL:
            case LSHR:
            case LUSHR:
                expected1 = BasicValue.LONG_VALUE;
                expected2 = BasicValue.INT_VALUE;
                break;
            case DADD:
            case DSUB:
            case DMUL:
            case DDIV:
            case DREM:
            case DCMPL:
            case DCMPG:
                expected1 = BasicValue.DOUBLE_VALUE;
                expected2 = BasicValue.DOUBLE_VALUE;
                break;
            case IF_ACMPEQ:
            case IF_ACMPNE:
                expected1 = BasicValue.REFERENCE_VALUE;
                expected2 = BasicValue.REFERENCE_VALUE;
                break;
            case PUTFIELD:
                FieldInsnNode fieldInsn = (FieldInsnNode) insn;
                expected1 = newValue(Type.getObjectType(fieldInsn.owner));
                expected2 = newValue(Type.getType(fieldInsn.desc));
                break;
            default:
                throw new AssertionError();
        }
        if (!isSubTypeOf(value1, expected1)) {
            throw new AnalyzerException(insn, "First argument", expected1, value1);
        } else if (!isSubTypeOf(value2, expected2)) {
            throw new AnalyzerException(insn, "Second argument", expected2, value2);
        }
        if (insn.getOpcode() == AALOAD) {
            return getElementValue(value1);
        } else {
            return _binaryOperation(insn, value1, value2);
        }
    }

    @Override
    public BasicValue ternaryOperation(final AbstractInsnNode insn, final BasicValue value1,
        final BasicValue value2, final BasicValue value3) throws AnalyzerException {
        BasicValue expected1;
        BasicValue expected3;
        switch (insn.getOpcode()) {
            case IASTORE:
                expected1 = newValue(Type.getType("[I"));
                expected3 = BasicValue.INT_VALUE;
                break;
            case BASTORE:
                if (isSubTypeOf(value1, newValue(Type.getType("[Z")))) {
                    expected1 = newValue(Type.getType("[Z"));
                } else {
                    expected1 = newValue(Type.getType("[B"));
                }
                expected3 = BasicValue.INT_VALUE;
                break;
            case CASTORE:
                expected1 = newValue(Type.getType("[C"));
                expected3 = BasicValue.INT_VALUE;
                break;
            case SASTORE:
                expected1 = newValue(Type.getType("[S"));
                expected3 = BasicValue.INT_VALUE;
                break;
            case LASTORE:
                expected1 = newValue(Type.getType("[J"));
                expected3 = BasicValue.LONG_VALUE;
                break;
            case FASTORE:
                expected1 = newValue(Type.getType("[F"));
                expected3 = BasicValue.FLOAT_VALUE;
                break;
            case DASTORE:
                expected1 = newValue(Type.getType("[D"));
                expected3 = BasicValue.DOUBLE_VALUE;
                break;
            case AASTORE:
                expected1 = value1;
                expected3 = BasicValue.REFERENCE_VALUE;
                break;
            default:
                throw new AssertionError();
        }
        if (!isSubTypeOf(value1, expected1)) {
            throw new AnalyzerException(
                insn, "First argument", "a " + expected1 + " array reference", value1);
        } else if (!BasicValue.INT_VALUE.equals(value2)) {
            throw new AnalyzerException(insn, "Second argument", BasicValue.INT_VALUE, value2);
        } else if (!isSubTypeOf(value3, expected3)) {
            throw new AnalyzerException(insn, "Third argument", expected3, value3);
        }
        return null;
    }

    @Override
    public BasicValue naryOperation(final AbstractInsnNode insn,
        final List<? extends BasicValue> values) throws AnalyzerException {
        int opcode = insn.getOpcode();
        if (opcode == MULTIANEWARRAY) {
            for (BasicValue value : values) {
                if (!BasicValue.INT_VALUE.equals(value)) {
                    throw new AnalyzerException(insn, null, BasicValue.INT_VALUE, value);
                }
            }
        } else {
            int i = 0;
            int j = 0;
            if (opcode != INVOKESTATIC && opcode != INVOKEDYNAMIC) {
                Type owner = Type.getObjectType(((MethodInsnNode) insn).owner);
                if (!isSubTypeOf(values.get(i++), newValue(owner))) {
                    throw new AnalyzerException(
                        insn, "Method owner", newValue(owner), values.get(0));
                }
            }
            String methodDescriptor =
                (opcode == INVOKEDYNAMIC) ? ((InvokeDynamicInsnNode) insn).desc
                    : ((MethodInsnNode) insn).desc;
            Type[] args = Type.getArgumentTypes(methodDescriptor);
            while (i < values.size()) {
                BasicValue expected = newValue(args[j++]);
                BasicValue actual = values.get(i++);
                if (!isSubTypeOf(actual, expected)) {
                    throw new AnalyzerException(insn, "Argument " + j, expected, actual);
                }
            }
        }
        return _naryOperation(insn, values);
    }

    @Override
    public void returnOperation(final AbstractInsnNode insn, final BasicValue value,
        final BasicValue expected) throws AnalyzerException {
        if (!isSubTypeOf(value, expected)) {
            throw new AnalyzerException(insn, "Incompatible return type", expected, value);
        }
    }

    /**
     * Returns whether the given value corresponds to an array reference.
     *
     * @param value a value.
     * @return whether 'value' corresponds to an array reference.
     */
    protected boolean isArrayValue(final BasicValue value) {
        return value.isReference();
    }

    /**
     * Returns the value corresponding to the type of the elements of the given array reference value.
     *
     * @param objectArrayValue a value corresponding to array of object (or array) references.
     * @return the value corresponding to the type of the elements of 'objectArrayValue'.
     * @throws AnalyzerException if objectArrayValue does not correspond to an array type.
     */
    protected BasicValue getElementValue(final BasicValue objectArrayValue)
        throws AnalyzerException {
        return BasicValue.REFERENCE_VALUE;
    }

    /**
     * Returns whether the type corresponding to the first argument is a subtype of the type
     * corresponding to the second argument.
     *
     * @param value a value.
     * @param expected another value.
     * @return whether the type corresponding to 'value' is a subtype of the type corresponding to
     *     'expected'.
     */
    protected boolean isSubTypeOf(final BasicValue value, final BasicValue expected) {
        if (value.isReference() && expected.isReference()) {
            return true;
        } else {
            return value.equals(expected);
        }
    }
}
