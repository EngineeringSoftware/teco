package org.teco.bcverifier;

import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.apache.commons.lang3.tuple.Pair;
import org.objectweb.asm.Label;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.Type;
import org.objectweb.asm.tree.AbstractInsnNode;
import org.objectweb.asm.tree.FieldInsnNode;
import org.objectweb.asm.tree.IincInsnNode;
import org.objectweb.asm.tree.InsnNode;
import org.objectweb.asm.tree.IntInsnNode;
import org.objectweb.asm.tree.InvokeDynamicInsnNode;
import org.objectweb.asm.tree.JumpInsnNode;
import org.objectweb.asm.tree.LabelNode;
import org.objectweb.asm.tree.LdcInsnNode;
import org.objectweb.asm.tree.LookupSwitchInsnNode;
import org.objectweb.asm.tree.MethodInsnNode;
import org.objectweb.asm.tree.MultiANewArrayInsnNode;
import org.objectweb.asm.tree.TableSwitchInsnNode;
import org.objectweb.asm.tree.TypeInsnNode;
import org.objectweb.asm.tree.VarInsnNode;
import org.objectweb.asm.util.Printer;

/**
 * Parses a list of tokens into a list of ASM instructions.  Instantiation of this class
 * performs the parse.  The state contains information like label indices, and can be
 * used to continue parsing more tokens for the same method.
 */
public class ParseToks {

    public static class ParseError extends RuntimeException {
        public int tokIndex;
        public String tok;

        public ParseError(String message, Throwable cause) {
            super(message, cause);
        }
    }

    public static class State implements Cloneable {
        Map<Integer, LabelNode> labelNodes = new HashMap<>();
        Set<Integer> definedLabelIdxes = new HashSet<>();

        public State clone() {
            State ret = new State();
            ret.labelNodes = new HashMap<>(labelNodes);
            ret.definedLabelIdxes = new HashSet<>(definedLabelIdxes);
            return ret;
        }
    }

    public static class Ret {
        public List<AbstractInsnNode> insns = new LinkedList<>();
        public List<Integer> tok2insnIndex = new LinkedList<>();
        public List<Pair<Integer, Integer>> insn2tokRange = new LinkedList<>();
        public List<ParseError> errors = new LinkedList<>();
    }

    public static final int INVALID = -1;
    public static final int LABEL = -2;

    public static final Map<String, Integer> tok2opcode;
    static {
        Map<String, Integer> tok2opcodeTmp = new HashMap<String, Integer>();
        for (int i = 0; i < Printer.OPCODES.length; i++) {
            String opcode = Printer.OPCODES[i];
            tok2opcodeTmp.put(opcode, i);
        }
        tok2opcodeTmp.put("LABEL", LABEL);
        tok2opcode = Collections.unmodifiableMap(tok2opcodeTmp);
    }

    public ParseToks(List<String> toks, State state) {
        this.toks = toks;
        this.state = state;

        parse();
    }

    public ParseToks(List<String> toks) {
        this(toks, new State());
    }

    public Ret getReturn() {
        return ret;
    }

    public List<AbstractInsnNode> getInsns() {
        return ret.insns;
    }

    public List<ParseError> getErrors() {
        return ret.errors;
    }

    public boolean isParseSuccess() {
        return ret.errors.isEmpty();
    }

    public State getState() {
        return state;
    }

    private List<String> toks;
    private State state;
    private int cur = 0;
    private Ret ret = new Ret();

    private boolean hasNext() {
        return cur < toks.size() - 1;
    }

    private void toNext() {
        if (!hasNext()) {
            throw newError("no more tokens", null);
        } else {
            ++cur;
        }
    }

    private void toPrev() {
        if (cur == 0) {
            throw newError("cannot go before first token", null);
        } else {
            --cur;
        }
    }

    private String peekNext() {
        if (!hasNext()) {
            throw newError("no more tokens", null);
        } else {
            return toks.get(cur + 1);
        }
    }

    private LabelNode defineLabelNode(int labelIdx) {
        LabelNode labelNode =
            state.labelNodes.computeIfAbsent(labelIdx, k -> new LabelNode(new Label()));
        if (state.definedLabelIdxes.contains(labelIdx)) {
            throw newError("label " + labelIdx + " already defined", null);
        }
        state.definedLabelIdxes.add(labelIdx);
        return labelNode;
    }

    private LabelNode useLabelNode(int labelIdx) {
        LabelNode labelNode =
            state.labelNodes.computeIfAbsent(labelIdx, k -> new LabelNode(new Label()));
        return labelNode;
    }

    private ParseError newError(String message, Throwable cause) {
        ParseError err = new ParseError(message, cause);
        err.tokIndex = cur;
        err.tok = toks.get(cur);
        return err;
    }

    public void parse() {
        cur = 0;
        String tok = null;
        int opcode = INVALID;
        AbstractInsnNode insn = null;
        int insnBeg = 0;
        while (cur < toks.size()) {
            insnBeg = cur;
            tok = toks.get(cur);
            opcode = tok2opcode.getOrDefault(tok, INVALID);
            insn = null;

            try {
                if (opcode == INVALID) {
                    // drop the current token and try to move on
                    throw newError("expecting operator", null);
                } else {
                    switch (opcode) {
                        case Opcodes.NOP:
                        case Opcodes.ACONST_NULL:
                        case Opcodes.ICONST_M1:
                        case Opcodes.ICONST_0:
                        case Opcodes.ICONST_1:
                        case Opcodes.ICONST_2:
                        case Opcodes.ICONST_3:
                        case Opcodes.ICONST_4:
                        case Opcodes.ICONST_5:
                        case Opcodes.LCONST_0:
                        case Opcodes.LCONST_1:
                        case Opcodes.FCONST_0:
                        case Opcodes.FCONST_1:
                        case Opcodes.FCONST_2:
                        case Opcodes.DCONST_0:
                        case Opcodes.DCONST_1:
                        case Opcodes.IALOAD:
                        case Opcodes.LALOAD:
                        case Opcodes.FALOAD:
                        case Opcodes.DALOAD:
                        case Opcodes.AALOAD:
                        case Opcodes.BALOAD:
                        case Opcodes.CALOAD:
                        case Opcodes.SALOAD:
                        case Opcodes.IASTORE:
                        case Opcodes.LASTORE:
                        case Opcodes.FASTORE:
                        case Opcodes.DASTORE:
                        case Opcodes.AASTORE:
                        case Opcodes.BASTORE:
                        case Opcodes.CASTORE:
                        case Opcodes.SASTORE:
                        case Opcodes.POP:
                        case Opcodes.POP2:
                        case Opcodes.DUP:
                        case Opcodes.DUP_X1:
                        case Opcodes.DUP_X2:
                        case Opcodes.DUP2:
                        case Opcodes.DUP2_X1:
                        case Opcodes.DUP2_X2:
                        case Opcodes.SWAP:
                        case Opcodes.IADD:
                        case Opcodes.LADD:
                        case Opcodes.FADD:
                        case Opcodes.DADD:
                        case Opcodes.ISUB:
                        case Opcodes.LSUB:
                        case Opcodes.FSUB:
                        case Opcodes.DSUB:
                        case Opcodes.IMUL:
                        case Opcodes.LMUL:
                        case Opcodes.FMUL:
                        case Opcodes.DMUL:
                        case Opcodes.IDIV:
                        case Opcodes.LDIV:
                        case Opcodes.FDIV:
                        case Opcodes.DDIV:
                        case Opcodes.IREM:
                        case Opcodes.LREM:
                        case Opcodes.FREM:
                        case Opcodes.DREM:
                        case Opcodes.INEG:
                        case Opcodes.LNEG:
                        case Opcodes.FNEG:
                        case Opcodes.DNEG:
                        case Opcodes.ISHL:
                        case Opcodes.LSHL:
                        case Opcodes.ISHR:
                        case Opcodes.LSHR:
                        case Opcodes.IUSHR:
                        case Opcodes.LUSHR:
                        case Opcodes.IAND:
                        case Opcodes.LAND:
                        case Opcodes.IOR:
                        case Opcodes.LOR:
                        case Opcodes.IXOR:
                        case Opcodes.LXOR:
                        case Opcodes.I2L:
                        case Opcodes.I2F:
                        case Opcodes.I2D:
                        case Opcodes.L2I:
                        case Opcodes.L2F:
                        case Opcodes.L2D:
                        case Opcodes.F2I:
                        case Opcodes.F2L:
                        case Opcodes.F2D:
                        case Opcodes.D2I:
                        case Opcodes.D2L:
                        case Opcodes.D2F:
                        case Opcodes.I2B:
                        case Opcodes.I2C:
                        case Opcodes.I2S:
                        case Opcodes.LCMP:
                        case Opcodes.FCMPL:
                        case Opcodes.FCMPG:
                        case Opcodes.DCMPL:
                        case Opcodes.DCMPG:
                        case Opcodes.IRETURN:
                        case Opcodes.LRETURN:
                        case Opcodes.FRETURN:
                        case Opcodes.DRETURN:
                        case Opcodes.ARETURN:
                        case Opcodes.RETURN:
                        case Opcodes.ARRAYLENGTH:
                        case Opcodes.ATHROW:
                        case Opcodes.MONITORENTER:
                        case Opcodes.MONITOREXIT:
                            insn = parseInsn();
                            break;
                        case Opcodes.BIPUSH:
                        case Opcodes.SIPUSH:
                        case Opcodes.NEWARRAY:
                            insn = parseIntInsn();
                            break;
                        case Opcodes.LDC:
                            insn = parseLdcInsn();
                            break;
                        case Opcodes.ILOAD:
                        case Opcodes.LLOAD:
                        case Opcodes.FLOAD:
                        case Opcodes.DLOAD:
                        case Opcodes.ALOAD:
                        case Opcodes.ISTORE:
                        case Opcodes.LSTORE:
                        case Opcodes.FSTORE:
                        case Opcodes.DSTORE:
                        case Opcodes.ASTORE:
                        case Opcodes.RET:
                            insn = parseVarInsn();
                            break;
                        case Opcodes.IINC:
                            insn = parseIincInsn();
                            break;
                        case LABEL:
                            insn = parseLabel();
                            break;
                        case Opcodes.IFEQ:
                        case Opcodes.IFNE:
                        case Opcodes.IFLT:
                        case Opcodes.IFGE:
                        case Opcodes.IFGT:
                        case Opcodes.IFLE:
                        case Opcodes.IF_ICMPEQ:
                        case Opcodes.IF_ICMPNE:
                        case Opcodes.IF_ICMPLT:
                        case Opcodes.IF_ICMPGE:
                        case Opcodes.IF_ICMPGT:
                        case Opcodes.IF_ICMPLE:
                        case Opcodes.IF_ACMPEQ:
                        case Opcodes.IF_ACMPNE:
                        case Opcodes.GOTO:
                        case Opcodes.JSR:
                        case Opcodes.IFNULL:
                        case Opcodes.IFNONNULL:
                            insn = parseJumpInsn();
                            break;
                        case Opcodes.TABLESWITCH:
                            insn = parseTableSwitchInsn();
                            break;
                        case Opcodes.LOOKUPSWITCH:
                            insn = parseLookupSwitchInsn();
                            break;
                        case Opcodes.GETSTATIC:
                        case Opcodes.PUTSTATIC:
                        case Opcodes.GETFIELD:
                        case Opcodes.PUTFIELD:
                            insn = parseFieldInsn();
                            break;
                        case Opcodes.INVOKEVIRTUAL:
                        case Opcodes.INVOKESPECIAL:
                        case Opcodes.INVOKESTATIC:
                        case Opcodes.INVOKEINTERFACE:
                            insn = parseMethodInsn();
                            break;
                        case Opcodes.INVOKEDYNAMIC:
                            insn = parseInvokeDynamicInsn();
                            break;
                        case Opcodes.NEW:
                        case Opcodes.ANEWARRAY:
                        case Opcodes.CHECKCAST:
                        case Opcodes.INSTANCEOF:
                            insn = parseTypeInsn();
                            break;
                        case Opcodes.MULTIANEWARRAY:
                            insn = parseMultiANewArrayInsn();
                            break;
                        default:
                            throw newError("unknown opcode: " + opcode, null);
                    }
                }

                ret.insns.add(insn);

                // fill in holes (errored tokens) in tok2insnIndex
                int lastNumberedTok = ret.tok2insnIndex.size();
                for (int i = lastNumberedTok; i < insnBeg; ++i) {
                    ret.tok2insnIndex.add(-1);
                }
                for (int i = insnBeg; i <= cur; ++i) {
                    ret.tok2insnIndex.add(ret.insns.size());
                }

                ret.insn2tokRange.add(Pair.of(insnBeg, cur + 1));
            } catch (ParseError e) {
                ret.errors.add(e);
            } finally {
                ++cur;
            }
        }
    }

    private InsnNode parseInsn() {
        return new InsnNode(tok2opcode.get(toks.get(cur)));
    }

    private IntInsnNode parseIntInsn() {
        int opcode = tok2opcode.get(toks.get(cur));
        toNext();
        int operand = parseInt();
        return new IntInsnNode(opcode, operand);
    }

    private LdcInsnNode parseLdcInsn() {
        toNext();
        String type = toks.get(cur);

        toNext();
        Object value = null;
        switch (type) {
            case "Double":
                value = parseDouble();
                break;
            case "Float":
                value = parseFloat();
                break;
            case "Integer":
                value = parseInt();
                break;
            case "Long":
                value = parseLong();
                break;
            case "String":
                value = parseStringInLdc();
                break;
            case "Type":
                value = parseType();
                break;
            default:
                toPrev();
                throw newError("not a valid LDC type", null);
        }

        if (!peekNext().equals("LDC_END")) {
            throw newError("expecting LDC_END", null);
        }
        toNext();

        return new LdcInsnNode(value);
    }

    private String parseStringInLdc() {
        List<String> toksOfTheString = new LinkedList<>();
        while (true) {
            toksOfTheString.add(toks.get(cur));
            if (peekNext().equals("LDC_END")) {
                break;
            } else {
                toNext();
            }
        }
        return String.join(" ", toksOfTheString);
    }

    private VarInsnNode parseVarInsn() {
        int opcode = tok2opcode.get(toks.get(cur));
        toNext();
        int var = parseInt();
        return new VarInsnNode(opcode, var);
    }

    private IincInsnNode parseIincInsn() {
        toNext();
        int var = parseInt();
        toNext();
        int incr = parseInt();
        return new IincInsnNode(var, incr);
    }

    private LabelNode parseLabel() {
        toNext();
        int labelIdx = parseInt();
        return defineLabelNode(labelIdx);
    }

    private JumpInsnNode parseJumpInsn() {
        int opcode = tok2opcode.get(toks.get(cur));
        toNext();
        LabelNode label = useLabelNode(parseInt());
        return new JumpInsnNode(opcode, label);
    }

    private TableSwitchInsnNode parseTableSwitchInsn() {
        toNext();
        LabelNode dflt = useLabelNode(parseInt());

        toNext();
        int min = parseInt();

        toNext();
        int max = parseInt();

        if (min > max) {
            throw newError("min must be less than of equal to max", null);
        }
        List<LabelNode> labels = new LinkedList<>();
        for (int i = min; i <= max; ++i) {
            toNext();
            labels.add(useLabelNode(parseInt()));
        }

        return new TableSwitchInsnNode(min, max, dflt, labels.toArray(new LabelNode[0]));
    }

    private LookupSwitchInsnNode parseLookupSwitchInsn() {
        toNext();
        LabelNode dflt = useLabelNode(parseInt());

        toNext();
        int npairs = parseInt();

        if (npairs < 0) {
            throw newError("npairs must be non-negative", null);
        }

        int[] keys = new int[npairs];
        LabelNode[] labels = new LabelNode[npairs];
        for (int i = 0; i < npairs; ++i) {
            toNext();
            keys[i] = parseInt();

            toNext();
            labels[i] = useLabelNode(parseInt());
        }

        return new LookupSwitchInsnNode(dflt, keys, labels);
    }

    private FieldInsnNode parseFieldInsn() {
        int opcode = tok2opcode.get(toks.get(cur));

        toNext();
        String owner = toks.get(cur);

        toNext();
        String name = toks.get(cur);

        toNext();
        String desc = toks.get(cur);

        return new FieldInsnNode(opcode, owner, name, desc);
    }

    private MethodInsnNode parseMethodInsn() {
        int opcode = tok2opcode.get(toks.get(cur));

        toNext();
        String owner = toks.get(cur);

        toNext();
        String name = toks.get(cur);

        toNext();
        String desc = toks.get(cur);

        return new MethodInsnNode(opcode, owner, name, desc);
    }

    private InvokeDynamicInsnNode parseInvokeDynamicInsn() {
        throw newError("invokedynamic is not supported", null);
        // toNext();
        // String dmType = toks.get(cur);

        // if (dmType == "LAMBDA") {
        //     throw newError("invokedynamic lambda is not supported", null);
        // } else if (dmType == "UNKNOWN") {
        //     throw newError("invokedynamic unknown is not supported", null);
        // } else {
        //     int handleTag;
        //     switch (dmType) {
        //         case "INVOKEINTERFACE":
        //             handleTag = Opcodes.H_INVOKEINTERFACE;
        //             break;
        //         case "INVOKESPECIAL":
        //             handleTag = Opcodes.H_INVOKESPECIAL;
        //             break;
        //         case "INVOKESTATIC":
        //             handleTag = Opcodes.H_INVOKESTATIC;
        //             break;
        //         case "INVOKEVIRTUAL":
        //             handleTag = Opcodes.H_INVOKEVIRTUAL;
        //             break;
        //         case "NEWINVOKESPECIAL":
        //             handleTag = Opcodes.H_NEWINVOKESPECIAL;
        //             break;
        //         case "GETFIELD":
        //         case "PUTFIELD":
        //         case "GETSTATIC":
        //         case "PUTSTATIC":
        //             throw newError("invokedynamic containing a field insn is not supported", null);
        //         default:
        //             throw newError("not a valid invokedynamic type", null);
        //     }
        //     // TODO
        // }
        // return null;
    }

    private TypeInsnNode parseTypeInsn() {
        int opcode = tok2opcode.get(toks.get(cur));

        toNext();
        String desc = toks.get(cur);

        return new TypeInsnNode(opcode, desc);
    }

    private MultiANewArrayInsnNode parseMultiANewArrayInsn() {
        toNext();
        String desc = toks.get(cur);

        toNext();
        int dims = parseInt();

        return new MultiANewArrayInsnNode(desc, dims);
    }

    private int parseInt() {
        try {
            return Integer.parseInt(toks.get(cur));
        } catch (NumberFormatException e) {
            throw newError("not an integer", e);
        }
    }

    private double parseDouble() {
        try {
            return Double.parseDouble(toks.get(cur));
        } catch (NumberFormatException e) {
            throw newError("not a double", e);
        }
    }

    private float parseFloat() {
        try {
            return Float.parseFloat(toks.get(cur));
        } catch (NumberFormatException e) {
            throw newError("not a float", e);
        }
    }

    private long parseLong() {
        try {
            return Long.parseLong(toks.get(cur));
        } catch (NumberFormatException e) {
            throw newError("not a long", e);
        }
    }

    private Type parseType() {
        try {
            return Type.getType(toks.get(cur));
        } catch (IllegalArgumentException e) {
            throw newError("not a valid type", e);
        }
    }
}
