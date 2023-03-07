package org.teco.util;

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.function.Function;
import org.apache.commons.collections4.OrderedMap;
import org.apache.commons.collections4.map.LinkedMap;
import org.apache.commons.lang3.tuple.Triple;
import org.objectweb.asm.Handle;
import org.objectweb.asm.Label;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.tree.AbstractInsnNode;
import org.objectweb.asm.tree.FieldInsnNode;
import org.objectweb.asm.tree.IincInsnNode;
import org.objectweb.asm.tree.InsnList;
import org.objectweb.asm.tree.InsnNode;
import org.objectweb.asm.tree.IntInsnNode;
import org.objectweb.asm.tree.InvokeDynamicInsnNode;
import org.objectweb.asm.tree.JumpInsnNode;
import org.objectweb.asm.tree.LabelNode;
import org.objectweb.asm.tree.LdcInsnNode;
import org.objectweb.asm.tree.LineNumberNode;
import org.objectweb.asm.tree.LookupSwitchInsnNode;
import org.objectweb.asm.tree.MethodInsnNode;
import org.objectweb.asm.tree.MultiANewArrayInsnNode;
import org.objectweb.asm.tree.TableSwitchInsnNode;
import org.objectweb.asm.tree.TypeInsnNode;
import org.objectweb.asm.tree.VarInsnNode;
import org.objectweb.asm.util.Printer;

/**
 * Class for encoding bytecode instructions into a list of tokens, and extract line number table.
 */
public class BytecodeEncoder {

    /** A function from a method spec (owner internal name, method name, method descriptor) to the method's instructions if found. */
    public Function<Triple<String, String, String>, Optional<InsnList>> getMethodInsns = null;
    public boolean resolveFieldInAccessMethod = false;
    public boolean resolveInvokeDynamic = false;

    public MethodBytecode encode(List<AbstractInsnNode> instructions) {
        InsnList insnList = new InsnList();
        for (AbstractInsnNode insn : instructions) {
            insnList.add(insn);
        }
        return encode(insnList);
    }

    public MethodBytecode encode(InsnList instructions) {
        MethodBytecode bytecode = new MethodBytecode();
        // blindly include all instructions, fake or real, and memorize the labels
        Map<Integer, AbstractInsnNode> insns = new HashMap<>();
        OrderedMap<Label, Integer> labelsDef = new LinkedMap<>();
        Set<Label> labelsUse = new HashSet<>();
        int insnI = 0;
        for (AbstractInsnNode insn : instructions) {
            if (insn instanceof LabelNode) {
                labelsDef.put(((LabelNode) insn).getLabel(), insnI);
            } else if (insn instanceof JumpInsnNode) {
                labelsUse.add(((JumpInsnNode) insn).label.getLabel());
            } else if (insn instanceof LookupSwitchInsnNode) {
                labelsUse.add(((LookupSwitchInsnNode) insn).dflt.getLabel());
                for (LabelNode l : ((LookupSwitchInsnNode) insn).labels) {
                    labelsUse.add(l.getLabel());
                }
            } else if (insn instanceof TableSwitchInsnNode) {
                labelsUse.add(((TableSwitchInsnNode) insn).dflt.getLabel());
                for (LabelNode l : ((TableSwitchInsnNode) insn).labels) {
                    labelsUse.add(l.getLabel());
                }
            }
            insns.put(insnI, insn);
            ++insnI;
        }

        // number the used labels, in the order of being defined
        Map<Label, Integer> labelsNo = new HashMap<>();
        int labelI = 0;
        for (Label label : labelsDef.keySet()) {
            if (labelsUse.contains(label)) {
                labelsNo.put(label, labelI);
                ++labelI;
            }
        }

        // extract the line number table; encode other instructions
        for (Map.Entry<Integer, AbstractInsnNode> entry : insns.entrySet()) {
            AbstractInsnNode insn = entry.getValue();
            if (insn instanceof LineNumberNode) {
                LineNumberNode lnInsn = (LineNumberNode) insn;
                int pc = labelsDef.get(lnInsn.start.getLabel());
                bytecode.lnt.put(pc, lnInsn.line);
            } else if (insn instanceof LabelNode) {
                // only keep the numbered (used) label node
                LabelNode lInsn = (LabelNode) insn;
                if (labelsNo.containsKey(lInsn.getLabel())) {
                    bytecode.insns.put(entry.getKey(), encodeInsn(lInsn, labelsNo));
                }
            } else if (insn.getOpcode() >= 0) {
                bytecode.insns.put(entry.getKey(), encodeInsn(insn, labelsNo));
            }
        }

        return bytecode;
    }

    protected List<String> encodeInsn(LabelNode lInsn, Map<Label, Integer> labelsNo) {
        int labelNo = labelsNo.get(lInsn.getLabel());
        return Arrays.asList("LABEL", String.valueOf(labelNo));
    }

    protected List<String> encodeInsn(AbstractInsnNode insn, Map<Label, Integer> labelsNo) {
        switch (insn.getType()) {
            case AbstractInsnNode.INSN:
                return encodeInsn((InsnNode) insn, labelsNo);
            case AbstractInsnNode.INT_INSN:
                return encodeInsn((IntInsnNode) insn, labelsNo);
            case AbstractInsnNode.VAR_INSN:
                return encodeInsn((VarInsnNode) insn, labelsNo);
            case AbstractInsnNode.TYPE_INSN:
                return encodeInsn((TypeInsnNode) insn, labelsNo);
            case AbstractInsnNode.FIELD_INSN:
                return encodeInsn((FieldInsnNode) insn, labelsNo);
            case AbstractInsnNode.METHOD_INSN:
                return encodeInsn((MethodInsnNode) insn, labelsNo);
            case AbstractInsnNode.INVOKE_DYNAMIC_INSN:
                return encodeInsn((InvokeDynamicInsnNode) insn, labelsNo);
            case AbstractInsnNode.JUMP_INSN:
                return encodeInsn((JumpInsnNode) insn, labelsNo);
            case AbstractInsnNode.LDC_INSN:
                return encodeInsn((LdcInsnNode) insn, labelsNo);
            case AbstractInsnNode.IINC_INSN:
                return encodeInsn((IincInsnNode) insn, labelsNo);
            case AbstractInsnNode.TABLESWITCH_INSN:
                return encodeInsn((TableSwitchInsnNode) insn, labelsNo);
            case AbstractInsnNode.LOOKUPSWITCH_INSN:
                return encodeInsn((LookupSwitchInsnNode) insn, labelsNo);
            case AbstractInsnNode.MULTIANEWARRAY_INSN:
                return encodeInsn((MultiANewArrayInsnNode) insn, labelsNo);
            default:
                throw new RuntimeException(
                    "encodeInsn not implemented for " + insn.getClass().getSimpleName());
        }
    }

    protected List<String> encodeInsn(InsnNode insn, Map<Label, Integer> labelsNo) {
        String op = Printer.OPCODES[insn.getOpcode()];
        return Arrays.asList(op);
    }

    protected List<String> encodeInsn(IntInsnNode iInsn, Map<Label, Integer> labelsNo) {
        String op = Printer.OPCODES[iInsn.getOpcode()];
        return Arrays.asList(op, String.valueOf(iInsn.operand));
    }

    protected List<String> encodeInsn(VarInsnNode vInsn, Map<Label, Integer> labelsNo) {
        String op = Printer.OPCODES[vInsn.getOpcode()];
        return Arrays.asList(op, String.valueOf(vInsn.var));
    }

    protected List<String> encodeInsn(TypeInsnNode tInsn, Map<Label, Integer> labelsNo) {
        String op = Printer.OPCODES[tInsn.getOpcode()];
        return Arrays.asList(op, tInsn.desc);
    }

    protected List<String> encodeInsn(FieldInsnNode fInsn, Map<Label, Integer> labelsNo) {
        String op = Printer.OPCODES[fInsn.getOpcode()];
        return Arrays.asList(op, fInsn.owner, fInsn.name, fInsn.desc);
    }

    protected List<String> encodeInsn(MethodInsnNode mInsn, Map<Label, Integer> labelsNo) {
        List<String> encoded = new LinkedList<>();

        String op = Printer.OPCODES[mInsn.getOpcode()];
        encoded.add(op);

        String clzName = mInsn.owner;
        encoded.add(clzName);

        encoded.add(mInsn.name);

        String desc = mInsn.desc;
        encoded.add(desc);

        if (resolveFieldInAccessMethod && mInsn.name.startsWith("access$")) {
            // try to get actual field accessed behind access$XXX
            Optional<List<String>> fieldInsns =
                getFieldInAccessMethod(Triple.of(mInsn.owner, mInsn.name, mInsn.desc));
            if (fieldInsns.isPresent()) {
                encoded.addAll(fieldInsns.get());
            }
        }

        return encoded;
    }

    protected Optional<List<String>> getFieldInAccessMethod(Triple<String, String, String> mspec) {
        if (this.getMethodInsns != null) {
            return this.getMethodInsns.apply(mspec).map(insns -> {
                for (AbstractInsnNode insn : insns) {
                    // find the first field insn
                    if (insn instanceof FieldInsnNode) {
                        FieldInsnNode fInsn = (FieldInsnNode) insn;
                        String op = Printer.OPCODES[insn.getOpcode()];
                        return Arrays.asList(op, fInsn.owner, fInsn.name);
                    }
                }
                return null;
            });
        } else {
            return Optional.empty();
        }
    }

    public static final Handle HANDLE_META_FACTORY = new Handle(
        Opcodes.H_INVOKESTATIC, "java/lang/invoke/LambdaMetafactory", "metafactory",
        "(Ljava/lang/invoke/MethodHandles$Lookup;Ljava/lang/String;Ljava/lang/invoke/MethodType;Ljava/lang/invoke/MethodType;Ljava/lang/invoke/MethodHandle;Ljava/lang/invoke/MethodType;)Ljava/lang/invoke/CallSite;",
        false);

    public static final Handle HANDLE_ALT_META_FACTORY = new Handle(
        Opcodes.H_INVOKESTATIC, "java/lang/invoke/LambdaMetafactory", "altMetafactory",
        "(Ljava/lang/invoke/MethodHandles$Lookup;Ljava/lang/String;Ljava/lang/invoke/MethodType;[Ljava/lang/Object;)Ljava/lang/invoke/CallSite;",
        false);

    protected List<String> encodeInsn(InvokeDynamicInsnNode dmInsn, Map<Label, Integer> labelsNo) {
        String op = Printer.OPCODES[dmInsn.getOpcode()];

        List<String> encoded = new LinkedList<>();
        encoded.add(op);

        if (resolveInvokeDynamic
            && ((dmInsn.bsm.equals(HANDLE_META_FACTORY) && dmInsn.bsmArgs.length == 3)
                || (dmInsn.bsm.equals(HANDLE_ALT_META_FACTORY) && dmInsn.bsmArgs.length == 5
                    && (int) dmInsn.bsmArgs[3] == 5 && (int) dmInsn.bsmArgs[4] == 0)
                    && dmInsn.bsmArgs[1] instanceof Handle)) {
            // TODO: actually output dmInsn.bsmArgs[0] and dmInsn.bsmArgs[2] as well
            Handle handle = (Handle) dmInsn.bsmArgs[1];
            if (handle.getName().contains("lambda$")) {
                encoded.add("LAMBDA");
                String desc = handle.getDesc();
                encoded.add(desc);
                // add all instructions from the lambda function
                Optional<List<String>> lambdaInsns = getInsnsInLambdaMethod(
                    Triple.of(handle.getOwner(), handle.getName(), handle.getDesc()));
                if (lambdaInsns.isPresent()) {
                    encoded.addAll(lambdaInsns.get());
                }
                encoded.add("LAMBDA_END");
            } else {
                switch (handle.getTag()) {
                    case Opcodes.H_INVOKEINTERFACE:
                        encoded.add("INVOKEINTERFACE");
                        break;
                    case Opcodes.H_INVOKESPECIAL:
                        encoded.add("INVOKESPECIAL");
                        break;
                    case Opcodes.H_INVOKESTATIC:
                        encoded.add("INVOKESTATIC");
                        break;
                    case Opcodes.H_INVOKEVIRTUAL:
                        encoded.add("INVOKEVIRTUAL");
                        break;
                    case Opcodes.H_NEWINVOKESPECIAL:
                        encoded.add("NEWINVOKESPECIAL");
                        break;
                    // TODO: stop supporting field insns here if they never appear
                    case Opcodes.H_GETFIELD:
                        encoded.add("GETFIELD");
                        break;
                    case Opcodes.H_PUTFIELD:
                        encoded.add("PUTFIELD");
                        break;
                    case Opcodes.H_GETSTATIC:
                        encoded.add("GETSTATIC");
                        break;
                    case Opcodes.H_PUTSTATIC:
                        encoded.add("PUTSTATIC");
                        break;
                    default:
                        throw new RuntimeException("Unknown handle tag: " + handle.getTag());
                }
                encoded.add(handle.getOwner());
                encoded.add(handle.getName());
                encoded.add(handle.getDesc());
            }
        } else {
            encoded.add("UNKNOWN");
            encoded.add(dmInsn.bsm.toString());
            encoded.add(Arrays.toString(dmInsn.bsmArgs));
            encoded.add(dmInsn.name);
            encoded.add(dmInsn.desc);
        }

        return encoded;
    }

    protected Optional<List<String>> getInsnsInLambdaMethod(Triple<String, String, String> mspec) {
        if (this.getMethodInsns != null) {
            return this.getMethodInsns.apply(mspec).map(insns -> {
                MethodBytecode mbc = encode(insns);
                mbc.coalesce();
                List<String> lambdaInsns = new LinkedList<>();
                for (List<String> insn : mbc.insns.values()) {
                    lambdaInsns.addAll(insn);
                }
                return lambdaInsns;
            });
        } else {
            return Optional.empty();
        }
    }

    protected List<String> encodeInsn(JumpInsnNode jInsn, Map<Label, Integer> labelsNo) {
        String op = Printer.OPCODES[jInsn.getOpcode()];
        int labelNo = labelsNo.get(jInsn.label.getLabel());
        return Arrays.asList(op, String.valueOf(labelNo));
    }

    protected List<String> encodeInsn(LdcInsnNode ldcInsn, Map<Label, Integer> labelsNo) {
        String op = Printer.OPCODES[ldcInsn.getOpcode()];
        return Arrays
            .asList(op, ldcInsn.cst.getClass().getSimpleName(), ldcInsn.cst.toString(), "LDC_END");
    }

    protected List<String> encodeInsn(IincInsnNode iiInsn, Map<Label, Integer> labelsNo) {
        String op = Printer.OPCODES[iiInsn.getOpcode()];
        return Arrays.asList(op, String.valueOf(iiInsn.var), String.valueOf(iiInsn.incr));
    }

    protected List<String> encodeInsn(LookupSwitchInsnNode lsInsn, Map<Label, Integer> labelsNo) {
        String op = Printer.OPCODES[lsInsn.getOpcode()];
        List<String> encoded = new LinkedList<>();
        encoded.add(op);

        int dfltNo = labelsNo.get(lsInsn.dflt.getLabel());
        encoded.add(String.valueOf(dfltNo));

        int npairs = lsInsn.labels.size();
        encoded.add(String.valueOf(npairs));

        for (int i = 0; i < npairs; ++i) {
            int match = lsInsn.keys.get(i);
            encoded.add(String.valueOf(match));

            int offsetNo = labelsNo.get(lsInsn.labels.get(i).getLabel());
            encoded.add(String.valueOf(offsetNo));
        }

        return encoded;
    }

    protected List<String> encodeInsn(TableSwitchInsnNode tsInsn, Map<Label, Integer> labelsNo) {
        String op = Printer.OPCODES[tsInsn.getOpcode()];
        List<String> encoded = new LinkedList<>();
        encoded.add(op);

        int dfltNo = labelsNo.get(tsInsn.dflt.getLabel());
        encoded.add(String.valueOf(dfltNo));

        encoded.add(String.valueOf(tsInsn.min));
        encoded.add(String.valueOf(tsInsn.max));

        for (int i = 0; i < tsInsn.labels.size(); ++i) {
            int offsetNo = labelsNo.get(tsInsn.labels.get(i).getLabel());
            encoded.add(String.valueOf(offsetNo));
        }

        return encoded;
    }

    protected List<String> encodeInsn(MultiANewArrayInsnNode manaInsn,
        Map<Label, Integer> labelsNo) {
        String op = Printer.OPCODES[manaInsn.getOpcode()];
        return Arrays.asList(op, manaInsn.desc, String.valueOf(manaInsn.dims));
    }
}
