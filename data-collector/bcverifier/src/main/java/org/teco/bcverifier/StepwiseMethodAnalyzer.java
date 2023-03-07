package org.teco.bcverifier;

import java.util.HashMap;
import java.util.Map;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.Type;
import org.objectweb.asm.tree.AbstractInsnNode;
import org.objectweb.asm.tree.JumpInsnNode;
import org.objectweb.asm.tree.LabelNode;
import org.objectweb.asm.tree.analysis.AnalyzerException;
import org.objectweb.asm.tree.analysis.Frame;
import org.objectweb.asm.tree.analysis.Interpreter;
import org.objectweb.asm.tree.analysis.Value;

/**
 * A modified version of {@link org.objectweb.asm.tree.analysis.Analyzer}, that allows 
 * the analysis of a method step by step.
 * 
 * Does not support the already deprecated JSR and RET instructions (and thus removed
 * the subroutine analysis for them).
 */
public class StepwiseMethodAnalyzer<V extends Value> implements Opcodes {

    private static final int MAX_LOCALS = 64;

    public static class State<V extends Value> implements Cloneable {
        public String owner;
        public Frame<V> frame;
        public Map<LabelNode, Frame<V>> futureLabelFrames = new HashMap<>();

        public State<V> clone() {
            State<V> ret = new State<>();
            ret.owner = owner;
            ret.frame = new Frame<V>(frame);
            ret.futureLabelFrames = new HashMap<>(futureLabelFrames);
            return ret;
        }
    }

    /** The interpreter to use to symbolically interpret the bytecode instructions. */
    private final Interpreter<V> interpreter;

    public StepwiseMethodAnalyzer(final Interpreter<V> interpreter) {
        this.interpreter = interpreter;
    }

    public State<V> init(final String owner, final String desc, final int access) {
        State<V> state = new State<>();
        state.owner = owner;
        state.frame = new Frame<>(MAX_LOCALS, -1);

        // initialize locals and returnValue
        int currentLocal = 0;
        boolean isInstanceMethod = (access & ACC_STATIC) == 0;
        if (isInstanceMethod) {
            Type ownerType = Type.getObjectType(owner);
            state.frame.setLocal(
                currentLocal,
                interpreter.newParameterValue(isInstanceMethod, currentLocal, ownerType));
            currentLocal++;
        }
        Type[] argumentTypes = Type.getArgumentTypes(desc);
        for (Type argumentType : argumentTypes) {
            state.frame.setLocal(
                currentLocal,
                interpreter.newParameterValue(isInstanceMethod, currentLocal, argumentType));
            currentLocal++;
            if (argumentType.getSize() == 2) {
                state.frame.setLocal(currentLocal, interpreter.newEmptyValue(currentLocal));
                currentLocal++;
            }
        }
        while (currentLocal < MAX_LOCALS) {
            state.frame.setLocal(currentLocal, interpreter.newEmptyValue(currentLocal));
            currentLocal++;
        }
        state.frame.setReturn(interpreter.newReturnTypeValue(Type.getReturnType(desc)));

        return state;
    }

    public State<V> step(final State<V> state, final AbstractInsnNode insn)
        throws AnalyzerException {
        State<V> ret = state.clone();

        int insnType = insn.getType();

        if (insnType == AbstractInsnNode.LINE || insnType == AbstractInsnNode.FRAME) {
            return ret;
        } else if (insnType == AbstractInsnNode.LABEL) {
            ret.frame = state.futureLabelFrames.getOrDefault((LabelNode) insn, ret.frame);
        } else {
            try {
                ret.frame.execute(insn, interpreter);
                if (insnType == AbstractInsnNode.JUMP_INSN) {
                    ret.futureLabelFrames.put(((JumpInsnNode) insn).label, ret.frame);
                }
            } catch (RuntimeException e) {
                throw new AnalyzerException(insn, "RuntimeError while executing instruction", e);
            }
        }
        return ret;
    }
}
