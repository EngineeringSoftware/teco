package org.teco.bcverifier;

import java.util.List;
import org.objectweb.asm.tree.AbstractInsnNode;
import org.objectweb.asm.tree.analysis.AnalyzerException;
import org.objectweb.asm.tree.analysis.BasicValue;

public class BytecodeConstrainerSession implements Cloneable {

    StepwiseMethodAnalyzer<BasicValue> analyzer =
        new StepwiseMethodAnalyzer<>(new BasicVerifierEx());

    ParseToks.State curParserState = new ParseToks.State();
    StepwiseMethodAnalyzer.State<BasicValue> curAnalyzerState;

    public BytecodeConstrainerSession() {
        this("org/somepackage/SomeTest", "()V", 0);
    }

    public BytecodeConstrainerSession(String owner, String desc, int access) {
        curAnalyzerState = analyzer.init(owner, desc, access);
    }

    // temp variables created after try, to be submitted
    ParseToks.State newParserState = null;
    StepwiseMethodAnalyzer.State<BasicValue> newAnalyzerState = null;

    public BytecodeConstrainerSession clone() {
        BytecodeConstrainerSession ret = new BytecodeConstrainerSession();
        ret.analyzer = this.analyzer;
        ret.curParserState = curParserState.clone();
        ret.curAnalyzerState = curAnalyzerState.clone();
        ret.newParserState = newParserState == null ? null : newParserState.clone();
        ret.newAnalyzerState = newAnalyzerState == null ? null : newAnalyzerState.clone();
        return ret;
    }

    public InsnConstraint tryStep(List<String> toks) throws AnalyzerException {
        newParserState = curParserState.clone();
        ParseToks parser = new ParseToks(toks, newParserState);
        if (!parser.isParseSuccess()) {
            throw new RuntimeException("Parse error");
        }

        newAnalyzerState = curAnalyzerState.clone();
        for (AbstractInsnNode insn : parser.getInsns()) {
            newAnalyzerState = analyzer.step(newAnalyzerState, insn);
        }

        // build the constraint
        InsnConstraint constraint = new InsnConstraint();
        for (int i = 0; i < newAnalyzerState.frame.getLocals(); ++i) {
            BasicValue v = newAnalyzerState.frame.getLocal(i);
            constraint.locals.add(encodeValue(v));
        }
        for (int i = 0; i < newAnalyzerState.frame.getStackSize(); ++i) {
            BasicValue v = newAnalyzerState.frame.getStack(i);
            constraint.values.add(encodeValue(v));
        }

        return constraint;
    }

    private static String encodeValue(BasicValue v) {
        if (v == BasicValue.UNINITIALIZED_VALUE) {
            return ".";
        } else if (v == BasicValue.REFERENCE_VALUE) {
            return "R";
        } else if (v == BasicValue.RETURNADDRESS_VALUE) {
            throw new RuntimeException("Return address value is not supported");
        } else {
            return v.getType().getDescriptor();
        }
    }

    public void submitStep() {
        if (newParserState == null || newAnalyzerState == null) {
            throw new RuntimeException("Nothing to submit");
        }

        curParserState = newParserState;
        newParserState = null;
        curAnalyzerState = newAnalyzerState;
        newAnalyzerState = null;
    }
}
