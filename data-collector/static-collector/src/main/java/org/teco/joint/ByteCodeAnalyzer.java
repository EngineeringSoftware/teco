package org.teco.joint;

import java.util.HashMap;
import java.util.Map;
import org.objectweb.asm.Type;
import org.objectweb.asm.tree.AnnotationNode;
import org.objectweb.asm.tree.ClassNode;
import org.objectweb.asm.tree.FieldNode;
import org.objectweb.asm.tree.MethodNode;
import org.teco.joint.ClassStructure.Scope;
import org.teco.util.ASMUtils;
import org.teco.util.BytecodeUtils;

public class ByteCodeAnalyzer {
    private ClassStructure cs;
    private ClassNode cn;

    public ByteCodeAnalyzer(ClassStructure cs, ClassNode cn) {
        this.cs = cs;
        this.cn = cn;
    }

    static ByteCodeAnalyzer ofCid(int cid) {
        if (!JointCollector.classNodes.containsKey(cid)) {
            throw new IllegalArgumentException("cid " + cid + " not found in classNodes");
        }
        return new ByteCodeAnalyzer(
            JointCollector.classes.get(cid), JointCollector.classNodes.get(cid));
    }

    public void scan() {
        cs.name = BytecodeUtils.i2qName(cn.name);
        synchronized (JointCollector.name2cid) {
            JointCollector.name2cid.put(cs.name, cs.id);
        }

        cs.access = cn.access;

        scanMethods();
        scanFields();
    }

    private static boolean shouldSkip(int access, Scope scope) {
        // skip private methods/fields for lib; private/package methods/fields for jre
        return ((scope == Scope.LIB && ASMUtils.isPrivate(access)) || (scope == Scope.JRE
            && !(ASMUtils.isPublic(access) || ASMUtils.isProtected(access))));
    }

    private void scanMethods() {
        // save method index for APP and TEST methods
        Map<String, Integer> sign2mid = null;
        if (cs.scope == Scope.APP || cs.scope == Scope.TEST) {
            sign2mid = new HashMap<>();
            synchronized (JointCollector.cid2sign2mid) {
                JointCollector.cid2sign2mid.put(cs.id, sign2mid);
            }

            // AllCollectors.warning("put into cid2sign2mid: cid=" + cs.id + ", class_name=" + cs.name);
        }

        for (MethodNode mn : cn.methods) {
            if (shouldSkip(mn.access, cs.scope)) {
                continue;
            }

            MethodStructure ms = new MethodStructure();
            int mid = JointCollector.addMethodStructure(ms);
            ms.clz = cs.id;
            cs.methods.add(mid);

            ms.name = mn.name;
            ms.access = mn.access;
            Type mType = Type.getType(mn.desc);
            for (Type t : mType.getArgumentTypes()) {
                ms.ptypes.add(t.getClassName());
            }
            ms.rtype = mType.getReturnType().getClassName();

            if (mn.exceptions != null) {
                for (String ttype : mn.exceptions) {
                    ms.ttypes.add(BytecodeUtils.i2qName(ttype));
                }
            }

            if (mn.visibleAnnotations != null) {
                boolean hasTestAnno = false;
                boolean hasIgnoreAnno = false;
                for (AnnotationNode an : mn.visibleAnnotations) {
                    String atype = Type.getType(an.desc).getClassName();

                    switch (atype) {
                        case "org.junit.Test": // junit 4
                        case "org.junit.jupiter.api.Test": // junit 5
                            hasTestAnno = true;
                            break;
                        case "org.junit.Ignore": // junit 4
                        case "org.junit.jupiter.api.Disabled": // junit 5
                            hasIgnoreAnno = true;
                            break;
                    }

                    ms.atypes.add(atype);
                }

                if (hasTestAnno && !hasIgnoreAnno) {
                    ms.isTest = true;
                }
            }

            // save method index for APP and TEST methods
            if (cs.scope == Scope.APP || cs.scope == Scope.TEST) {
                sign2mid.put(ms.getSign(), mid);
                synchronized (JointCollector.methodNodes) {
                    JointCollector.methodNodes.put(mid, mn);
                }
            }
        }
    }

    private void scanFields() {
        for (FieldNode fn : cn.fields) {
            if (shouldSkip(fn.access, cs.scope)) {
                continue;
            }

            FieldStructure fs = new FieldStructure();
            int fid = JointCollector.addFieldStructure(fs);
            fs.clz = cs.id;
            cs.fields.add(fid);

            fs.name = fn.name;
            fs.access = fn.access;
            fs.type = Type.getType(fn.desc).getClassName();
        }
    }

    void fillClassRelations() {
        if (cn.superName != null) {
            cs.ext = JointCollector.name2cid.getOrDefault(BytecodeUtils.i2qName(cn.superName), -2);
            if (cs.ext == -2) {
                // AllCollectors.debug("Cannot find super class " + cn.superName + " | "
                // + BytecodeUtils.i2qName(cn.superName) + " for " + cs.name);
            }
        }

        for (String intf : cn.interfaces) {
            cs.impl.add(JointCollector.name2cid.getOrDefault(BytecodeUtils.i2qName(intf), -2));
        }
    }

}
