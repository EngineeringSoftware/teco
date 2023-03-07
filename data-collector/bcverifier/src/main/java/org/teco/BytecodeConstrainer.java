package org.teco;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.teco.bcverifier.BytecodeConstrainerSession;
import org.teco.bcverifier.InsnConstraint;
import com.facebook.nailgun.NGContext;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

public class BytecodeConstrainer {

    public static final Gson GSON = new GsonBuilder().disableHtmlEscaping().serializeNulls()
        .registerTypeAdapter(InsnConstraint.class, InsnConstraint.sSerializer).create();

    static Map<Integer, BytecodeConstrainerSession> constrainerSessions = new HashMap<>();

    public static void startSession(List<String> args) throws Exception {
        BytecodeConstrainerSession session;

        if (args.size() == 0) {
            session = new BytecodeConstrainerSession();
        } else if (args.size() == 3) {
            session = new BytecodeConstrainerSession(
                args.get(0), args.get(1), Integer.parseInt(args.get(2)));
        } else {
            throw new Exception("Either provide no arguments or 3 arguments: owner, desc, access");
        }

        int sessionId = session.hashCode();
        constrainerSessions.put(sessionId, session);
        // return value: session id
        System.out.println(sessionId);
    }

    public static void forkSession(List<String> args) throws Exception {
        int sessionId = Integer.parseInt(args.get(0));
        BytecodeConstrainerSession session = constrainerSessions.get(sessionId);
        if (session == null) {
            throw new RuntimeException("Session " + sessionId + " not found");
        }

        BytecodeConstrainerSession clone = session.clone();
        int cloneId = clone.hashCode();
        constrainerSessions.put(cloneId, clone);

        // return value: new session id
        System.out.println(cloneId);
    }

    public static void endSession(List<String> args) throws Exception {
        int sessionId = Integer.parseInt(args.get(0));
        if (constrainerSessions.remove(sessionId) == null) {
            System.err.println("Session " + sessionId + " not found");
        }
    }

    public static void tryStepSession(List<String> args) throws Exception {
        int sessionId = Integer.parseInt(args.get(0));
        BytecodeConstrainerSession session = constrainerSessions.get(sessionId);
        if (session == null) {
            throw new RuntimeException("Session " + sessionId + " not found");
        }

        InsnConstraint insnConstr = session.tryStep(args.subList(1, args.size()));

        // return value: insn constraint
        System.out.println(GSON.toJson(insnConstr));
    }

    public static void submitStepSession(List<String> args) throws Exception {
        int sessionId = Integer.parseInt(args.get(0));
        BytecodeConstrainerSession session = constrainerSessions.get(sessionId);
        if (session == null) {
            throw new RuntimeException("Session " + sessionId + " not found");
        }

        session.submitStep();
    }

    // nailgun entry point
    public static void nailMain(NGContext context) throws Exception {
        String[] args = context.getArgs();
        if (args.length == 0) {
            throw new RuntimeException("Usage(nailgun): BytecodeVerifier action toks");
        }

        String action = args[0];
        List<String> otherArgs = Arrays.asList(args).subList(1, args.length);

        try {
            switch (action) {
                case "start_session":
                    startSession(otherArgs);
                    return;
                case "fork_session":
                    forkSession(otherArgs);
                    return;
                case "end_session":
                    endSession(otherArgs);
                    return;
                case "try_step":
                    tryStepSession(otherArgs);
                    return;
                case "submit_step":
                    submitStepSession(otherArgs);
                    return;
            }
        } catch (Exception e) {
            System.err.println("Exception during " + action + ": " + e.getClass());
            e.printStackTrace();
            throw e;
        }
        throw new RuntimeException("unknown command: " + context.getCommand());
    }

}
