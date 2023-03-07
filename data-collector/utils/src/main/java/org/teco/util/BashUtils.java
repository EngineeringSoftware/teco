package org.teco.util;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.stream.Collectors;

public class BashUtils {

    public static class RunResult {
        public final int exitCode;
        public final String stdout;
        public final String stderr;

        public RunResult(int exitCode, String stdout, String stderr) {
            this.exitCode = exitCode;
            this.stdout = stdout;
            this.stderr = stderr;
        }
    }

    public static RunResult run(String cmd) {
        return run(cmd, null);
    }

    public static RunResult run(String cmd, Integer expectedReturnCode) {
        try {
            Runtime rt = Runtime.getRuntime();
            String[] commands = {"/bin/bash", "-c", cmd};
            Process proc = rt.exec(commands);
            proc.waitFor();

            String stdout = new BufferedReader(new InputStreamReader(proc.getInputStream())).lines()
                .collect(Collectors.joining("\n"));
            String stderr = new BufferedReader(new InputStreamReader(proc.getErrorStream())).lines()
                .collect(Collectors.joining("\n"));
            int exitCode = proc.exitValue();

            // Check expected return code
            if (expectedReturnCode != null && exitCode != expectedReturnCode) {
                // TODO: implement print limit for stdout & stderr, dump to file if they ex
                throw new RuntimeException(
                    "Expected " + expectedReturnCode + " but returned " + exitCode
                        + " while executing bash command '" + cmd + "'.\n" + "stdout: " + stdout
                        + "\n" + "stderr: " + stderr);
            }
            return new RunResult(exitCode, stdout, stderr);
        } catch (IOException | InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    public static Path getTempDir() {
        return Paths.get(run("mktemp -d", 0).stdout.trim());
    }

    public static Path getTempFile() {
        return Paths.get(run("mktemp", 0).stdout.trim());
    }
}
