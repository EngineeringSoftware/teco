package org.teco;

import java.io.File;
import java.nio.file.Paths;
import java.util.Map;
import org.teco.joint.JointCollector;
import org.teco.util.AbstractConfig;
import org.teco.util.LoggingUtils;
import org.teco.util.Option;

public class AllCollectors {

    public static class AllCollectorsConfig extends AbstractConfig {

        // Input
        @Option
        public String appSrcPath;
        @Option
        public String testSrcPath;
        @Option
        public String appClassPath;
        @Option
        public String testClassPath;
        @Option
        public String dependencyClassPath;
        @Option
        public String jreClassPath;
        @Option
        public String jreDataPath;

        // Output
        @Option
        public String outputDir;

        // Debugging
        @Option
        public boolean debug = false;
        @Option
        public String debugPath;

        /**
         * Automatically infers and completes some config values, after loading from
         * file.
         */
        public void autoInfer() {
            // Create debugPath
            if (debugPath != null) {
                File debugPathF = Paths.get(debugPath).toFile();
                if (!debugPathF.isDirectory()) {
                    debugPathF.mkdirs();
                }
            }
        }

        /**
         * Checks if the config is ok.
         * 
         * @return true if the config is ok, false otherwise.
         */
        public boolean repOk() {
            if (outputDir == null || outputDir.equals("")) {
                return false;
            }

            return true;
        }
    }

    public static AllCollectorsConfig sConfig;
    public static Map<String, String> classPathMap;

    public static void collect() {
        debug("START COLLECTING");
        if (sConfig.jreClassPath != null) {
            JointCollector.collectJRE();
        } else {
            // MethodSrcCodeCollector.collect();
            // MethodByteCodeCollector.collect();
            // CallGraphCollector.collect();
            JointCollector.collect();
        }
        debug("FINISH COLLECTING");
    }

    public static void debug(String message) {
        if (sConfig.debug) {
            LoggingUtils.logToFile(Paths.get(sConfig.debugPath).resolve("debug.txt"), message);
        }
    }

    public static void warning(String message) {
        if (sConfig.debugPath != null) {
            LoggingUtils.logToFile(Paths.get(sConfig.debugPath).resolve("warning.txt"), message);
        }
    }

    public static void main(String[] args) {
        if (args.length != 1) {
            System.err.println("Exactly one argument, the path to the json config, is required");
            System.exit(-1);
        }

        sConfig = AbstractConfig.load(Paths.get(args[0]), AllCollectorsConfig.class);
        collect();
    }
}
