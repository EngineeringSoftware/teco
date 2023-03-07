package org.teco.util;

import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

public class LoggingUtils {

    private static DateTimeFormatter dateTimeFormatter = DateTimeFormatter.ISO_DATE_TIME;

    public static boolean includeTime = true;
    public static boolean includeStackTrace = false;

    public static String compose(String message) {
        StringBuilder sb = new StringBuilder();

        if (includeTime) {
            sb.append("[").append(dateTimeFormatter.format(LocalDateTime.now())).append("] ");
        }

        sb.append(message);

        if (includeStackTrace) {
            StackTraceElement[] stackTrace = Thread.currentThread().getStackTrace();
            sb.append("\n  -----STACKTRACE-----");
            // Start from the 3rd element in stack trace - discard getStackTrace and this
            // method
            for (int i = 2; i < stackTrace.length; i++) {
                sb.append("\n  ").append(stackTrace[i]);
            }
        }

        sb.append("\n");
        return sb.toString();
    }

    public static synchronized void logToFile(Path path, String message, boolean wrap) {
        if (wrap) {
            message = compose(message);
        }

        try (FileWriter fw = new FileWriter(path.toFile(), true)) {
            fw.write(message);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static void logToFile(Path path, String message) {
        logToFile(path, message, true);
    }
}
