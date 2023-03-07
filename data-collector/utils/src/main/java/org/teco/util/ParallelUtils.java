package org.teco.util;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

public class ParallelUtils {

    public static int sNumThreads = Runtime.getRuntime().availableProcessors();

    public static <T> void parallelForEach(Iterable<T> inputs, Consumer<T> function) {
        ExecutorService exec = Executors.newFixedThreadPool(sNumThreads);
        try {
            for (T input : inputs) {
                exec.submit(() -> function.accept(input));
            }
            exec.shutdown();
            exec.awaitTermination(1800, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

}
