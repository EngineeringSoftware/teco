package org.teco.util;

import java.lang.reflect.Field;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class ReflectionUtils {

    public static List<Field> getAllFields(Class<?> clz) {
        return getAllFieldsRecursive(clz).collect(Collectors.toList());
    }

    private static Stream<Field> getAllFieldsRecursive(Class<?> clz) {
        Stream<Field> fields = Arrays.stream(clz.getDeclaredFields());

        Class<?> superClz = clz.getSuperclass();
        if (superClz != null) {
            return Stream.concat(fields, getAllFieldsRecursive(superClz));
        }

        return fields;
    }
}
