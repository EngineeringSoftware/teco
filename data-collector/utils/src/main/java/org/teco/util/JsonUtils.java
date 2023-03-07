package org.teco.util;

import org.apache.commons.lang3.StringUtils;

public class JsonUtils {

    public static final int INDENTATION_INC = 2;

    public static String indent(int indentation) {
        return StringUtils.repeat(' ', indentation);
    }
}
