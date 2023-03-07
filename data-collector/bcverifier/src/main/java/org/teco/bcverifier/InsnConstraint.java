package org.teco.bcverifier;

import java.util.LinkedList;
import java.util.List;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonSerializer;

public class InsnConstraint {
    // types of the values on the operand stack
    public List<String> values = new LinkedList<>();
    // types of the defined local variables
    public List<String> locals = new LinkedList<>();

    // serialization
    public static final JsonSerializer<InsnConstraint> sSerializer = getSerializer();

    public static JsonSerializer<InsnConstraint> getSerializer() {
        return (d, type, jsonSerializationContext) -> {
            JsonObject ret = new JsonObject();

            JsonArray aValues = new JsonArray();
            for (String s : d.values) {
                aValues.add(s);
            }
            ret.add("values", aValues);

            JsonArray aLocals = new JsonArray();
            for (String s : d.locals) {
                aLocals.add(s);
            }
            ret.add("locals", aLocals);

            return ret;
        };
    }
}
