package org.teco.joint;

import com.google.gson.JsonDeserializer;
import com.google.gson.JsonObject;
import com.google.gson.JsonSerializer;

public class FieldStructure {
    // unique id
    public int id = -1;

    // access code
    public int access = 0;

    // pointer to the defining class
    public int clz = -1;

    // name
    public String name;

    // type (runtime)
    public String type;

    // (de)serialization
    public static final JsonSerializer<FieldStructure> sSerializer = getSerializer();

    public static JsonSerializer<FieldStructure> getSerializer() {
        return (d, type, jsonSerializationContext) -> {
            JsonObject object = new JsonObject();
            object.addProperty("id", d.id);
            object.addProperty("access", d.access);
            object.addProperty("clz", d.clz);
            object.addProperty("name", d.name);
            object.addProperty("type", d.type);
            return object;
        };
    }

    public static final JsonDeserializer<FieldStructure> sDeserializer = getDeserializer();

    public static JsonDeserializer<FieldStructure> getDeserializer() {
        return (json, type, jsonDeserializationContext) -> {
            JsonObject object = json.getAsJsonObject();

            FieldStructure d = new FieldStructure();
            d.id = object.get("id").getAsInt();
            d.access = object.get("access").getAsInt();
            d.clz = object.get("clz").getAsInt();
            d.name = object.get("name").getAsString();
            try {
                d.type = object.get("type").getAsString();
            } catch (Exception e) {
                d.type = null;
            }
            return d;
        };
    }
}
