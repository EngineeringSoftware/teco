package org.teco.joint;

import java.util.LinkedList;
import java.util.List;
import com.google.gson.JsonArray;
import com.google.gson.JsonDeserializer;
import com.google.gson.JsonObject;
import com.google.gson.JsonSerializer;


public class ClassStructure {

    // unique id (inside the project)
    public int id = -1;

    // access code
    public int access = 0;

    // scope: app / test / lib / jre
    public Scope scope;

    // pointer to the super class, if any
    public int ext = -1;

    // pointer to interfaces
    public List<Integer> impl = new LinkedList<>();

    // name (runtime)
    public String name;

    // pointers to declared fields
    public List<Integer> fields = new LinkedList<>();

    // pointers to declared methods
    public List<Integer> methods = new LinkedList<>();

    public static enum Scope {
        APP, TEST, LIB, JRE
    }

    // (de)serialization
    public static final JsonSerializer<ClassStructure> sSerializer = getSerializer();

    public static JsonSerializer<ClassStructure> getSerializer() {
        return (d, type, jsonSerializationContext) -> {
            JsonObject object = new JsonObject();
            object.addProperty("id", d.id);
            object.addProperty("access", d.access);
            object.addProperty("scope", d.scope.toString());
            object.addProperty("ext", d.ext);
            object.addProperty("name", d.name);

            JsonArray array = new JsonArray();
            for (int i : d.impl) {
                array.add(i);
            }
            object.add("impl", array);

            array = new JsonArray();
            for (int i : d.fields) {
                array.add(i);
            }
            object.add("fields", array);

            array = new JsonArray();
            for (int i : d.methods) {
                array.add(i);
            }
            object.add("methods", array);

            return object;
        };
    }

    public static final JsonDeserializer<ClassStructure> sDeserializer = getDeserializer();

    public static JsonDeserializer<ClassStructure> getDeserializer() {
        return (json, type, jsonDeserializationContext) -> {
            JsonObject object = json.getAsJsonObject();
            ClassStructure d = new ClassStructure();
            d.id = object.get("id").getAsInt();
            d.scope = Scope.valueOf(object.get("scope").getAsString());
            d.ext = object.get("ext").getAsInt();
            d.name = object.get("name").getAsString();

            JsonArray array = object.getAsJsonArray("impl");
            for (int i = 0; i < array.size(); ++i) {
                d.impl.add(array.get(i).getAsInt());
            }

            array = object.getAsJsonArray("fields");
            for (int i = 0; i < array.size(); ++i) {
                d.fields.add(array.get(i).getAsInt());
            }

            array = object.getAsJsonArray("methods");
            for (int i = 0; i < array.size(); ++i) {
                d.methods.add(array.get(i).getAsInt());
            }

            return d;
        };
    }
}
