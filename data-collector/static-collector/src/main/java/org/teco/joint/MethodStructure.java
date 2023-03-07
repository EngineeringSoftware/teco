package org.teco.joint;

import java.util.LinkedList;
import java.util.List;
import org.teco.util.MethodBytecode;
import com.google.gson.JsonArray;
import com.google.gson.JsonDeserializer;
import com.google.gson.JsonObject;
import com.google.gson.JsonSerializer;

public class MethodStructure {
    // unique id (inside the project)
    public int id = -1;

    // access code
    public int access = 0;

    // pointer to the defining class
    public int clz = -1;

    // whether this method is test
    public boolean isTest;

    // method name
    public String name;

    // parameter types (runtime)
    public List<String> ptypes = new LinkedList<>();

    // return type (runtime)
    public String rtype;

    // throw types (runtime)
    public List<String> ttypes = new LinkedList<>();

    // annotation types (runtime)
    public List<String> atypes = new LinkedList<>();

    // code (literal)
    public String code;

    // AST
    public AST ast;

    // bytecode
    public MethodBytecode bytecode;

    public String getSign() {
        return name + "(" + String.join(",", ptypes) + ")" + rtype;
    }

    // (de)serialization
    public static final JsonSerializer<MethodStructure> sSerializer = getSerializer();

    public static JsonSerializer<MethodStructure> getSerializer() {
        return (d, type, jsonSerializationContext) -> {
            JsonObject object = new JsonObject();
            object.addProperty("id", d.id);
            object.addProperty("access", d.access);
            object.addProperty("clz", d.clz);
            object.addProperty("is_test", d.isTest);
            object.addProperty("name", d.name);

            JsonArray array = new JsonArray();
            for (String s : d.ptypes) {
                array.add(s);
            }
            object.add("ptypes", array);

            object.addProperty("rtype", d.rtype);

            array = new JsonArray();
            for (String s : d.ttypes) {
                array.add(s);
            }
            object.add("ttypes", array);

            array = new JsonArray();
            for (String s : d.atypes) {
                array.add(s);
            }
            object.add("atypes", array);

            if (d.code != null) {
                object.addProperty("code", d.code);
            }

            if (d.ast != null) {
                object.add("ast", jsonSerializationContext.serialize(d.ast));
            }

            if (d.bytecode != null) {
                object.add("bytecode", jsonSerializationContext.serialize(d.bytecode));
            }

            return object;
        };
    }

    public static final JsonDeserializer<MethodStructure> sDeserializer = getDeserializer();

    public static JsonDeserializer<MethodStructure> getDeserializer() {
        return (json, type, jsonDeserializationContext) -> {
            JsonObject object = json.getAsJsonObject();

            MethodStructure d = new MethodStructure();
            d.id = object.get("id").getAsInt();
            d.access = object.get("access").getAsInt();
            d.clz = object.get("clz").getAsInt();
            try {
                d.isTest = object.get("is_test").getAsBoolean();
            } catch (UnsupportedOperationException e) {
                d.isTest = false;
            }
            try {
                d.name = object.get("name").getAsString();
            } catch (UnsupportedOperationException e) {
                d.name = null;
            }

            JsonArray array = object.getAsJsonArray("ptypes");
            for (int i = 0; i < array.size(); i++) {
                d.ptypes.add(array.get(i).getAsString());
            }

            try {
                d.rtype = object.get("rtype").getAsString();
            } catch (UnsupportedOperationException e) {
                d.rtype = null;
            }

            array = object.getAsJsonArray("ttypes");
            for (int i = 0; i < array.size(); i++) {
                d.ttypes.add(array.get(i).getAsString());
            }

            array = object.getAsJsonArray("atypes");
            for (int i = 0; i < array.size(); i++) {
                d.atypes.add(array.get(i).getAsString());
            }

            // we don't need to load code or AST or bytecode back (from JRE data)
            d.code = null;
            d.ast = null;
            d.bytecode = null;

            return d;
        };
    }

}
