package org.teco.util;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.commons.collections4.OrderedMap;
import org.apache.commons.collections4.map.LinkedMap;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonSerializer;

public class MethodBytecode {

    // instructions, assigned ids to maintain order (some ids may be skipped)
    // each instruction is a list of elements;
    // the first element is operator, and the rest are operands
    public OrderedMap<Integer, List<String>> insns = new LinkedMap<>();

    // line number table
    // list of mapping from instruction id to source code line number
    public OrderedMap<Integer, Integer> lnt = new LinkedMap<>();

    /**
     * Coalesces the instructions list and removes gaps in their ids, with adjusting other attributes (e.g., line number table) appropriately.
     */
    public void coalesce() {
        if (insns.size() == 0) {
            return;
        }

        // collect the mapping from old id to new id, and update the instruction list on the way
        Map<Integer, Integer> old2new = new HashMap<>();
        int lastOldId = 0;
        int newId = 0;
        OrderedMap<Integer, List<String>> newInsns = new LinkedMap<>();
        for (int oldId : insns.keySet().stream().sorted().toArray(Integer[]::new)) {
            while (oldId > lastOldId) {
                old2new.put(lastOldId, newId);
                ++lastOldId;
            }
            old2new.put(oldId, newId);
            newInsns.put(newId, insns.get(oldId));
            ++newId;
        }
        insns = newInsns;

        // update the line number table
        OrderedMap<Integer, Integer> newLnt = new LinkedMap<>();
        for (Map.Entry<Integer, Integer> entry : lnt.entrySet()) {
            newLnt.put(old2new.getOrDefault(entry.getKey(), newId - 1), entry.getValue());
        }
        lnt = newLnt;
    }


    // (de)serialization
    public static final JsonSerializer<MethodBytecode> sSerializer = getSerializer();

    public static JsonSerializer<MethodBytecode> getSerializer() {
        return (d, type, jsonSerializationContext) -> {
            JsonObject object = new JsonObject();

            JsonObject jInsns = new JsonObject();
            for (Map.Entry<Integer, List<String>> entry : d.insns.entrySet()) {
                JsonArray jInsn = new JsonArray();
                for (String insn : entry.getValue()) {
                    jInsn.add(insn);
                }
                jInsns.add(entry.getKey().toString(), jInsn);
            }
            object.add("insns", jInsns);

            JsonObject jLNT = new JsonObject();
            for (Map.Entry<Integer, Integer> entry : d.lnt.entrySet()) {
                jLNT.addProperty(entry.getKey().toString(), entry.getValue());
            }
            object.add("lnt", jLNT);

            return object;
        };
    }
}
