package org.teco.joint;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import org.apache.commons.lang3.tuple.Pair;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonSerializer;

public class CallGraph {

    // a -> b: a depend on b
    public Map<Integer, Set<Pair<Integer, String>>> edges = new HashMap<>();

    public static final String EDGE_CALL = "C";
    public static final String EDGE_OVERRIDE = "O";

    synchronized public void addEdge(int from, int to, String label) {
        edges.computeIfAbsent(from, k -> new HashSet<>()).add(Pair.of(to, label));
    }

    // Serialization
    public static final JsonSerializer<CallGraph> sSerializer = getSerializer();

    public static JsonSerializer<CallGraph> getSerializer() {
        return (d, type, jsonSerializationContext) -> {
            JsonObject obj = new JsonObject();
            JsonObject edgesObj = new JsonObject();

            for (Map.Entry<Integer, Set<Pair<Integer, String>>> entry : d.edges.entrySet()) {
                JsonArray edgesArray = new JsonArray();
                for (Pair<Integer, String> edge : entry.getValue()) {
                    JsonArray edgeArray = new JsonArray();
                    edgeArray.add(edge.getLeft());
                    edgeArray.add(edge.getRight());
                    edgesArray.add(edgeArray);
                }
                edgesObj.add(entry.getKey().toString(), edgesArray);
            }
            obj.add("edges", edgesObj);
            return obj;
        };
    }
}
