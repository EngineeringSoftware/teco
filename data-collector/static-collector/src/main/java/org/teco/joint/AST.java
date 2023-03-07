package org.teco.joint;

import java.util.List;
import com.google.gson.JsonArray;
import com.google.gson.JsonSerializer;

public class AST {
    // the AST type
    public String astType = null;
    // (terminal only) the token kind
    public String tokKind = null;

    // (terminal only) the literal token
    public String tok = null;
    // (non-terminal only) the children
    public List<AST> children = null;

    // lineno range of this node
    public String lineno = null;

    public void setLineno(int linenoBeg, int linenoEnd) {
        if (linenoBeg == linenoEnd) {
            lineno = String.valueOf(linenoBeg);
        } else {
            lineno = String.valueOf(linenoBeg) + "-" + String.valueOf(linenoEnd);
        }
    }

    public static final JsonSerializer<AST> sSerializer = getSerializer();

    public static JsonSerializer<AST> getSerializer() {
        return (d, type, jsonSerializationContext) -> {
            // target format:
            // non-terminal: [astType, lineno, children...]
            // terminal: [astType:tokKind, lineno, tok]
            JsonArray ret = new JsonArray();

            if (d.tokKind != null) {
                // terminal
                ret.add(d.astType + ":" + d.tokKind);
            } else {
                // non-terminal
                ret.add(d.astType);
            }

            ret.add(d.lineno);

            if (d.tokKind != null) {
                // terminal
                ret.add(d.tok);
            } else {
                // non-terminal
                for (AST child : d.children) {
                    ret.add(jsonSerializationContext.serialize(child));
                }
            }

            return ret;
        };
    }
}
