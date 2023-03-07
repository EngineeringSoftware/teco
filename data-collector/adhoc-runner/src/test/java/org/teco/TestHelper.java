package org.teco;

import static org.junit.Assert.assertEquals;
import static org.junit.Assume.assumeTrue;
import java.nio.file.Files;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

public class TestHelper {

    @Rule
    public TemporaryFolder logPath = new TemporaryFolder();

    @Test
    public void testValueOfPrimitive() {
        assertEquals("true", Helper.valueOf(true));
        assertEquals("false", Helper.valueOf(false));
        assertEquals("1", Helper.valueOf((byte) 1));
        assertEquals("'a'", Helper.valueOf('a'));
        assertEquals("'\\''", Helper.valueOf('\''));
        assertEquals("'\\n'", Helper.valueOf('\n'));
        assertEquals("'\\r'", Helper.valueOf('\r'));
        assertEquals("1.0", Helper.valueOf(1.0));
        assertEquals("1.0", Helper.valueOf(1.0f));
        assertEquals("1", Helper.valueOf(1));
        assertEquals("1", Helper.valueOf(1L));
        assertEquals("1", Helper.valueOf((short) 1));
    }

    @Test
    public void testValueOfString() {
        assertEquals(Helper.valueOf("abc"), "\"abc\"");
        assertEquals(Helper.valueOf("a\nb\rc\""), "\"a\\nb\\rc\\\"\"");
    }

    @Test
    public void testValueOfOtherObj() {
        assertEquals(Helper.valueOf(null), Helper.OBJ_VALUE_NULL);
        assertEquals(Helper.valueOf(new Object()), Helper.OBJ_VALUE_NOTNULL);
    }

    public String readLog(String fileName) throws Exception {
        return new String(Files.readAllBytes(logPath.getRoot().toPath().resolve(fileName)));
    }

    @Test
    public void testLogPrimitiveVar() throws Exception {
        Helper.logVarDepth("intVar", 100, logPath.getRoot().getAbsolutePath(), 0, 0);
        String log = readLog("typevalue-0");
        assertEquals("intVar int 100\n", log);
    }

    @Test
    public void testLogVeryLongString() throws Exception {
        String longString = "";
        for (int i = 0; i < Helper.MAX_STR_LENGTH + 1; ++i) {
            longString += "a";
        }
        Helper
            .logVarDepth("longString", longString, logPath.getRoot().getAbsolutePath(), 0, 0);
        String log = readLog("typevalue-0");
        String expectedString = "\"";
        for (int i = 0; i < Helper.MAX_STR_LENGTH - 5; ++i) {
            expectedString += "a";
        }
        expectedString += "..." + "\"";
        assertEquals("longString String " + expectedString + "\n", log);
    }

    static class CompoundType {
        public static int STATIC_INT = -100;
        private int i;
        public CompoundType c;

        public CompoundType(int i, CompoundType c) {
            this.i = i;
            this.c = c;
        }
    }

    static class InheritedType extends CompoundType {
        private int i;
        private int j;

        public InheritedType(int i, CompoundType c, int j) {
            super(i, c);
            this.i = i + j;
            this.j = j;
        }
    }

    @Test
    public void testLogCompoundVar() throws Exception {
        assumeTrue(Helper.MAX_DEPTH >= 1);
        CompoundType var = new CompoundType(100, null);
        Helper.logVarDepth("cvar", var, logPath.getRoot().getAbsolutePath(), 0, 0);
        String log = readLog("typevalue-0");
        assertEquals(
            "cvar CompoundType " + Helper.OBJ_VALUE_NOTNULL + "\n" + "cvar.i int 100\n"
                + "cvar.c null " + Helper.OBJ_VALUE_NULL + "\n",
            log);
    }

    @Test
    public void testLogNestedCompoundVar() throws Exception {
        assumeTrue(Helper.MAX_DEPTH >= 2);
        CompoundType var = new CompoundType(100, new CompoundType(200, null));
        Helper.logVarDepth("cvar", var, logPath.getRoot().getAbsolutePath(), 0, 0);
        String log = readLog("typevalue-0");
        assertEquals(
            "cvar CompoundType " + Helper.OBJ_VALUE_NOTNULL + "\n" + "cvar.i int 100\n"
                + "cvar.c CompoundType " + Helper.OBJ_VALUE_NOTNULL + "\n"
                + "cvar.c.i int 200\n" + "cvar.c.c null " + Helper.OBJ_VALUE_NULL + "\n",
            log);
    }

    @Test
    public void testLogInheritedVar() throws Exception {
        // we're also considering fields in base class, and considering fields hidden by same-name fields in derived class
        assumeTrue(Helper.MAX_DEPTH >= 1);
        CompoundType var = new InheritedType(100, null, 300);
        Helper.logVarDepth("cvar", var, logPath.getRoot().getAbsolutePath(), 0, 0);
        String log = readLog("typevalue-0");
        assertEquals(
            "cvar InheritedType " + Helper.OBJ_VALUE_NOTNULL + "\n" + "cvar.i int 400\n"
                + "cvar.j int 300\n" + "cvar.c null " + Helper.OBJ_VALUE_NULL + "\n",
            log);
    }
}
