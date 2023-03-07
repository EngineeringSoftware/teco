package org.teco.subjects;

import static org.hamcrest.Matchers.equalTo;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThat;
import org.junit.Assert;
import org.junit.Test;

public class C {

    private int utilMethod1(int x) {
        return x * x;
    }

    public int utilMethod2(int x) {
        return x + 1;
    }

    @Test
    public void testA() {
        assertEquals(16, utilMethod1(4));
        assertThat(utilMethod2(6), equalTo(7));
    }

    @Test
    public void testB() {
        Assert.assertEquals(16, utilMethod1(4));
        Assert.assertThat(utilMethod2(6), equalTo(7));
    }

    public static void main(String... args) {
        C c = new C();
        assertEquals(16, c.utilMethod1(4));
        assertThat(c.utilMethod2(6), equalTo(7));
    }
}
