/**
 * 
 */
// package random;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;
import java.util.StringTokenizer;

public class A {
    private String currentState = "[START]";
    private String previousState = null;
    private Map<String, String> transitions = new HashMap<>();
    private boolean isOverlapping;
    private StringBuilder sequenceBuilder = new StringBuilder();

    public void addTransaction(String currentState, String nextState, int input, int output) {
        System.out.println(currentState + " " + previousState + " " + input + " " + output);
        String key = currentState + "@" + input;
        transitions.put(key, nextState);
    }

    public void execute(int input) {
        String key = currentState + "@" + input;
        if (transitions.containsKey(key)) {
            previousState = currentState;
            currentState = transitions.get(key);

            if (currentState.equals(previousState)) {
                isOverlapping = true;
            }

            sequenceBuilder.append(input);
        }
    }

    public String isOverLappingSequence() {
        return isOverlapping ? "Overlapping Sequence Detector" : "Non Overlapping Sequence Detector";
    }

    public String getSequence() {
        return sequenceBuilder.toString();
    }

    public static void main(String[] args) {
        // TODO Auto-generated method stub
        A detector = new A();
        detector.addTransaction("a", "b", 1, 0);
        detector.addTransaction("b", "c", 0, 0);

        FastScanner fs = new FastScanner(System.in);

        int n = fs.nextInt();
        fs.nextLine();

        for (int i = 0; i < n; i++) {
            try {
                String line = fs.nextLine();
                String[] tokens = line.split(" ");

                String currentState = tokens[0];
                String nextState = tokens[1];
                int input = Integer.parseInt(tokens[2]);
                int output = Integer.parseInt(tokens[3]);

                detector.addTransaction(currentState, nextState, input, output);

            } catch (Exception e) {
                // TODO: handle exception
            } finally {
                detector.execute(i);
            }
        }

        System.out.println(detector.getSequence());
        System.out.println(detector.isOverLappingSequence());
    }

    public static class FastScanner {
        private BufferedReader reader = null;
        private StringTokenizer tokenizer = null;

        public FastScanner(InputStream in) {
            reader = new BufferedReader(new InputStreamReader(in));
            tokenizer = null;
        }

        public String next() {
            if (tokenizer == null || !tokenizer.hasMoreTokens()) {
                try {
                    tokenizer = new StringTokenizer(reader.readLine());
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
            return tokenizer.nextToken();
        }

        public String nextLine() {
            if (tokenizer == null || !tokenizer.hasMoreTokens()) {
                try {
                    return reader.readLine();
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }

            return tokenizer.nextToken("\n");
        }

        public long nextLong() {
            return Long.parseLong(next());
        }

        public int nextInt() {
            return Integer.parseInt(next());
        }

        public double nextDouble() {
            return Double.parseDouble(next());
        }

        public int[] nextIntArray(int n) {
            int[] a = new int[n];
            for (int i = 0; i < n; i++)
                a[i] = nextInt();
            return a;
        }

        public long[] nextLongArray(int n) {
            long[] a = new long[n];
            for (int i = 0; i < n; i++)
                a[i] = nextLong();
            return a;
        }
    }
}