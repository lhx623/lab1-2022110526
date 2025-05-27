import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

public class Lab1Test {
    private DirectedGraph graph;

    @BeforeEach
    void setUp() {
        // Initialize graph with sample text: "the quick fox jumps and the big dog jumps and fox runs and quick jumps"
        graph = new DirectedGraph();
        String text = "the quick fox jumps and the big dog jumps and fox runs and quick jumps";
        String[] words = text.toLowerCase().split("\\s+");
        Map<String, Integer> frequencies = new HashMap<>();
        for (String word : words) {
            frequencies.put(word, frequencies.getOrDefault(word, 0) + 1);
        }
        graph.setTotalWordCount(words.length);
        graph.setNodeFrequencies(frequencies);
        for (int i = 0; i < words.length - 1; i++) {
            graph.addEdge(words[i], words[i + 1]);
        }
        graph.addNode(words[words.length - 1]);
        Lab1.graph = graph; // Set the static graph in Lab1
    }

    @Test
    void testQueryBridgeWordsSingleBridge() {
        String result = Lab1.queryBridgeWords("quick", "and");
        assertEquals("The bridge words from \"quick\" to \"and\" is: \"jumps\".", result,
                "Should find 'jumps' as the only bridge word.");
    }

    @Test
    void testQueryBridgeWordsMultipleBridges() {
        String result = Lab1.queryBridgeWords("fox", "and");
        // Allow for flexible ordering since sorting is optional in DirectedGraph.findBridgeWords
        String expected1 = "The bridge words from \"fox\" to \"and\" are: \"jumps\", and \"runs\".";
        String expected2 = "The bridge words from \"fox\" to \"and\" are: \"runs\", and \"jumps\".";
        assertTrue(result.equals(expected1) || result.equals(expected2),
                "Should find 'jumps' and 'runs' as bridge words, got: " + result);
    }

    @Test
    void testQueryBridgeWordsNullGraph() {
        Lab1.graph = null;
        String result = Lab1.queryBridgeWords("quick", "and");
        assertEquals("Graph not initialized.", result, "Should handle uninitialized graph.");
    }

    @Test
    void testQueryBridgeWordsEmptyWord1() {
        String result = Lab1.queryBridgeWords("", "and");
        assertEquals("Please provide two valid words.", result, "Should handle empty word1.");
    }

    @Test
    void testQueryBridgeWordsSameWords() {
        String result = Lab1.queryBridgeWords("the", "the");
        assertEquals("No bridge words from \"the\" to \"the\"!", result, "Should handle same words.");
    }

    @Test
    void testQueryBridgeWordsNoBridgeWords() {
        String result = Lab1.queryBridgeWords("jumps", "big");
        assertEquals("No bridge words from \"jumps\" to \"big\"!", result, "Should handle no bridge words.");
    }

    @Test
    void testQueryBridgeWordsWord2NotInGraph() {
        String result = Lab1.queryBridgeWords("quick", "xyz");
        assertEquals("No \"xyz\" in the graph!", result, "Should handle word2 not in graph.");
    }

    @Test
    void testQueryBridgeWordsWord1NotInGraph() {
        String result = Lab1.queryBridgeWords("xyz", "and");
        assertEquals("No \"xyz\" in the graph!", result, "Should handle word1 not in graph.");
    }

    @Test
    void testQueryBridgeWordsBothWordsNotInGraph() {
        String result = Lab1.queryBridgeWords("xyz", "abc");
        assertEquals("No \"xyz\" and \"abc\" in the graph!", result, "Should handle both words not in graph.");
    }
}