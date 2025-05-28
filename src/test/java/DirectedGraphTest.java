import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.lang.reflect.Field;
import java.util.List;
import java.util.Map;

public class DirectedGraphTest {
    private DirectedGraph graph;

    @BeforeEach
    void setUp() throws NoSuchFieldException, IllegalAccessException {
        graph = new DirectedGraph();
        // 确保每次测试从干净状态开始
    }

    @Test
    void testPerformRandomWalkEmptyGraph() {
        // 测试用例 1: 空图
        List<String> result = graph.performRandomWalk();
        System.out.println(result);
        assertEquals(List.of(), result, "Empty graph should return empty path.");
    }

    @Test
    void testPerformRandomWalkSingleNodeNoEdges() {
        // 测试用例 2: 单一节点，无边
        graph.addNode("node1");
        List<String> result = graph.performRandomWalk();
        System.out.println(result);
        assertEquals(List.of("node1"), result, "Single node with no edges should return [node1].");
    }

    @Test
    void testPerformRandomWalkTwoNodesDirected() {
        // 测试用例3: 两个节点，单向边 node1 -> node2
        graph.addEdge("node1", "node2");
        List<String> result = graph.performRandomWalk();
        System.out.println(result);
        List<List<String>> expected = List.of(
                List.of("node1", "node2"),
                List.of("node2")
        );
        assertTrue(expected.contains(result), "Random walk should be either node1 -> node2 or node2 alone.");
    }

    @Test
    void testPerformRandomWalkSelfLoop() {
        // 测试用例 4: 一个节点，自循环 node1 -> node1
        graph.addEdge("node1", "node1");
        List<String> result = graph.performRandomWalk();
        System.out.println(result);
        List<String> expected = List.of("node1", "node1");
        assertEquals(expected, result, "Should walk node1 -> node1 and stop at repeated edge.");
    }

    @Test
    void testPerformRandomWalkSingleNodeNoValidEdges() {
        // 测试用例 5: 一个节点，无有效边
        graph.addNode("node1");
        List<String> result = graph.performRandomWalk();
        System.out.println(result);
        assertEquals(List.of("node1"), result, "Single node with no valid edges should return [node1].");
    }

    @Test
    void testPerformRandomWalkEmptyNodeList() throws NoSuchFieldException, IllegalAccessException {
        // 测试用例 6: adjList 初始化但未添加节点
        Field adjListField = DirectedGraph.class.getDeclaredField("adjList");
        adjListField.setAccessible(true);
        Map<String, Map<String, Integer>> adjList = (Map<String, Map<String, Integer>>) adjListField.get(graph);
        adjList.clear(); // 清空 adjList 模拟异常状态
        List<String> result = graph.performRandomWalk();
        System.out.println(result);
        assertEquals(List.of(), result, "Graph with empty node list should return empty path.");
    }
    @Test
    void testPerformRandomWalkInvalidNodeList() throws NoSuchFieldException, IllegalAccessException {
        // 测试用例 7: 模拟 nodesList 为空，触发 random.nextInt 异常
        Field adjListField = DirectedGraph.class.getDeclaredField("adjList");
        adjListField.setAccessible(true);
        Map<String, Map<String, Integer>> adjList = (Map<String, Map<String, Integer>>) adjListField.get(graph);
        adjList.clear(); // 清空 adjList
        List<String> result = graph.performRandomWalk();
        System.out.println(result);
        assertEquals(List.of(), result, "Should handle empty nodesList gracefully.");
    }
}