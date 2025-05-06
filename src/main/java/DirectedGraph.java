import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Represents a directed graph where nodes are words (Strings)
 * and edges have weights corresponding to adjacency counts.
 */
class DirectedGraph {
    // Adjacency list: Map<SourceNode, Map<DestinationNode, Weight>>
    private final Map<String, Map<String, Integer>> adjList;
    // Store frequency of each word (node)
    private Map<String, Integer> nodeFrequencies;
    // Total words in the source text for TF calculation
    private long totalWordCount;

    /** Constructor */
    public DirectedGraph() {
        adjList = new HashMap<>();
        nodeFrequencies = new HashMap<>();
        totalWordCount = 0;
    }

    // --- Graph Modification Methods ---

    /** Sets the total word count from the processed text. */
    public void setTotalWordCount(long count) {
        this.totalWordCount = count;
    }

    /** Sets the node frequencies map. */
    public void setNodeFrequencies(Map<String, Integer> frequencies) {
        this.nodeFrequencies = new HashMap<>(frequencies); // Use a copy
        // Ensure all nodes with frequencies are in the adjacency list keys
        for (String node : frequencies.keySet()) {
            adjList.putIfAbsent(node, new HashMap<>());
        }
    }

    /** Adds a node to the graph structure if it doesn't exist. */
    public void addNode(String word) {
        adjList.putIfAbsent(word, new HashMap<>());
        // Frequency should be set via setNodeFrequencies based on initial text processing
    }

    /** Adds a directed edge from source to destination, incrementing weight. Ensures nodes exist. */
    public void addEdge(String source, String destination) {
        // Ensure nodes exist in the adjacency list
        addNode(source);
        addNode(destination);
        // Increment edge weight
        Map<String, Integer> neighbors = adjList.get(source);
        neighbors.put(destination, neighbors.getOrDefault(destination, 0) + 1);
    }

    // --- Graph Query Methods ---

    /** Checks if a node exists in the graph. */
    public boolean containsNode(String node) {
        return adjList.containsKey(node);
    }

    /** Gets the neighbors and edge weights for a given node. */
    public Map<String, Integer> getNeighbors(String node) {
        return adjList.getOrDefault(node, Collections.emptyMap());
    }

    /** Gets all nodes (words) present in the graph. */
    public Set<String> getNodes() {
        return Collections.unmodifiableSet(adjList.keySet());
    }

    /** Gets the frequency of a specific node (word). */
    public int getNodeFrequency(String node) {
        return nodeFrequencies.getOrDefault(node, 0);
    }

    /** Checks if the graph is empty (contains no nodes). */
    public boolean isEmpty() {
        return adjList.isEmpty();
    }

    // --- Core Functionality Implementations ---

    /** Finds bridge words between word1 and word2. */
    public List<String> findBridgeWords(String word1, String word2) {
        List<String> bridges = new ArrayList<>();
        // Assumes caller checks word1/word2 existence
        Map<String, Integer> word1Neighbors = getNeighbors(word1);
        for (String potentialBridge : word1Neighbors.keySet()) {
            if (getNeighbors(potentialBridge).containsKey(word2)) {
                bridges.add(potentialBridge);
            }
        }
        // Sort for consistent output if needed (optional)
        // Collections.sort(bridges);
        return bridges;
    }

    // --- Shortest Path Algorithms ---

    /** Finds ONE shortest path using Dijkstra's algorithm. */
    public ShortestPathResult findShortestPath(String startNode, String endNode) {
        if (!containsNode(startNode) || !containsNode(endNode)) return null;

        Map<String, Integer> distances = new HashMap<>();
        Map<String, String> previousNodes = new HashMap<>();
        PriorityQueue<Map.Entry<String, Integer>> pq = new PriorityQueue<>(Map.Entry.comparingByValue());

        for (String node : adjList.keySet()) distances.put(node, Integer.MAX_VALUE);
        distances.put(startNode, 0);
        pq.add(new AbstractMap.SimpleEntry<>(startNode, 0));

        while (!pq.isEmpty()) {
            Map.Entry<String, Integer> entry = pq.poll();
            String u = entry.getKey();
            int distU = entry.getValue();

            if (distU > distances.get(u)) continue;
            if (u.equals(endNode)) break; // Found the shortest path to the target

            for (Map.Entry<String, Integer> neighborEntry : getNeighbors(u).entrySet()) {
                String v = neighborEntry.getKey();
                int weight = neighborEntry.getValue();
                if (distances.get(u) != Integer.MAX_VALUE) {
                    int newDist = distances.get(u) + weight;
                    if (newDist < distances.get(v)) {
                        distances.put(v, newDist);
                        previousNodes.put(v, u);
                        pq.add(new AbstractMap.SimpleEntry<>(v, newDist));
                    }
                }
            }
        }

        return reconstructPath(startNode, endNode, distances, previousNodes);
    }

    /** Optional: Finds ALL shortest paths between startNode and endNode. */
    public AllShortestPathsResult findAllShortestPaths(String startNode, String endNode) {
        if (!containsNode(startNode) || !containsNode(endNode)) return null;

        // 1. Dijkstra to find shortest distance and populate distances map
        Map<String, Integer> distances = new HashMap<>();
        PriorityQueue<Map.Entry<String, Integer>> pq = new PriorityQueue<>(Map.Entry.comparingByValue());
        for (String node : adjList.keySet()) distances.put(node, Integer.MAX_VALUE);
        distances.put(startNode, 0);
        pq.add(new AbstractMap.SimpleEntry<>(startNode, 0));

        while (!pq.isEmpty()) {
            Map.Entry<String, Integer> entry = pq.poll();
            String u = entry.getKey();
            if (entry.getValue() > distances.get(u)) continue;

            for (Map.Entry<String, Integer> neighborEntry : getNeighbors(u).entrySet()) {
                String v = neighborEntry.getKey();
                int weight = neighborEntry.getValue();
                if (distances.get(u) != Integer.MAX_VALUE) {
                    int newDist = distances.get(u) + weight;
                    if (newDist < distances.get(v)) {
                        distances.put(v, newDist);
                        pq.add(new AbstractMap.SimpleEntry<>(v, newDist));
                    }
                }
            }
        }

        int shortestLength = distances.get(endNode);
        if (shortestLength == Integer.MAX_VALUE) {
            return new AllShortestPathsResult(Collections.emptyList(), -1);
        }

        // 2. DFS from startNode to find all paths matching shortestLength
        List<List<String>> allPaths = new ArrayList<>();
        LinkedList<String> currentPath = new LinkedList<>();
        currentPath.add(startNode);
        findAllPathsDFS(startNode, endNode, distances, currentPath, allPaths, 0, shortestLength);

        // Remove duplicates if the DFS approach generates them (can happen with cycles)
        List<List<String>> uniquePaths = allPaths.stream().distinct().collect(Collectors.toList());

        return new AllShortestPathsResult(uniquePaths, shortestLength);
    }

    // Recursive DFS helper for findAllShortestPaths - adapted to use distances
    private void findAllPathsDFS(String u, String endNode, Map<String, Integer> distances,
                                 LinkedList<String> currentPath, List<List<String>> allPaths,
                                 int currentWeight, int targetWeight) {

        if (u.equals(endNode)) {
            if (currentWeight == targetWeight) { // Check if path length matches shortest
                allPaths.add(new ArrayList<>(currentPath));
            }
            return;
        }

        // Pruning: If current path weight exceeds target, stop exploring this path
        if (currentWeight > targetWeight) {
            return;
        }

        for (Map.Entry<String, Integer> neighborEntry : getNeighbors(u).entrySet()) {
            String v = neighborEntry.getKey();
            int edgeWeight = neighborEntry.getValue();

            // Explore neighbor 'v' only if it could potentially lead to *a* shortest path.
            // This check ensures we are moving towards the target along valid path segments.
            // distance(start, u) + weight(u, v) + estimate(v, end) >= shortestLength
            // A simple check: distance(start, u) + weight(u, v) == distance(start, v) calculated by Dijkstra
            // This confirms this edge is part of at least one shortest path found by Dijkstra.
            if (distances.get(u) != Integer.MAX_VALUE && distances.get(v) != Integer.MAX_VALUE &&
                    distances.get(u) + edgeWeight == distances.get(v))
            {
                currentPath.addLast(v);
                findAllPathsDFS(v, endNode, distances, currentPath, allPaths, currentWeight + edgeWeight, targetWeight);
                currentPath.removeLast(); // Backtrack
            }
            // Note: Simpler DFS without distance check might explore non-shortest paths.
            // Relying on Dijkstra's distances guides the DFS efficiently.
        }
    }


    /** Optional: Finds shortest paths from startNode to all other reachable nodes. */
    public Map<String, ShortestPathResult> findAllShortestPathsFrom(String startNode) {
        if (!containsNode(startNode)) return Collections.emptyMap();

        Map<String, ShortestPathResult> results = new HashMap<>();
        Map<String, Integer> distances = new HashMap<>();
        Map<String, String> previousNodes = new HashMap<>();
        PriorityQueue<Map.Entry<String, Integer>> pq = new PriorityQueue<>(Map.Entry.comparingByValue());

        for (String node : adjList.keySet()) distances.put(node, Integer.MAX_VALUE);
        distances.put(startNode, 0);
        pq.add(new AbstractMap.SimpleEntry<>(startNode, 0));

        while (!pq.isEmpty()) {
            Map.Entry<String, Integer> entry = pq.poll();
            String u = entry.getKey();
            if (entry.getValue() > distances.get(u)) continue;

            for (Map.Entry<String, Integer> neighborEntry : getNeighbors(u).entrySet()) {
                String v = neighborEntry.getKey();
                int weight = neighborEntry.getValue();
                if (distances.get(u) != Integer.MAX_VALUE) {
                    int newDist = distances.get(u) + weight;
                    if (newDist < distances.get(v)) {
                        distances.put(v, newDist);
                        previousNodes.put(v, u);
                        pq.add(new AbstractMap.SimpleEntry<>(v, newDist));
                    }
                }
            }
        }

        // Reconstruct paths for all nodes
        for (String endNode : adjList.keySet()) {
            results.put(endNode, reconstructPath(startNode, endNode, distances, previousNodes));
        }
        return results;
    }

    // Helper method to reconstruct a single path from Dijkstra results
    private ShortestPathResult reconstructPath(String startNode, String endNode,
                                               Map<String, Integer> distances,
                                               Map<String, String> previousNodes) {
        LinkedList<String> path = new LinkedList<>();
        int length = distances.getOrDefault(endNode, Integer.MAX_VALUE);

        if (length == Integer.MAX_VALUE && !startNode.equals(endNode)) {
            return new ShortestPathResult(Collections.emptyList(), Integer.MAX_VALUE); // Unreachable
        }

        String current = endNode;
        while (current != null) {
            path.addFirst(current);
            if (current.equals(startNode)) break; // Found start
            current = previousNodes.get(current); // Go to previous
        }

        // Verify path validity (should start with startNode if reachable)
        if (path.isEmpty() || !path.getFirst().equals(startNode)) {
            if (startNode.equals(endNode)) { // Handle start == end case
                return new ShortestPathResult(Collections.singletonList(startNode), 0);
            }
            // Path reconstruction failed, means unreachable or isolated node
            return new ShortestPathResult(Collections.emptyList(), Integer.MAX_VALUE);
        }

        return new ShortestPathResult(path, length);
    }


    // --- Helper Records for Path Results ---
    public record ShortestPathResult(List<String> path, int length) {
        public List<String> getPath() { return Collections.unmodifiableList(path); } // Return unmodifiable list
        public int getLength() { return length; }
    }
    public record AllShortestPathsResult(List<List<String>> paths, int length) {
        public List<List<String>> getPaths() { // Return unmodifiable list of unmodifiable lists
            return Collections.unmodifiableList(
                    paths.stream().map(Collections::unmodifiableList).collect(Collectors.toList())
            );
        }
        public int getLength() { return length; }
    }

    // --- PageRank Implementation ---

    /** Calculates PageRank using iteration. Allows TF-based initial rank. */
    public Map<String, Double> calculatePageRank(double dampingFactor, boolean useTfBasedInitialRank) {
        Map<String, Double> ranks = new HashMap<>();
        Map<String, Integer> outDegree = new HashMap<>();
        Set<String> nodes = getNodes();
        int n = nodes.size();
        if (n == 0) return ranks;

        initializeRanks(ranks, n, useTfBasedInitialRank);

        // 预计算出度
        for (String node : nodes) {
            outDegree.put(node, getNeighbors(node).size());
        }

        Map<String, Double> newRanks = new HashMap<>();
        double epsilon = 1e-6;         // 收敛误差阈值
        int maxIterations = 100;
        int iteration = 0;

        while (iteration++ < maxIterations) {
            double delta = 0.0;
            double sinkSum = 0.0;

            // 处理“死节点”的权重补偿
            for (String node : nodes) {
                if (outDegree.getOrDefault(node, 0) == 0) {
                    sinkSum += ranks.get(node);
                }
            }

            for (String node : nodes) {
                double sum = 0.0;

                // 遍历所有可能指向 node 的节点
                for (String other : nodes) {
                    if (getNeighbors(other).containsKey(node)) {
                        sum += ranks.get(other) / outDegree.get(other);
                    }
                }

                // PageRank 核心公式（含“死节点”补偿）
                double newRank = (1 - dampingFactor) / n +
                        dampingFactor * (sum + sinkSum / n);

                newRanks.put(node, newRank);
                delta += Math.abs(newRank - ranks.get(node));
            }

            ranks.putAll(newRanks);

            if (delta < epsilon) {
                System.out.println("PageRank converged after " + iteration + " iterations.");
                break;
            }
        }

        return ranks;
    }


    // Helper: Initialize ranks (Uniform or TF-based)
    private void initializeRanks(Map<String, Double> ranks, int n, boolean useTf) {
        if (useTf && totalWordCount > 0 && nodeFrequencies != null && !nodeFrequencies.isEmpty()) {
            double totalTf = 0;
            Map<String, Double> tfRanks = new HashMap<>();
            for (String node : getNodes()) {
                double tf = (double) getNodeFrequency(node) / totalWordCount;
                tfRanks.put(node, tf);
                totalTf += tf;
            }
            if (totalTf > 0) {
                // Normalize TF values to sum to 1
                for (String node : getNodes()) {
                    ranks.put(node, tfRanks.get(node) / totalTf);
                }
                System.out.println("Initialized PageRank using Term Frequency.");
                return; // Exit after successful TF initialization
            } else {
                System.out.println("Warning: TF sum was zero, falling back to uniform initial PageRank.");
            }
        }
        // Default: Uniform initialization
        double initialRank = 1.0 / n;
        for (String node : getNodes()) {
            ranks.put(node, initialRank);
        }
        if(useTf) System.out.println("Warning: Could not use TF-based initial rank (check frequencies/total count). Using uniform.");

    }

    // Helper: Calculate sum of ranks for sink nodes
    private double calculateSinkRankSum(Map<String, Double> ranks, Map<String, Integer> outDegree) {
        double sinkSum = 0;
        for (Map.Entry<String, Integer> entry : outDegree.entrySet()) {
            if (entry.getValue() == 0) { // Node is a sink
                sinkSum += ranks.get(entry.getKey());
            }
        }
        return sinkSum;
    }

    // Helper: Calculate rank contribution from incoming links
    private double calculateRankFromInputs(String targetNode, Map<String, Double> ranks, Map<String, Integer> outDegree) {
        double rankSum = 0;
        // Find nodes that link *to* targetNode
        for (String sourceNode : getNodes()) {
            if (getNeighbors(sourceNode).containsKey(targetNode)) {
                int sourceOutDegree = outDegree.get(sourceNode);
                if (sourceOutDegree > 0) {
                    rankSum += ranks.get(sourceNode) / sourceOutDegree;
                }
            }
        }
        return rankSum;
    }


    // --- Random Walk Implementation ---

    /** Performs a random walk until an edge is repeated or a dead end is hit. */
    public List<String> performRandomWalk() {
        List<String> pathNodes = new ArrayList<>();
        Set<String> visitedEdges = new HashSet<>(); // Store edges as "node1->node2"
        Random random = new Random();

        if (adjList.isEmpty()) return pathNodes;

        List<String> nodesList = new ArrayList<>(adjList.keySet());
        if (nodesList.isEmpty()) return pathNodes;
        String currentNode = nodesList.get(random.nextInt(nodesList.size()));
        pathNodes.add(currentNode);

        // Add loop for user stop mechanism if needed (requires concurrency or different input handling)

        while (true) {
            Map<String, Integer> neighbors = getNeighbors(currentNode);
            if (neighbors.isEmpty()) {
                // Dead end
                break;
            }

            List<String> neighborList = new ArrayList<>(neighbors.keySet());
            String nextNode = neighborList.get(random.nextInt(neighborList.size()));
            String edge = currentNode + "->" + nextNode;

            if (!visitedEdges.add(edge)) { // add returns false if element already exists
                // Edge repeated
                pathNodes.add(nextNode); // Add the node that completes the repeated edge
                break;
            }

            pathNodes.add(nextNode);
            currentNode = nextNode;
        }
        return pathNodes;
    }


    // --- Graph Output Methods ---

    /** Generates a string representation for CLI display. */
    @Override
    public String toString() {
        return generateStringRepresentation(Collections.emptySet(), Collections.emptySet());
    }

    /** Generates a string representation highlighting a specific path. */
    public String toStringWithPath(List<String> path) {
        Set<String> pathNodes = new HashSet<>(path);
        Set<String> pathEdges = new HashSet<>();
        for (int i = 0; i < path.size() - 1; i++) {
            pathEdges.add(path.get(i) + "->" + path.get(i + 1));
        }
        return generateStringRepresentation(pathNodes, pathEdges);
    }

    // Helper for generating string representation with highlighting
    private String generateStringRepresentation(Set<String> markedNodes, Set<String> markedEdges) {
        StringBuilder sb = new StringBuilder();
        // Sort nodes for consistent output
        List<String> sortedNodes = adjList.keySet().stream().sorted().collect(Collectors.toList());

        for (String node : sortedNodes) {
            String nodeMarker = markedNodes.contains(node) ? "***" : "";
            sb.append(nodeMarker).append(node).append(nodeMarker).append(" -> {");

            Map<String, Integer> neighbors = getNeighbors(node);
            // Sort neighbors for consistent output
            List<String> sortedNeighbors = neighbors.keySet().stream().sorted().collect(Collectors.toList());

            StringJoiner sj = new StringJoiner(", ");
            for (String neighbor : sortedNeighbors) {
                String edge = node + "->" + neighbor;
                String edgeMarker = markedEdges.contains(edge) ? "***" : "";
                sj.add(edgeMarker + neighbor + "(" + neighbors.get(neighbor) + ")" + edgeMarker);
            }
            sb.append(sj.toString()).append("}\n");
        }
        if (sortedNodes.isEmpty()) {
            sb.append("(Graph is empty)\n");
        }
        return sb.toString();
    }

    /** Optional: Generates a DOT file representation for Graphviz. */
    public void generateDotFile(String filename) throws IOException {
        // Use try-with-resources for automatic closing
        try (PrintWriter writer = new PrintWriter(new FileWriter(filename))) {
            writer.println("digraph G {");
            writer.println("  rankdir=LR;"); // Layout direction
            writer.println("  node [shape=ellipse, style=filled, color=lightblue];"); // Node style
            writer.println("  edge [color=gray];"); // Edge style

            List<String> sortedNodes = adjList.keySet().stream().sorted().collect(Collectors.toList());

            for (String sourceNode : sortedNodes) {
                // Write node definition even if it has no outgoing edges listed explicitly below
                writer.println("  \"" + escapeDot(sourceNode) + "\";");

                Map<String, Integer> neighbors = getNeighbors(sourceNode);
                List<String> sortedNeighbors = neighbors.keySet().stream().sorted().collect(Collectors.toList());

                for (String destNode : sortedNeighbors) {
                    int weight = neighbors.get(destNode);
                    writer.println("  \"" + escapeDot(sourceNode) + "\" -> \"" + escapeDot(destNode)
                            + "\" [label=\"" + weight + "\", weight=" + weight + "];"); // Add edge weight for layout hint
                }
            }
            writer.println("}");
        }
    }

    // Helper to escape strings for DOT format
    private String escapeDot(String s) {
        // Basic escaping for quotes. More complex escaping might be needed for other special chars.
        return s.replace("\"", "\\\"");
    }
}