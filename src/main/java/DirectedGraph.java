import java.security.SecureRandom;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.StringJoiner;
import java.util.stream.Collectors;

/**
 * Represents a directed graph where nodes are words (Strings)
 * and edges have weights corresponding to adjacency counts.
 */

class DirectedGraph {
  final Map<String, Map<String, Integer>> adjList;
  private Map<String, Integer> nodeFrequencies;
  // Total words in the source text for TF/TF-IDF calculation
  private long totalWordCount;

  /**
     * .
     * Constructor
     */
  public DirectedGraph() {
    adjList = new HashMap<>();
    nodeFrequencies = new HashMap<>();
    totalWordCount = 0;
  }
  /**
   * Sets the total word count from the processed text.
   */

  public void setTotalWordCount(long count) {
    this.totalWordCount = count;
  }

  /**
     * Sets the node frequencies map.
     */
  public void setNodeFrequencies(Map<String, Integer> frequencies) {
    this.nodeFrequencies = new HashMap<>(frequencies); // Use a copy
    // Ensure all nodes with frequencies are in the adjacency list keys
    for (String node : frequencies.keySet()) {
      adjList.putIfAbsent(node, new HashMap<>());
    }
  }

  /**
     * Adds a node to the graph structure if it doesn't exist.
     */

  public void addNode(String word) {
    adjList.putIfAbsent(word, new HashMap<>());
    // Frequency should be set via setNodeFrequencies based on initial text processing
  }

  /**
     * Adds a directed edge from source to destination, incrementing weight. Ensures nodes exist.
     */
  public void addEdge(String source, String destination) {
    // Ensure nodes exist in the adjacency list
    addNode(source);
    addNode(destination);
    // Increment edge weight
    Map<String, Integer> neighbors = adjList.get(source);
    neighbors.put(destination, neighbors.getOrDefault(destination, 0) + 1);
  }

  // --- Graph Query Methods ---

  /**
     * Checks if a node exists in the graph.
     */
  public boolean containsNode(String node) {
    return adjList.containsKey(node);
  }

  /**
     * Gets the neighbors and edge weights for a given node.
     */
  public Map<String, Integer> getNeighbors(String node) {
    return adjList.getOrDefault(node, Collections.emptyMap());
  }

  /**
     * Gets all nodes (words) present in the graph.
     */
  public Set<String> getNodes() {
    return Collections.unmodifiableSet(adjList.keySet());
  }
  /**
     * Gets the frequency of a specific node (word).
     */

  public int getNodeFrequency(String node) {
    return nodeFrequencies.getOrDefault(node, 0);
  }

  /**
     * Checks if the graph is empty (contains no nodes).
     */
  public boolean isEmpty() {
    return adjList.isEmpty();
  }

  // --- Core Functionality Implementations ---

  /**
     * Finds bridge words between word1 and word2.
     */
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
  // ... (Shortest path methods remain unchanged) ...

  /**
     * Finds ONE shortest path using Dijkstra's algorithm.
     */
  public ShortestPathResult findShortestPath(String startNode, String endNode) {
    if (!containsNode(startNode) || !containsNode(endNode)) {
      return null;
    }

    Map<String, Integer> distances = new HashMap<>();
    PriorityQueue<
            Map.Entry<String, Integer>
            > pq = new PriorityQueue<>(
            Map.Entry.comparingByValue()
    );
    for (String node : adjList.keySet()) {
      distances.put(node, Integer.MAX_VALUE);
    }
    distances.put(startNode, 0);
    Map<String, String> previousNodes = new HashMap<>();
    pq.add(new AbstractMap.SimpleEntry<>(startNode, 0));

    while (!pq.isEmpty()) {
      Map.Entry<String, Integer> entry = pq.poll();
      String u = entry.getKey();
      int distU = entry.getValue();

      if (distU > distances.get(u)) {
        continue;
      }
      if (u.equals(endNode)) {
        break; // Found the shortest path to the target
      }

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

  /**
     * Optional: Finds ALL shortest paths between startNode and endNode.
     */
  public AllShortestPathsResult findAllShortestPaths(String startNode, String endNode) {
    if (!containsNode(startNode) || !containsNode(endNode)) {
      return null;
    }

    // 1. Dijkstra to find shortest distance and populate distances map
    Map<String, Integer> distances = new HashMap<>();
    PriorityQueue<Map.Entry<String, Integer>> pq =
            new PriorityQueue<>(Map.Entry.comparingByValue());
    for (String node : adjList.keySet()) {
      distances.put(node, Integer.MAX_VALUE);
    }
    distances.put(startNode, 0);
    pq.add(new AbstractMap.SimpleEntry<>(startNode, 0));

    while (!pq.isEmpty()) {
      Map.Entry<String, Integer> entry = pq.poll();
      String u = entry.getKey();
      if (entry.getValue() > distances.get(u)) {
        continue;
      }

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
  private void findAllPathsDFS(
          String u,
          String endNode,
          Map<String, Integer> distances,
          LinkedList<String> currentPath,
          List<List<String>> allPaths,
          int currentWeight,
          int targetWeight
  ) {
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
      if (distances.get(u) != Integer.MAX_VALUE
              && distances.get(v) != Integer.MAX_VALUE
              && distances.get(u) + edgeWeight == distances.get(v)) {
        currentPath.addLast(v);
        findAllPathsDFS(
                v,
                endNode,
                distances,
                currentPath,
                allPaths,
                currentWeight + edgeWeight,
                targetWeight
        );
        currentPath.removeLast(); // Backtrack
      }
    }
  }


  /**
     * Optional: Finds shortest paths from startNode to all other reachable nodes.
     */
  public Map<String, ShortestPathResult> findAllShortestPathsFrom(String startNode) {
    if (!containsNode(startNode)) {
      return Collections.emptyMap();
    }


    Map<String, Integer> distances = new HashMap<>();

    PriorityQueue<Map.Entry<String, Integer>> pq = new PriorityQueue<>(
            Map.Entry.comparingByValue()
    );

    for (String node : adjList.keySet()) {
      distances.put(node, Integer.MAX_VALUE);
    }
    distances.put(startNode, 0);
    pq.add(new AbstractMap.SimpleEntry<>(startNode, 0));
    Map<String, String> previousNodes = new HashMap<>();
    while (!pq.isEmpty()) {
      Map.Entry<String, Integer> entry = pq.poll();
      String u = entry.getKey();
      if (entry.getValue() > distances.get(u)) {
        continue;
      }

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
    Map<String, ShortestPathResult> results = new HashMap<>();
    // Reconstruct paths for all nodes
    for (String endNode : adjList.keySet()) {
      results.put(endNode, reconstructPath(startNode, endNode, distances, previousNodes));
    }
    return results;
  }

  // Helper method to reconstruct a single path from Dijkstra results
  private ShortestPathResult reconstructPath(
          String startNode,
          String endNode,
          Map<String, Integer> distances,
          Map<String, String> previousNodes
  ) {
    LinkedList<String> path = new LinkedList<>();
    int length = distances.getOrDefault(endNode, Integer.MAX_VALUE);

    if (length == Integer.MAX_VALUE && !startNode.equals(endNode)) {
      return new ShortestPathResult(Collections.emptyList(), Integer.MAX_VALUE); // Unreachable
    }

    String current = endNode;
    while (current != null) {
      path.addFirst(current);
      if (current.equals(startNode)) {
        break; // Found start
      }
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
    // 紧凑构造函数：自动验证和规范化参数
    public ShortestPathResult {
      // 对传入的path进行防御性复制
      path = List.copyOf(path);
    }

    // 直接返回不可变列表（无需额外处理）
    public List<String> getPath() {
      return path;
    }

    public int getLength() {
      return length;
    }
  }

  public record AllShortestPathsResult(List<List<String>> paths, int length) {
    // 紧凑构造函数，用于对传入的paths进行处理
    public AllShortestPathsResult {
      // 对每个子列表进行不可变处理，再将整个列表转换为不可变列表
      paths
              = paths.stream()
              .map(this::makeImmutableList)
              .toList();
      // 这里虽然不能直接赋值给paths（record特性），但上述操作已修改了传入的paths内容
    }

    private List<String> makeImmutableList(List<String> list) {
      return List.copyOf(list);
    }

    public List<List<String>> getPaths() {
      // 直接返回经过处理后的不可变paths
      return List.copyOf(paths);
    }

    public int getLength() {
      return length;
    }
  }
  // --- PageRank Implementation ---
  /**
     * Calculates PageRank using iteration. Allows uniform or TF-IDF-based initial rank.
     *
     * @param dampingFactor            The damping factor (typically 0.85).
     * @param useTfIdfBasedInitialRank If true,
   *                                 *initializes ranks based on TF-IDF;
   *                                  otherwise, uses uniform initialization.
     * @return A map where keys are node names (words) and values are their PageRank scores.
     */

  public Map<String, Double> calculatePageRank(
          double dampingFactor,
          boolean useTfIdfBasedInitialRank
  ) {
    Map<String, Double> ranks = new HashMap<>();
    Map<String, Integer> outDegree = new HashMap<>();
    Set<String> nodes = getNodes();
    int n = nodes.size();
    if (n == 0) {
      return ranks;
    }
    // Initialize ranks based on the chosen method (Uniform or TF-IDF)
    initializeRanks(ranks, n, useTfIdfBasedInitialRank); // <--- MODIFIED CALL

    // Pre-calculate out-degrees for efficiency
    for (String node : nodes) {
      outDegree.put(node, getNeighbors(node).size());
    }
    Map<String, Double> newRanks = new HashMap<>();
    double epsilon = 1e-6;         // Convergence threshold
    int maxIterations = 100;
    int iteration = 0;
    while (iteration++ < maxIterations) {
      double delta = 0.0; // Sum of changes in this iteration
      double sinkSum = 0.0; // Sum of ranks of sink nodes (out-degree 0)
      // Calculate the contribution from sink nodes (nodes with no outgoing links)
      // This ensures their rank is distributed among all nodes
      for (String node : nodes) {
        if (outDegree.getOrDefault(node, 0) == 0) {
          sinkSum += ranks.get(node);
        }
      }
      for (String node : nodes) {
        double incomingRankSum = 0.0;
        // Sum ranks from nodes pointing to the current node
        for (String other : nodes) {
          // Check if 'other' node points to the current 'node'
          if (getNeighbors(other).containsKey(node)) {
            int otherOutDegree = outDegree.get(other);
            if (otherOutDegree > 0) { // Avoid division by zero (shouldn't happen with pre-calc)
              incomingRankSum += ranks.get(other) / otherOutDegree;
            }
          }
        }
        // PageRank formula: (1-d)/N + d * (Sum(IncomingPR/OutgoingLinks) + SinkSum/N)
        double newRank = (1 - dampingFactor) / n
                + dampingFactor * (
                incomingRankSum
                        + sinkSum / n
        );
        newRanks.put(node, newRank);
        delta += Math.abs(newRank - ranks.get(node));
      }
      // Update ranks for the next iteration
      ranks.putAll(newRanks);
      // Check for convergence
      if (delta < epsilon) {
        System.out.println("PageRank converged after " + iteration + " iterations.");
        break; // Exit loop if converged
      }
    }
    if (iteration >= maxIterations) {
      System.out.println("PageRank did not converge within " + maxIterations + " iterations.");
    }
    return ranks;
  }

  // Helper: Initialize ranks (Uniform or TF-IDF based)
  private void initializeRanks(Map<String, Double> ranks, int n, boolean useTfIdf) {
    // Attempt TF-IDF initialization if requested and data is available
    if (useTfIdf && totalWordCount > 0 && nodeFrequencies != null && !nodeFrequencies.isEmpty()) {
      Map<String, Double> tfIdfScores = new HashMap<>();
      double totalTfIdf = 0;

      // Calculate TF-IDF for each node (word)
      for (String node : getNodes()) {
        int frequency = getNodeFrequency(node);
        if (frequency == 0) {
          continue; // Should not happen for nodes in getNodes() if populated correctly
        }
        double tf = (double) frequency / totalWordCount;
        double idf = Math.log((double) totalWordCount / (frequency + 1.0));

        // Calculate TF-IDF
        double tfIdfScore = tf * idf;

        tfIdfScores.put(node, tfIdfScore);
        totalTfIdf += tfIdfScore;
      }
      // Normalize TF-IDF scores to sum to 1 for initial rank distribution
      if (totalTfIdf > 0) {
        for (String node : getNodes()) {
          // Get the calculated TF-IDF score, default to 0 if node somehow wasn't processed
          double nodeTfIdf = tfIdfScores.getOrDefault(node, 0.0);
          ranks.put(node, nodeTfIdf / totalTfIdf); // Normalize
        }
        System.out.println("Initialized PageRank using TF-IDF (single-document heuristic).");
        return; // Exit after successful TF-IDF initialization
      } else {
        System.out.println(
                "Warning: TF-IDF sum was zero or negative "
                        +
                        "(check frequencies/total count). "
                        +
                        "Falling back to uniform initial PageRank."
        );
      }
    }
    System.out.println("Using uniform initial PageRank.");
    double initialRank = 1.0 / n;
    for (String node : getNodes()) {
      ranks.put(node, initialRank);
    }
    // Add a warning if TF-IDF was requested but couldn't be used
    if (useTfIdf
            && !(totalWordCount > 0
            && nodeFrequencies != null
            && !nodeFrequencies.isEmpty())
    ) {
      System.out.println(
              "Warning: Could not use TF-IDF-based initial rank "
                      +
                      "(totalWordCount or frequencies missing). "
                      +
                      "Using uniform."
      );
    }
  }

  /**
     * Performs a random walk until an edge is repeated or a dead end is hit.
     */
  public List<String> performRandomWalk() {
    List<String> pathNodes = new ArrayList<>();

    SecureRandom random = new SecureRandom();


    if (adjList.isEmpty()) {
      return pathNodes;
    }
    Set<String> visitedEdges = new HashSet<>(); // Store edges as "node1->node2"
    List<String> nodesList = new ArrayList<>(adjList.keySet());
    if (nodesList.isEmpty()) {
      return pathNodes;
    }
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
        break;
      }

      pathNodes.add(nextNode);
      currentNode = nextNode;
    }
    return pathNodes;
  }
  /**
     * Generates a string representation for CLI display.
     */

  @Override
    public String toString() {
    return generateStringRepresentation(Collections.emptySet(), Collections.emptySet());
  }

  /**
     * Generates a string representation highlighting a specific path.
     */
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
    List<String> sortedNodes = adjList.keySet().stream()
            .sorted()
            .toList();

    for (String node : sortedNodes) {
      String nodeMarker = markedNodes.contains(node) ? "***" : "";
      sb.append(nodeMarker).append(node).append(nodeMarker).append(" -> {");

      Map<String, Integer> neighbors = getNeighbors(node);
      // Sort neighbors for consistent output
      List<String> sortedNeighbors = neighbors.keySet()
              .stream()
              .sorted()
              .toList();

      StringJoiner sj = new StringJoiner(", ");
      for (String neighbor : sortedNeighbors) {
        String edge = node + "->" + neighbor;
        String edgeMarker = markedEdges.contains(edge) ? "***" : "";
        sj.add(edgeMarker + neighbor + "(" + neighbors.get(neighbor) + ")" + edgeMarker);
      }
      sb.append(sj).append("}\n");
    }
    if (sortedNodes.isEmpty()) {
      sb.append("(Graph is empty)\n");
    }
    return sb.toString();
  }

}