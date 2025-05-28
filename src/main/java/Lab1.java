import guru.nidi.graphviz.engine.Format;
import guru.nidi.graphviz.engine.Graphviz;
import guru.nidi.graphviz.model.MutableGraph;
import guru.nidi.graphviz.model.Node; // Use this specific Node
import static guru.nidi.graphviz.model.Factory.*; // For graph creation methods
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.security.SecureRandom;
import java.util.*;
import java.util.stream.Collectors;
// Imports for graphviz-java

public class Lab1 {
  private static final SecureRandom random = new SecureRandom();
  static DirectedGraph graph;
  // private static String originalTextContent; // Store preprocessed text if needed elsewhere

  // ... (main method and other parts up to handleCalcPageRank remain the same) ...
  public static void main(String[] args) {
    // --- 1. 获取输入文件路径 ---
    String filePath = "";
    Scanner scanner = new Scanner(System.in, "UTF-8"); // Keep scanner open for menu

    if (args.length > 0) {
      filePath = args[0];
      System.out.println("Reading file from command line argument: " + filePath);
    } else {
      System.out.print("Enter the path to the text file: ");
      filePath = scanner.nextLine();
    }

    // --- 2. 读取文件并构建图 ---
    try {
      String processedText = readFileAndPreprocess(filePath);
      if (processedText.isEmpty()) {
        System.err.println("Error: Preprocessed text is empty. Please check the input file.");
        scanner.close();
        return;
      }
      graph = buildGraph(processedText);
      System.out.println("Graph built successfully!");

      // --- 3. 用户交互菜单 ---
      int choice = -1;
      while (choice != 0) {
        displayMenu(); // Display updated menu
        try {
          String inputLine = scanner.nextLine();
          if (inputLine.trim().isEmpty()) {
            System.out.println("Invalid input. Please enter a number.");
            continue;
          }
          choice = Integer.parseInt(inputLine);

          switch (choice) {
            case 1:
              showDirectedGraph(graph);
              break;
            case 2:
              handleQueryBridgeWords(scanner);
              break;
            case 3:
              handleGenerateNewText(scanner);
              break;
            case 4:
              handleCalcShortestPath(scanner);
              break;
            case 5:
              handleCalcAllShortestPaths(scanner);
              break;
            case 6:
              handleCalcPageRank(scanner); // MODIFIED CALL HERE
              break;
            case 7:
              handleRandomWalk();
              break;
            case 8: // Updated: Save graph as image
              handleSaveGraphAsImage(scanner); // Call new handler
              break;
            case 0:
              System.out.println("Exiting...");
              break;
            default:
              System.out.println("Invalid choice. Please try again.");
          }
        } catch (NumberFormatException e) {
          System.out.println("Invalid input. Please enter a number.");
        } catch (Exception e) {
          System.err.println("An error occurred: " + e.getMessage());
          // Check specifically for Graphviz engine errors if needed
          if (e.getMessage() != null && e.getMessage().contains("Cannot run program \"dot\"")) {
            System.err.println("Graphviz 'dot' command not found. Please ensure Graphviz is installed and in your system PATH.");
          } else {
            e.printStackTrace(); // For other unexpected errors
          }
        }
        if (choice != 0) {
          System.out.println("\nPress Enter to continue...");
          scanner.nextLine();
        }
      }

    } catch (IOException e) {
      System.err.println("Error reading file '" + filePath + "': " + e.getMessage());
    } catch (IllegalArgumentException e) {
      System.err.println("Error: " + e.getMessage());
    } finally {
      scanner.close();
    }
  }

  // ... (displayMenu, handleSaveGraphAsImage, etc., remain the same) ...
  private static void displayMenu() {
    System.out.println("\n========== Lab 1 Menu ==========");
    System.out.println(" 1. Show Directed Graph (CLI)");
    System.out.println(" 2. Query Bridge Words");
    System.out.println(" 3. Generate New Text from Bridge Words");
    System.out.println(" 4. Calculate Shortest Path (One or All from Source)");
    System.out.println(" 5. Calculate ALL Shortest Paths (Between Two Words)");
    System.out.println(" 6. Calculate PageRank");
    System.out.println(" 7. Perform Random Walk");
    System.out.println(" 8. Save Graph as Image (e.g., PNG)"); // Updated description
    System.out.println(" 0. Exit");
    System.out.println("================================");
    System.out.print("Enter your choice: ");
  }

  // --- New Handler for Saving Image ---
  private static void handleSaveGraphAsImage(Scanner scanner) {
    System.out.print("Enter filename to save graph image: ");
    String imageFilename = scanner.nextLine();
    // Add .png extension if missing (or allow other formats)
    if (!imageFilename.toLowerCase().endsWith(".png") &&
            !imageFilename.toLowerCase().endsWith(".svg")){
      System.out.println("Adding .png extension automatically.");
      imageFilename += ".png";
    }
    saveGraphAsImage(graph, imageFilename); // Call the image saving function
  }

  // --- Other handler methods remain the same as before ---
  private static void handleQueryBridgeWords(Scanner scanner) {
    System.out.print("Enter word1: ");
    String word1 = scanner.nextLine().toLowerCase();
    System.out.print("Enter word2: ");
    String word2 = scanner.nextLine().toLowerCase();
    System.out.println(queryBridgeWords(word1, word2));
  }

  private static void handleGenerateNewText(Scanner scanner) {
    System.out.print("Enter new text: ");
    String inputText = scanner.nextLine();
    System.out.println("Generated text: " + generateNewText(inputText));
  }

  private static void handleCalcShortestPath(Scanner scanner) {
    System.out.print("Enter word1 for shortest path: ");
    String word1 = scanner.nextLine().toLowerCase();
    System.out.print("Enter word2 (optional, press Enter to calculate paths to all): ");
    String word2Input = scanner.nextLine().toLowerCase();
    String word2 = word2Input.isEmpty() ? null : word2Input;
    System.out.println(calcShortestPath(word1, word2));
  }

  private static void handleCalcAllShortestPaths(Scanner scanner) {
    System.out.print("Enter word1 for ALL shortest paths: ");
    String word1 = scanner.nextLine().toLowerCase();
    System.out.print("Enter word2 for ALL shortest paths: ");
    String word2 = scanner.nextLine().toLowerCase();
    System.out.println(calcAllShortestPaths(word1, word2));
  }


  // MODIFIED: Handler for PageRank calculation
  private static void handleCalcPageRank(Scanner scanner) {
    System.out.print("Enter word for PageRank (leave blank for all): ");
    String wordInput = scanner.nextLine().toLowerCase();

    // Ask user whether to use TF-IDF for initialization
    boolean useTfIdfPr = false;
    System.out.print("Use TF-IDF for initial PageRank? (yes/no, default no): "); // <--- MODIFIED PROMPT
    String tfIdfChoice = scanner.nextLine().toLowerCase();
    if (tfIdfChoice.equals("yes")) {
      useTfIdfPr = true;
    }

    // Call the PageRank calculation method with the flag
    Map<String, Double> pageRanks = calPageRank(useTfIdfPr); // <--- MODIFIED CALL

    if (pageRanks != null) {
      if (wordInput.isEmpty()) {
        System.out.println("\n--- PageRank Results (Top 20) ---");
        pageRanks.entrySet().stream()
                .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
                .limit(20)
                .forEach(entry -> System.out.printf("%-15s: %.6f%n", entry.getKey(), entry.getValue()));
        System.out.println("---------------------------------");
      } else if (pageRanks.containsKey(wordInput)) {
        System.out.printf("PageRank for \"%s\": %.6f%n", wordInput, pageRanks.get(wordInput));
      } else {
        System.out.println("Word \"" + wordInput + "\" not found in graph.");
      }
    } else {
      System.out.println("PageRank calculation failed or graph was empty.");
    }
  }

  private static void handleRandomWalk() {
    System.out.println("Starting random walk...");
    System.out.println(randomWalk());
  }


  // --- 文件读取和预处理 (Same as before) ---
  public static String readFileAndPreprocess(String filePath) throws IOException {
    StringBuilder content = new StringBuilder();
    try (BufferedReader reader = new BufferedReader(
            new InputStreamReader(new FileInputStream(filePath), "UTF-8"))) {
      String line;
      while ((line = reader.readLine()) != null) {
        content.append(line).append(" ");
      }
    }
    String processed = content.toString().toLowerCase()
            .replaceAll("[^a-z\\s]+", " ")
            .replaceAll("\\s+", " ");
    return processed.trim();
  }

  // --- 图构建 (Same as before) ---
  public static DirectedGraph buildGraph(String text) {
    String[] words = text.split("\\s+");
    List<String> wordList = Arrays.stream(words)
            .filter(s -> !s.isEmpty())
            .collect(Collectors.toList());
    if (wordList.isEmpty()) {
      throw new IllegalArgumentException("Input text contains no valid words after processing.");
    }
    DirectedGraph newGraph = new DirectedGraph();
    newGraph.setTotalWordCount(wordList.size());
    Map<String, Integer> frequencies = new HashMap<>();
    for(String word : wordList) {
      frequencies.put(word, frequencies.getOrDefault(word, 0) + 1);
    }
    newGraph.setNodeFrequencies(frequencies);
    for (int i = 0; i < wordList.size() - 1; i++) {
      String word1 = wordList.get(i);
      String word2 = wordList.get(i + 1);
      newGraph.addEdge(word1, word2);
    }
    // Ensure the last word is added as a node if it wasn't already a source
    if (!wordList.isEmpty()) {
      newGraph.addNode(wordList.get(wordList.size() - 1));
    }
    return newGraph;
  }


  // --- 功能实现 (Core logic calls graph methods) ---

  /** Function 2: Show Directed Graph */
  public static void showDirectedGraph(DirectedGraph G) {
    System.out.println("\n--- Directed Graph (CLI Representation) ---");
    if (G == null || G.isEmpty()) {
      System.out.println("Graph is empty or not initialized.");
    } else {
      System.out.println(G.toString());
    }
    System.out.println("-------------------------------------------");
  }

  // ... (queryBridgeWords, generateNewText, calcShortestPath, calcAllShortestPaths remain the same) ...
  /** Function 3: Query Bridge Words */
  public static String queryBridgeWords(String word1, String word2) {
    if (graph == null) return "Graph not initialized.";
    if (word1 == null || word1.isEmpty() || word2 == null || word2.isEmpty()) {
      return "Please provide two valid words.";
    }
    boolean word1Exists = graph.containsNode(word1);
    boolean word2Exists = graph.containsNode(word2);

    if (!word1Exists || !word2Exists) {
      if (!word1Exists && !word2Exists) return "No \"" + word1 + "\" and \"" + word2 + "\" in the graph!";
      if (!word1Exists) return "No \"" + word1 + "\" in the graph!";
      return "No \"" + word2 + "\" in the graph!"; // Only word2 doesn't exist
    }

    List<String> bridgeWords = graph.findBridgeWords(word1, word2);

    if (bridgeWords.isEmpty()) {
      return "No bridge words from \"" + word1 + "\" to \"" + word2 + "\"!";
    } else {
      StringJoiner sj;
      if (bridgeWords.size() > 1) {
        sj = new StringJoiner(", ", "The bridge words from \"" + word1 + "\" to \"" + word2 + "\" are: ", ".");
        for (int i = 0; i < bridgeWords.size() - 1; i++) {
          sj.add("\"" + bridgeWords.get(i) + "\"");
        }
        sj.add("and \"" + bridgeWords.get(bridgeWords.size() - 1) + "\"");
      } else {
        sj = new StringJoiner("", "The bridge words from \"" + word1 + "\" to \"" + word2 + "\" is: \"", "\".");
        sj.add(bridgeWords.get(0));
      }
      return sj.toString();
    }
  }

  /** Function 4: Generate New Text */
  public static String generateNewText(String inputText) {
    if (graph == null) return "Graph not initialized. Cannot generate text.";
    if (inputText == null || inputText.trim().isEmpty()) return "Input text is empty.";

    String processedInput = inputText.toLowerCase().replaceAll("[^a-z\\s]+", " ").replaceAll("\\s+", " ").trim();
    String[] words = Arrays.stream(processedInput.split("\\s+")).filter(s -> !s.isEmpty()).toArray(String[]::new);

    if (words.length < 2) {
      return inputText; // Not enough words to find bridge words
    }

    StringBuilder newText = new StringBuilder();
    newText.append(words[0]);

    for (int i = 0; i < words.length - 1; i++) {
      String word1 = words[i];
      String word2 = words[i + 1];

      // Only query if both words exist in the graph
      if (graph.containsNode(word1) && graph.containsNode(word2)) {
        List<String> bridges = graph.findBridgeWords(word1, word2);
        if (!bridges.isEmpty()) {
          String bridgeWord = bridges.get(random.nextInt(bridges.size()));
          newText.append(" ").append(bridgeWord);
        }
      }
      newText.append(" ").append(word2);
    }
    return newText.toString();
  }

  /** Function 5 + Optional: Calculate Shortest Path(s) */
  public static String calcShortestPath(String word1, String word2) {
    if (graph == null) return "Graph not initialized.";
    if (word1 == null || word1.isEmpty()) return "Start word cannot be empty.";
    if (!graph.containsNode(word1)) return "Start word \"" + word1 + "\" not found in the graph.";

    if (word2 != null) { // Two words provided
      if (word2.isEmpty()) return "End word cannot be empty if provided.";
      if (!graph.containsNode(word2)) return "End word \"" + word2 + "\" not found in the graph.";
      if (word1.equals(word2)) {
        showDirectedGraphWithPathHighlight(graph, Collections.singletonList(word1));
        return "Shortest path (" + word1 + " to " + word2 + "): " + word1 + "\nLength: 0";
      }

      DirectedGraph.ShortestPathResult result = graph.findShortestPath(word1, word2);

      if (result == null || result.getLength() == Integer.MAX_VALUE) {
        return "No path exists between \"" + word1 + "\" and \"" + word2 + "\".";
      } else {
        StringJoiner sj = new StringJoiner(" -> ");
        result.getPath().forEach(sj::add);
        showDirectedGraphWithPathHighlight(graph, result.getPath());
        return "Shortest path (" + word1 + " to " + word2 + "): " + sj.toString() + "\nLength: " + result.getLength();
      }
    } else { // Optional: One word provided - find paths to all others
      Map<String, DirectedGraph.ShortestPathResult> allPaths = graph.findAllShortestPathsFrom(word1);
      if (allPaths.isEmpty()) return "No paths found from \"" + word1 + "\".";

      StringBuilder resultBuilder = new StringBuilder();
      resultBuilder.append("Shortest paths from \"").append(word1).append("\":\n");
      List<Map.Entry<String, DirectedGraph.ShortestPathResult>> sortedPaths = allPaths.entrySet()
              .stream()
              .filter(entry -> !entry.getKey().equals(word1)) // Exclude path to self
              .sorted(Map.Entry.comparingByKey())
              .collect(Collectors.toList());

      if (sortedPaths.isEmpty()) {
        resultBuilder.append("  (No other reachable nodes)\n");
      } else {
        for (Map.Entry<String, DirectedGraph.ShortestPathResult> entry : sortedPaths) {
          String dest = entry.getKey();
          DirectedGraph.ShortestPathResult pathResult = entry.getValue();
          if (pathResult.getLength() == Integer.MAX_VALUE) {
            resultBuilder.append("  -> \"").append(dest).append("\": Unreachable\n");
          } else {
            StringJoiner sj = new StringJoiner(" -> ");
            pathResult.getPath().forEach(sj::add);
            resultBuilder.append("  -> \"").append(dest).append("\" (Length: ").append(pathResult.getLength()).append("): ").append(sj.toString()).append("\n");
          }
        }
      }
      return resultBuilder.toString();
    }
  }

  /** Optional Function: Calculate ALL Shortest Paths */
  public static String calcAllShortestPaths(String word1, String word2) {
    if (graph == null) return "Graph not initialized.";
    if (word1 == null || word1.isEmpty() || word2 == null || word2.isEmpty()) {
      return "Please provide two valid words.";
    }
    if (!graph.containsNode(word1)) return "Word \"" + word1 + "\" not found in the graph.";
    if (!graph.containsNode(word2)) return "Word \"" + word2 + "\" not found in the graph.";
    if (word1.equals(word2)) {
      return "All shortest paths (" + word1 + " to " + word2 + "):\n1: " + word1 + "\nLength: 0";
    }

    DirectedGraph.AllShortestPathsResult results = graph.findAllShortestPaths(word1, word2);

    if (results == null || results.getPaths().isEmpty()) {
      return "No path exists between \"" + word1 + "\" and \"" + word2 + "\".";
    } else {
      StringBuilder sb = new StringBuilder();
      sb.append("Found ").append(results.getPaths().size()).append(" shortest path(s) between \"")
              .append(word1).append("\" and \"").append(word2).append("\":\n");
      sb.append("Length: ").append(results.getLength()).append("\n");
      int pathNum = 1;
      for (List<String> path : results.getPaths()) {
        StringJoiner sj = new StringJoiner(" -> ");
        path.forEach(sj::add);
        sb.append(pathNum++).append(": ").append(sj.toString()).append("\n");
      }
      return sb.toString();
    }
  }

  /** Helper for shortest path highlighting */
  private static void showDirectedGraphWithPathHighlight(DirectedGraph G, List<String> path) {
    System.out.println("\n--- Graph with Shortest Path Highlighted ---");
    if (G == null || G.isEmpty()) {
      System.out.println("Graph is empty.");
    } else {
      System.out.println(G.toStringWithPath(path));
    }
    System.out.println("-------------------------------------------");
  }


  // MODIFIED: Wrapper for PageRank calculation call
  public static Map<String, Double> calPageRank(boolean useTfIdfBasedInitialRank) { // <--- MODIFIED PARAM NAME
    if (graph == null || graph.isEmpty()) {
      System.out.println("Graph is not initialized or empty. Cannot calculate PageRank.");
      return null;
    }
    // Pass the flag to the graph's PageRank method
    return graph.calculatePageRank(0.85, useTfIdfBasedInitialRank); // <--- MODIFIED CALL
  }

  // ... (randomWalk and saveGraphAsImage methods remain the same) ...
  /** Function 7: Random Walk */
  public static String randomWalk() {
    if (graph == null || graph.isEmpty()) {
      return "Graph is empty, cannot perform random walk.";
    }
    List<String> walkPath = graph.performRandomWalk();
    if (walkPath.isEmpty()) {
      return "Random walk could not start (graph might be empty or have isolated nodes).";
    }

    StringJoiner sj = new StringJoiner(" ");
    walkPath.forEach(sj::add);
    String result = sj.toString();
    String filename = "random_walk_output.txt";
    try (OutputStreamWriter osw = new OutputStreamWriter(new FileOutputStream(filename), StandardCharsets.UTF_8);
         java.io.PrintWriter writer = new java.io.PrintWriter(osw)) {
      writer.println(result);
      System.out.println("Random walk path saved to " + filename);
    } catch (IOException e) {
      System.err.println("Error writing random walk to file '" + filename + "': " + e.getMessage());
    }
    return "Walk path: " + result;
  }


  // --- Updated Function to Save Graph as Image ---

  /**
   * Optional Function: Saves the graph to an image file using graphviz-java.
   * @param G The graph.
   * @param filename The output image filename (e.g., graph.png).
   */
  public static void saveGraphAsImage(DirectedGraph G, String filename) {
    if (G == null || G.isEmpty()) {
      System.out.println("Graph is empty, cannot save image.");
      return;
    }

    // Determine output format from filename extension
    Format format;
    if (filename.toLowerCase().endsWith(".png")) {
      format = Format.PNG;
    } else if (filename.toLowerCase().endsWith(".svg")) {
      format = Format.SVG;
    } else {
      System.err.println("Unsupported image format: " + filename + ". Please use .png, .svg, or .jpg");
      return;
    }

    try {
      // 1. Build the graph object for graphviz-java
      MutableGraph gvGraph = mutGraph("lab1Graph").setDirected(true).use((gr, ctx) -> {
        // Optional global settings
        gr.graphAttrs().add("rankdir", "LR"); // Layout direction
        gr.nodeAttrs().add("shape", "ellipse");
        gr.linkAttrs().add("color", "gray"); // Default edge color

        // Keep track of nodes added to avoid duplicates
        Set<String> nodesAdded = new HashSet<>();

        // Iterate through our graph data (adjList)
        List<String> sortedNodes = G.getNodes().stream().sorted().collect(Collectors.toList());

        for (String sourceWord : sortedNodes) {
          // Ensure source node exists
          Node sourceNode = node(sourceWord);
          if (nodesAdded.add(sourceWord)) {
            gr.add(sourceNode); // Add node explicitly only once if desired
          }

          Map<String, Integer> neighbors = G.getNeighbors(sourceWord);
          if (neighbors.isEmpty() && !nodesAdded.contains(sourceWord)) {
            // Add isolated nodes if not added via edges
            gr.add(node(sourceWord));
            nodesAdded.add(sourceWord);
          } else {
            List<String> sortedNeighbors = neighbors.keySet().stream().sorted().collect(Collectors.toList());
            for (String destWord : sortedNeighbors) {
              int weight = neighbors.get(destWord);
              Node destNode = node(destWord);
              // Ensure destination node exists (optional, linking might add them)
              if (nodesAdded.add(destWord)) {
                gr.add(destNode);
              }
              // Add the link (edge)
              gr.add(sourceNode.link(
                      to(destNode).with(guru.nidi.graphviz.attribute.Label.of(String.valueOf(weight)))
              ));
            }
          }
        }
      });

      // 2. Render the graph to a file
      Graphviz.fromGraph(gvGraph)
              .render(format)
              .toFile(new File(filename));

      System.out.println("Graph image saved successfully to " + filename);

    } catch (IOException e) {
      // Catch IO errors during file writing
      System.err.println("Error saving graph image to '" + filename + "': " + e.getMessage());
    } catch (Exception e) {
      // Catch potential runtime errors from Graphviz execution
      System.err.println("Error generating graph image: " + e.getMessage());
      // Check if it's the common 'dot' command not found error
      if (e.getMessage() != null && e.getMessage().contains("Cannot run program \"dot\"")) {
        System.err.println("----> Please ensure Graphviz is installed and the 'dot' command is in your system's PATH environment variable. <----");
      } else {
        e.printStackTrace(); // Print stack trace for other errors
      }
    }
  }

} // End of Lab1 class