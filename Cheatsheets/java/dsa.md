I’ll provide you with concise yet in-depth notes on **Data Structures and Algorithms (DSA)** in Java, focusing on key concepts, implementations, and their applications. These notes are structured to help you understand DSA fundamentals and how they’re coded in Java. Let’s dive in!

---

### **1. Introduction to Data Structures and Algorithms**
- **Data Structure**: A way to organize and store data efficiently (e.g., arrays, linked lists, trees).
- **Algorithm**: A step-by-step procedure to solve a problem (e.g., sorting, searching).
- **Why Java for DSA?**: Java is object-oriented, has a rich standard library (e.g., `java.util`), and enforces strong typing, making it great for learning DSA.

**Key Metrics**:
- **Time Complexity**: How execution time grows with input size (e.g., O(n), O(log n)).
- **Space Complexity**: How memory usage grows with input size.

---

### **2. Arrays**
- **Definition**: A fixed-size, contiguous collection of elements of the same type.
- **Java Implementation**:
  ```java
  int[] arr = new int[5]; // Declaration
  arr[0] = 10; // Assigning value
  ```
- **Operations**:
  - Access: O(1) using index.
  - Insertion/Deletion: O(n) due to shifting elements.
- **Use Case**: Fast lookups when size is fixed (e.g., storing roll numbers).

**Example** (Find max element):
```java
public int findMax(int[] arr) {
    int max = arr[0];
    for (int i = 1; i < arr.length; i++) {
        if (arr[i] > max) max = arr[i];
    }
    return max;
}
```

---

### **3. Linked Lists**
- **Definition**: A linear structure where elements (nodes) are linked via pointers.
- **Types**:
  - Singly Linked List: Each node points to the next.
  - Doubly Linked List: Each node points to next and previous.
- **Java Implementation** (Singly Linked List):
  ```java
  class Node {
      int data;
      Node next;
      Node(int data) {
          this.data = data;
          this.next = null;
      }
  }

  class LinkedList {
      Node head;
      void add(int data) {
          Node newNode = new Node(data);
          if (head == null) head = newNode;
          else {
              Node current = head;
              while (current.next != null) current = current.next;
              current.next = newNode;
          }
      }
  }
  ```
- **Operations**:
  - Access: O(n).
  - Insertion/Deletion: O(1) if position known, O(n) otherwise.
- **Use Case**: Dynamic size, frequent insertions (e.g., playlist management).

---

### **4. Stacks**
- **Definition**: A LIFO (Last In, First Out) data structure.
- **Java Implementation** (Using Array):
  ```java
  class Stack {
      int[] arr;
      int top, capacity;
      Stack(int size) {
          arr = new int[size];
          capacity = size;
          top = -1;
      }
      void push(int x) {
          if (top < capacity - 1) arr[++top] = x;
          else System.out.println("Stack Overflow");
      }
      int pop() {
          if (top >= 0) return arr[top--];
          else throw new RuntimeException("Stack Underflow");
      }
  }
  ```
- **Operations**: Push, Pop, Peek - O(1).
- **Use Case**: Function call management, undo operations.

---

### **5. Queues**
- **Definition**: A FIFO (First In, First Out) data structure.
- **Java Implementation** (Using Array):
  ```java
  class Queue {
      int[] arr;
      int front, rear, size, capacity;
      Queue(int capacity) {
          this.capacity = capacity;
          arr = new int[capacity];
          front = 0; rear = -1; size = 0;
      }
      void enqueue(int x) {
          if (size < capacity) arr[++rear] = x;
          else System.out.println("Queue Full");
          size++;
      }
      int dequeue() {
          if (size > 0) {
              size--;
              return arr[front++];
          }
          throw new RuntimeException("Queue Empty");
      }
  }
  ```
- **Operations**: Enqueue, Dequeue - O(1).
- **Use Case**: Task scheduling, breadth-first search.

---

### **6. Trees**
- **Definition**: A hierarchical structure with a root and child nodes.
- **Binary Tree Implementation**:
  ```java
  class TreeNode {
      int val;
      TreeNode left, right;
      TreeNode(int val) {
          this.val = val;
      }
  }
  ```
- **Binary Search Tree (BST)**:
  - Left subtree < Root < Right subtree.
  - Insertion:
    ```java
    TreeNode insert(TreeNode root, int val) {
        if (root == null) return new TreeNode(val);
        if (val < root.val) root.left = insert(root.left, val);
        else root.right = insert(root.right, val);
        return root;
    }
    ```
- **Operations**:
  - Search, Insert, Delete: O(h) where h is height (O(log n) if balanced).
- **Use Case**: Databases, hierarchical data.

---

### **7. Graphs**
- **Definition**: A set of nodes (vertices) connected by edges.
- **Representation**:
  - Adjacency List:
    ```java
    import java.util.*;
    class Graph {
        int V;
        LinkedList<Integer>[] adj;
        Graph(int V) {
            this.V = V;
            adj = new LinkedList[V];
            for (int i = 0; i < V; i++) adj[i] = new LinkedList<>();
        }
        void addEdge(int u, int v) {
            adj[u].add(v);
        }
    }
    ```
- **Traversal**:
  - DFS (Depth-First Search): O(V + E).
    ```java
    void DFS(int v, boolean[] visited) {
        visited[v] = true;
        System.out.print(v + " ");
        for (int u : adj[v]) {
            if (!visited[u]) DFS(u, visited);
        }
    }
    ```
  - BFS (Breadth-First Search): O(V + E).
- **Use Case**: Social networks, shortest path finding.

---

### **8. Sorting Algorithms**
- **Bubble Sort**: O(n²).
  ```java
  void bubbleSort(int[] arr) {
      int n = arr.length;
      for (int i = 0; i < n - 1; i++) {
          for (int j = 0; j < n - i - 1; j++) {
              if (arr[j] > arr[j + 1]) {
                  int temp = arr[j];
                  arr[j] = arr[j + 1];
                  arr[j + 1] = temp;
              }
          }
      }
  }
  ```
- **Quick Sort**: O(n log n) average.
  ```java
  void quickSort(int[] arr, int low, int high) {
      if (low < high) {
          int pi = partition(arr, low, high);
          quickSort(arr, low, pi - 1);
          quickSort(arr, pi + 1, high);
      }
  }
  int partition(int[] arr, int low, int high) {
      int pivot = arr[high];
      int i = low - 1;
      for (int j = low; j < high; j++) {
          if (arr[j] <= pivot) {
              i++;
              int temp = arr[i];
              arr[i] = arr[j];
              arr[j] = temp;
          }
      }
      int temp = arr[i + 1];
      arr[i + 1] = arr[high];
      arr[high] = temp;
      return i + 1;
  }
  ```
- **Use Case**: Sorting data for efficient searching.

---

### **9. Searching Algorithms**
- **Linear Search**: O(n).
  ```java
  int linearSearch(int[] arr, int x) {
      for (int i = 0; i < arr.length; i++) {
          if (arr[i] == x) return i;
      }
      return -1;
  }
  ```
- **Binary Search** (Sorted Array): O(log n).
  ```java
  int binarySearch(int[] arr, int x) {
      int left = 0, right = arr.length - 1;
      while (left <= right) {
          int mid = left + (right - left) / 2;
          if (arr[mid] == x) return mid;
          if (arr[mid] < x) left = mid + 1;
          else right = mid - 1;
      }
      return -1;
  }
  ```

---

### **10. Hashing**
- **Definition**: Mapping data to a fixed-size array using a hash function.
- **Java Implementation** (Using HashMap):
  ```java
  import java.util.HashMap;
  HashMap<Integer, String> map = new HashMap<>();
  map.put(1, "Apple"); // Insert
  String val = map.get(1); // Access
  ```
- **Operations**: Average O(1) for insertion, deletion, lookup.
- **Use Case**: Fast lookups (e.g., caching, dictionaries).

---

### **Key Takeaways**
- Choose the right data structure based on the problem (e.g., arrays for static data, linked lists for dynamic).
- Understand time/space trade-offs for algorithms.
- Practice implementing these in Java to solidify concepts.

Let me know if you’d like a deeper dive into any specific topic or more examples!
