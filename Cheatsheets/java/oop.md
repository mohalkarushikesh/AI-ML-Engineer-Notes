Below is an in-depth set of notes on Object-Oriented Programming (OOP) in Java. These notes cover the core principles, key concepts, and practical examples to help you understand OOP in the context of Java. Since Java is a fully object-oriented language, mastering these concepts is fundamental to writing efficient and maintainable code.

---

### **1. Introduction to OOP in Java**
- **What is OOP?**
  - Object-Oriented Programming is a paradigm that organizes code into objects, which are instances of classes.
  - It emphasizes modularity, reusability, and abstraction through real-world modeling.

- **Why Java Uses OOP?**
  - Java enforces OOP principles to ensure robust, scalable, and maintainable applications.
  - Everything in Java (except primitives) is an object.

- **Core Principles of OOP**:
  1. **Encapsulation**
  2. **Inheritance**
  3. **Polymorphism**
  4. **Abstraction**

---

### **2. Core OOP Concepts in Java**

#### **2.1 Classes and Objects**
- **Class**: A blueprint for creating objects. It defines properties (fields) and behaviors (methods).
- **Object**: An instance of a class, representing a specific entity.
- **Syntax**:
  ```java
  class ClassName {
      // Fields (attributes)
      datatype fieldName;

      // Constructor
      ClassName() {
          // Initialization code
      }

      // Methods (behaviors)
      returnType methodName() {
          // Logic
      }
  }
  ```
- **Example**:
  ```java
  class Car {
      String brand;  // Field
      int speed;

      // Constructor
      Car(String brand, int speed) {
          this.brand = brand;  // 'this' refers to current instance
          this.speed = speed;
      }

      // Method
      void display() {
          System.out.println("Brand: " + brand + ", Speed: " + speed);
      }
  }

  public class Main {
      public static void main(String[] args) {
          Car car1 = new Car("Toyota", 120);  // Object creation
          car1.display();  // Output: Brand: Toyota, Speed: 120
      }
  }
  ```

#### **2.2 Encapsulation**
- **Definition**: Bundling data (fields) and methods that operate on that data into a single unit (class), while restricting direct access to some components.
- **Purpose**: Hide implementation details and protect data integrity.
- **How Achieved**:
  - Use **access modifiers** (`private`, `protected`, `public`).
  - Provide **getters** and **setters** to access private fields.
- **Example**:
  ```java
  class Employee {
      private String name;  // Private field
      private double salary;

      // Constructor
      Employee(String name, double salary) {
          this.name = name;
          this.salary = salary;
      }

      // Getter
      public String getName() {
          return name;
      }

      // Setter with validation
      public void setSalary(double salary) {
          if (salary > 0) {
              this.salary = salary;
          }
      }

      public double getSalary() {
          return salary;
      }
  }

  public class Main {
      public static void main(String[] args) {
          Employee emp = new Employee("Alice", 50000);
          emp.setSalary(60000);
          System.out.println(emp.getName() + " earns " + emp.getSalary());
      }
  }
  ```
- **Key Points**:
  - `private`: Accessible only within the class.
  - `public`: Accessible everywhere.
  - `this`: Resolves ambiguity between fields and parameters.

#### **2.3 Inheritance**
- **Definition**: A mechanism where one class (subclass/child) inherits fields and methods from another class (superclass/parent).
- **Purpose**: Promote code reuse and establish a hierarchical relationship.
- **Syntax**: Use the `extends` keyword.
- **Example**:
  ```java
  class Animal {
      String name;

      Animal(String name) {
          this.name = name;
      }

      void eat() {
          System.out.println(name + " is eating.");
      }
  }

  class Dog extends Animal {
      Dog(String name) {
          super(name);  // Call parent constructor
      }

      void bark() {
          System.out.println(name + " is barking.");
      }
  }

  public class Main {
      public static void main(String[] args) {
          Dog dog = new Dog("Rex");
          dog.eat();  // Inherited method
          dog.bark(); // Child-specific method
      }
  }
  ```
- **Key Points**:
  - `super`: Refers to the superclass (e.g., to call parent constructor or methods).
  - Java supports **single inheritance** (one class can inherit from only one parent).
  - All classes implicitly inherit from `java.lang.Object`.

#### **2.4 Polymorphism**
- **Definition**: The ability of an object to take on many forms. In Java, it’s achieved through method overloading and overriding.
- **Types**:
  1. **Compile-Time Polymorphism (Overloading)**:
     - Multiple methods with the same name but different parameters.
     - Example:
       ```java
       class Calculator {
           int add(int a, int b) {
               return a + b;
           }

           double add(double a, double b) {
               return a + b;
           }
       }

       public class Main {
           public static void main(String[] args) {
               Calculator calc = new Calculator();
               System.out.println(calc.add(5, 3));      // Output: 8
               System.out.println(calc.add(5.5, 3.2)); // Output: 8.7
           }
       }
       ```
  2. **Run-Time Polymorphism (Overriding)**:
     - Subclass provides a specific implementation of a method defined in the superclass.
     - Requires **inheritance** and the `@Override` annotation (optional but recommended).
     - Example:
       ```java
       class Animal {
           void sound() {
               System.out.println("Some generic sound");
           }
       }

       class Cat extends Animal {
           @Override
           void sound() {
               System.out.println("Meow");
           }
       }

       public class Main {
           public static void main(String[] args) {
               Animal myCat = new Cat(); // Upcasting
               myCat.sound(); // Output: Meow (run-time decision)
           }
       }
       ```
- **Key Points**:
  - **Upcasting**: Treating a subclass object as a superclass type.
  - **Dynamic Method Dispatch**: JVM decides which overridden method to call at runtime.

#### **2.5 Abstraction**
- **Definition**: Hiding complex implementation details and showing only essential features.
- **How Achieved**:
  - **Abstract Classes**: Cannot be instantiated; may contain abstract methods.
  - **Interfaces**: Purely abstract; define a contract for classes to implement.
- **Abstract Class Example**:
  ```java
  abstract class Shape {
      abstract double area(); // Abstract method (no body)

      void display() { // Concrete method
          System.out.println("This is a shape.");
      }
  }

  class Circle extends Shape {
      double radius;

      Circle(double radius) {
          this.radius = radius;
      }

      @Override
      double area() {
          return Math.PI * radius * radius;
      }
  }

  public class Main {
      public static void main(String[] args) {
          Circle circle = new Circle(5);
          circle.display();
          System.out.println("Area: " + circle.area());
      }
  }
  ```
- **Interface Example** (Java 8+ allows default methods):
  ```java
  interface Vehicle {
      void start(); // Abstract method

      default void stop() { // Default method
          System.out.println("Vehicle stopped.");
      }
  }

  class Bike implements Vehicle {
      @Override
      public void start() {
          System.out.println("Bike started.");
      }
  }

  public class Main {
      public static void main(String[] args) {
          Bike bike = new Bike();
          bike.start(); // Output: Bike started.
          bike.stop();  // Output: Vehicle stopped.
      }
  }
  ```
- **Key Points**:
  - Use `abstract` keyword for abstract classes and methods.
  - Use `implements` for interfaces (a class can implement multiple interfaces).
  - Java 8 introduced `default` and `static` methods in interfaces.

---

### **3. Additional OOP Features in Java**

#### **3.1 Constructors**
- Special methods called when an object is created.
- **Types**:
  - **Default Constructor**: No parameters, provided by Java if none defined.
  - **Parameterized Constructor**: Takes arguments to initialize fields.
- **Example**:
  ```java
  class Student {
      String name;
      int age;

      Student() { // Default
          name = "Unknown";
          age = 0;
      }

      Student(String name, int age) { // Parameterized
          this.name = name;
          this.age = age;
      }
  }
  ```

#### **3.2 Method Overloading vs. Overriding**
- **Overloading**: Same method name, different parameters (compile-time).
- **Overriding**: Same method name and signature, different implementation (run-time).

#### **3.3 Static Keyword**
- **Purpose**: Belongs to the class, not instances.
- **Usage**:
  - Static fields: Shared across all objects.
  - Static methods: Can be called without creating an object.
- **Example**:
  ```java
  class Counter {
      static int count = 0;

      Counter() {
          count++;
      }

      static void showCount() {
          System.out.println("Total objects: " + count);
      }
  }

  public class Main {
      public static void main(String[] args) {
          Counter c1 = new Counter();
          Counter c2 = new Counter();
          Counter.showCount(); // Output: Total objects: 2
      }
  }
  ```

#### **3.4 Final Keyword**
- **Purpose**: Prevents modification or extension.
- **Usage**:
  - `final` variable: Constant (cannot be reassigned).
  - `final` method: Cannot be overridden.
  - `final` class: Cannot be inherited.
- **Example**:
  ```java
  final class Immutable {
      final int value = 10;
  }
  ```

#### **3.5 Packages**
- **Purpose**: Organize classes and avoid naming conflicts.
- **Syntax**: `package com.example;`
- **Import**: `import com.example.ClassName;`

---

### **4. Best Practices in Java OOP**
- **Follow Naming Conventions**: 
  - Classes: PascalCase (e.g., `MyClass`).
  - Methods/Variables: camelCase (e.g., `myMethod`).
- **Encapsulate Properly**: Use private fields with public getters/setters.
- **Favor Composition Over Inheritance**: Use “has-a” relationships instead of “is-a” when possible.
- **Use Interfaces**: For flexibility and multiple inheritance-like behavior.
- **Keep Classes Small and Focused**: Single Responsibility Principle (SRP).

---

### **5. Example Combining OOP Concepts**
Here’s a practical example combining all principles:
```java
// Abstract class
abstract class Animal {
    protected String name;

    Animal(String name) {
        this.name = name;
    }

    abstract void makeSound(); // Abstract method
}

// Interface
interface Movable {
    void move();
}

// Dog class inheriting and implementing
class Dog extends Animal implements Movable {
    private int speed; // Encapsulation

    Dog(String name, int speed) {
        super(name);
        this.speed = speed;
    }

    @Override
    void makeSound() { // Polymorphism (overriding)
        System.out.println(name + " barks.");
    }

    @Override
    public void move() {
        System.out.println(name + " runs at " + speed + " km/h.");
    }
}

public class Main {
    public static void main(String[] args) {
        Dog dog = new Dog("Buddy", 30);
        dog.makeSound(); // Output: Buddy barks.
        dog.move();      // Output: Buddy runs at 30 km/h.

        // Upcasting
        Animal animal = dog;
        animal.makeSound(); // Output: Buddy barks.
    }
}
```

---

These notes provide a solid foundation for understanding OOP in Java. Let me know if you’d like deeper exploration into topics like design patterns, exception handling, or advanced Java features!
