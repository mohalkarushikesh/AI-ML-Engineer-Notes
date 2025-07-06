**Big Data and Spark with Python**

**Distibuted machines:** 
 - advantage of easily scaling
 - Fault tolerance : if one machine fails, still whole network still go on
  
  **Hadoop** is a way to distribute very large files across multiple machines
  - It uses Hadoop Distributed File System (HDFS)
  - HDFS allows user to work with large dataset
  - HDFS also duplicate blocks of data for fault tolerance 
  - It also then uses MapReduce 
  - Mapreduces allows computations on that data 

  - HDFS uses blocks of data, with a size of data 128MB by default
  - Each of these blocks replicated three times 
  - The blocks are distributed in a way to support fauld tolerance 
  - Smaller blocks provide more parallelization during the processing 
  - Mulitple copies of block prevent loss of data due to failure of a node 
  
**MapReduce** is a way of splitting a computation task to a distributed set of files (HDFS)
 - It consists of job tracker and multiple task trackers 
 - The job tracker sends code to run the multiple task trackers
 - The task trackers allocate CPU and memory for tasks and monitor tasks on the worker nodes 

**Conclusion**: 
 - Using HDFS to distribute large datasets
 - Using MapReduce ro distribute computational task to a distributed dataset 
  
Spark improves on the concept of distribution 
    - lastest technologies being used to quickly and easily handle big-data
    - It's open-source project on Apache & first released in 2013 
    - It is created at AMPLab and UC Berkely
    - Flexible alternative to MapReduce
    - Spark can use data stored in variety of formats ex: Cassandra, AWS S3, HDFS and more
    - MapReduces requires files to be stored in HDFS, Spark does not!
    - Spark also can perform operations upto 100x faster than MapReduce 
  - How does it achives this speed ?
    - MapRedues write most of the data to disk after each map and reduce operation
    - Spark keeps most of the data in memory after each transformation 
    - Spark can spill over data to disk if the memory get's filled 

    - At the core of spark is the idea of **Resilient Distributed Dataset (RDD)**
    - RDD has four mail features: 
      - Distributed collection of data 
      - Fault Tolerant 
      - Parallel Operaion - partioned 
      - Ability to use many data sources
      - RDD are immutable, lazily evaluated and cacheable 
      - There are two types of RDDS operations:
        - **Transformations**:
          - Filter: Applier function to each element and return elements that evaluate to true  
          - Map: Transform each element and preserves # of elements, very similar idea to pandas .apply()
            - Grabbing first letter of a list of names 
          - FlatMap: Transform each element into 0-N elements and changes # of elements 
            - Transforming corpus of text into list of names 
        - **Actions**:
          - First: Return the first element of the RDD 
          - Collect: Return all the elements of the RDD as an array of driver program 
          - Count: Return the count of the RDD
          - Take: Return an array with the first n elements of the RDD 


    - RDDs will be holding their values in tuples 
      - (key, value)
    - This offers better partitioning of data and leads to functionality based on reduction 
    - Reduce() : An action that will aggregate RDD elements using function that returns a single element 
    - ReduceByKey(): An action that will aggregate pair action elements using a function that returns a pair RDD
    - these ideas similar to group of operation 

- Spark ecosystem now includes : 
  - Spark SQL 
  - Spark DataFrames 
  - MLib
  - GraphX
  - Spart streaming 

  
