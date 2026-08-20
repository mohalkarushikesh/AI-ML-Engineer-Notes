Algorithm: KNN_Classifier(Training_Data, New_Point, K)
--------------------------------------------------------
1. For each point P in Training_Data:
     Calculate distance between P and New_Point
     Store (distance, P's class label)

2. Sort all stored points by distance in ascending order

3. Select the first K points (the K closest neighbors)

4. Count the frequency of each class label among these K points

5. Return the class label with the highest count (majority vote)
