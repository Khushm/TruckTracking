# vit-ai-project

### Problem Statement:
    Identify and count the number of trucks within the given timeframe.

### Specification:
    Detection Algorithm - YOLOv3 Model
    Tracking Logic - based on area, centroid and imageSubtraction methods
    Database - MongoDB

### Accuracy:
    -
### Constraint of the Project:
     1. First occurrence of the truck must be detected else the truck would never get an ID
     2. If area centroid doesn't match and bgSubtraction error is less than threshold then error in passing the truck ID
     3. Day-Night lightening condition error

### Solution:
     1. Improve Truck detection algorithm
     2. Maintain dict of all truck objects detected and match with them, instead of just storing previous image values
     3. None

### Output: 
    -