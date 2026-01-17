## 📌 OpenCV Cheat Sheet

### 🔹 Installation
```bash
pip install opencv-python
pip install opencv-contrib-python   # for extra modules
```

---

### 🔹 Imports
```python
import cv2
```

---

### 🔹 Reading & Writing Images
```python
img = cv2.imread("image.jpg")          # read image
cv2.imshow("Window", img)              # display image
cv2.waitKey(0)                         # wait for key press
cv2.destroyAllWindows()                # close window

cv2.imwrite("output.jpg", img)         # save image
```

---

### 🔹 Video Capture
```python
cap = cv2.VideoCapture(0)              # 0 = webcam
while True:
    ret, frame = cap.read()
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

---

### 🔹 Image Transformations
```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # grayscale
blur = cv2.GaussianBlur(img, (5,5), 0)         # blur
edges = cv2.Canny(img, 100, 200)               # edge detection
resize = cv2.resize(img, (200,200))            # resize
flip = cv2.flip(img, 1)                        # flip horizontally
```

---

### 🔹 Drawing Shapes & Text
```python
cv2.line(img, (0,0), (200,200), (255,0,0), 5)
cv2.rectangle(img, (50,50), (150,150), (0,255,0), 3)
cv2.circle(img, (100,100), 50, (0,0,255), -1)
cv2.putText(img, "Hello", (10,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
```

---

### 🔹 Thresholding & Contours
```python
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0,255,0), 2)
```

---

### 🔹 Face Detection (Haar Cascades)
```python
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
```

---

### 🔹 Key Functions Summary
- `cv2.imread()` → read image  
- `cv2.imshow()` → show image  
- `cv2.VideoCapture()` → capture video  
- `cv2.cvtColor()` → convert color spaces  
- `cv2.GaussianBlur()` → blur image  
- `cv2.Canny()` → edge detection  
- `cv2.findContours()` → detect contours  
- `cv2.CascadeClassifier()` → object detection  

---

## ⚡ Why OpenCV?
- Fast, optimized C++ backend with Python bindings  
- Huge set of tools for image/video processing  
- Works with deep learning frameworks (TensorFlow, PyTorch) for advanced CV tasks  

---
