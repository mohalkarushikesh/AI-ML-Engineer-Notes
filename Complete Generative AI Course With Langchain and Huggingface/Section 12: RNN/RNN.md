### 📝 One‑Hot Encoding Example
Sentence: **“The food is good”**  
- Remove stop word **“is”**  
- Vocabulary = {The, food, good}  

Encoding:
- **The** → `[1 0 0]`  
- **food** → `[0 1 0]`  
- **good** → `[0 0 1]`  

---

👉 **In short:**  
One‑hot encoding turns each word into a vector where **only one position is “1”** (the word’s slot in the vocabulary), and all others ar


### Weight calculation 

<img width="1312" height="512" alt="image" src="https://github.com/user-attachments/assets/0f97af8f-4de6-475e-9ca1-e6998a26c449" />


### Forward Propogation: 

<img width="1199" height="567" alt="image" src="https://github.com/user-attachments/assets/5ac82aa8-f76b-4290-9e85-159ea305eafe" />
<img width="1108" height="264" alt="image" src="https://github.com/user-attachments/assets/99a0d7bc-0c6e-4b7b-bf0f-262d636d36ad" />

### Backword propogation 

<img width="1319" height="624" alt="image" src="https://github.com/user-attachments/assets/72b1f83c-da5e-4d31-b664-91347f9dcd59" />
<img width="1351" height="646" alt="image" src="https://github.com/user-attachments/assets/dae133bd-1fcf-4370-b00b-1873622723b6" />
<img width="1335" height="669" alt="image" src="https://github.com/user-attachments/assets/a7e038d4-eb84-468b-b20d-a0b53ad700ca" />
<img width="1314" height="452" alt="image" src="https://github.com/user-attachments/assets/2d0be567-667c-4460-a732-ad2cfe302b3d" />
<img width="1237" height="642" alt="image" src="https://github.com/user-attachments/assets/6017ac34-7cab-4266-9ea0-4e70e9df05f6" />
<img width="1159" height="261" alt="image" src="https://github.com/user-attachments/assets/a5dca014-e8bd-4fb4-9f22-492496354e74" />

---
### Problems with RNN's 

<img width="1425" height="440" alt="image" src="https://github.com/user-attachments/assets/7d275e37-b379-4c8c-8a82-756724851208" />
<img width="1359" height="537" alt="image" src="https://github.com/user-attachments/assets/76093b1c-c825-4ff2-abd1-1fb0e53fd403" />
<img width="1185" height="217" alt="image" src="https://github.com/user-attachments/assets/f680fc77-8de6-402a-9918-53684e390d71" />
<img width="1294" height="658" alt="image" src="https://github.com/user-attachments/assets/c732c114-9428-47ca-8d73-963274b391dc" />

<img width="1273" height="178" alt="image" src="https://github.com/user-attachments/assets/84d45c6b-f85c-4ea1-bdd9-5e69e8ba188b" />
<img width="1363" height="666" alt="image" src="https://github.com/user-attachments/assets/5c6614ce-945f-4b22-bb63-4e1108d8ef77" />
<img width="1380" height="666" alt="image" src="https://github.com/user-attachments/assets/0d9d2526-e105-4162-b9b2-2af677b3a344" />
<img width="1401" height="301" alt="image" src="https://github.com/user-attachments/assets/eb3ff0cb-8488-4a4c-aea3-eedf5e2c9a65" />


---
