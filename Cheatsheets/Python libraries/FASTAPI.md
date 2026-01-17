## 📌 FastAPI Cheat Sheet

### 🔹 Install
```bash
pip install fastapi uvicorn
```

---

### 🔹 Basic App
```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

# Run with: uvicorn main:app --reload
```

---

### 🔹 Path Parameters
```python
@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
```

---

### 🔹 Request Body
```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float
    is_offer: bool = None

@app.post("/items/")
def create_item(item: Item):
    return item
```

---

### 🔹 Query Parameters
```python
@app.get("/users/")
def read_users(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}
```

---

### 🔹 Response Models
```python
@app.get("/items/{item_id}", response_model=Item)
def read_item(item_id: int):
    return {"name": "Book", "price": 10.5, "is_offer": True}
```

---

### 🔹 Dependency Injection
```python
from fastapi import Depends

def common_parameters(q: str = None, skip: int = 0, limit: int = 10):
    return {"q": q, "skip": skip, "limit": limit}

@app.get("/items/")
def read_items(commons: dict = Depends(common_parameters)):
    return commons
```

---

### 🔹 Middleware
```python
@app.middleware("http")
async def add_process_time_header(request, call_next):
    response = await call_next(request)
    response.headers["X-Process-Time"] = "123ms"
    return response
```

---

### 🔹 Authentication (OAuth2 + JWT)
```python
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.get("/users/me")
def read_users_me(token: str = Depends(oauth2_scheme)):
    return {"token": token}
```

---

### 🔹 Run Server
```bash
uvicorn main:app --reload
```

---

## ⚡ Why FastAPI?
- 🚀 **High performance** (built on Starlette + Pydantic)  
- 📝 **Automatic docs** at `/docs` (Swagger UI) and `/redoc`  
- 🔒 **Type safety** with Python type hints  
- ⚡ **Async support** out of the box  

---
