#
from fastapi import FastAPI, HTTPException

app = FastAPI()
global_model = {"model_data": b"global model"}

@app.post("/upload_model/")
async def upload_model(client_id: str, model_data: bytes):
    print(f"Received update from {client_id}")
    # 处理模型数据
    global_model["model_data"] = b"new model data"
    return {"message": "Model update received"}

@app.get("/get_global_model/")
async def get_global_model():
    return global_model
