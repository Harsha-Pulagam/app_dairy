from fastapi import FastAPI,APIRouter
import uvicorn
from inference import Inference
from model import Model
import logging 

logging.basicConfig(level=logging.INFO)

app = FastAPI()
router = APIRouter()
inference = Inference()


@router.get("/")
async def home():
  return {"message": "Machine Learning service"}

@router.post("/wishper")
async def fielpath(path:str):
  try:
    input_text = path
    res = inference.calling_fuction(input_text)
    return res
  except Exception as e:
    logging.error("Something went wrong")
    
    
app.include_router(router)

if __name__ == "__main__":
  uvicorn.run("app:app", reload=True, host="0.0.0.0", port=8051)