from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
from face_attendance import recognize_and_log

app = FastAPI()

# ✅ إعداد CORS للسماح بالوصول من أي دومين
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # يمكنك تحديد ["http://localhost:5500"] أو أي دومين
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📸 API لرفع الصورة والتعرف على الوجه
@app.post("/recognize/")
async def recognize(file: UploadFile = File(...)):
    try:
        with open("temp.jpg", "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        name = recognize_and_log("temp.jpg")
        return JSONResponse(content={"name": name})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
