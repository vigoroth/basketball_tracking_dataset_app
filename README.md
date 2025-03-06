# Basketball Tracking and Dataset Creation App

## Description
A web application for basketball object detection, tracking, and automated dataset creation using YOLO object detection models. Built with **FastAPI** (backend) and **Streamlit** (frontend).

## Features
- **Object detection:** Detect and track basketball objects including ball, player, rim, shot (made), and other basketball-related elements using YOLO models.
- **Automated Dataset Generation:** Create datasets from images/videos, automatically annotated in YOLO format.
- **Dataset Splitting:** Generate finalized datasets with customizable train, validation, and test splits, provided as JSON or downloadable ZIP files.

## Class Names for YOLO Detection
The detection model (`detection_model.pt`) supports the following classes:
```python
["ball", "made", "person", "rim", "shoot"]
```


#Setup Instructions:
```bash
 git clone <repo_link>
cd basketball_tracking_dataset_app
pip install -r requirements.txt
```



#Run Backend (FastAPI):

```bash
uvicorn backend.main:app --reload --port 8001
```

#Run Frontend (Streamlit):
```bash
streamlit run app.py
```

#REQUIERMENTS

---

```code
streamlit
fastapi
uvicorn
ultralytics
opencv-python
numpy
requests
pyyaml
python-multipart
```



#Project Structure:


![image](https://github.com/user-attachments/assets/5b0fe7f9-1e47-453a-ae3a-e66468435340)


