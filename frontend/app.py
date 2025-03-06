import streamlit as st
import requests

# Update this to point to your FastAPI backend
API_BASE_URL = "http://localhost:8001"

# -------------------------------------
# Helper function with caching for API calls
# -------------------------------------
@st.cache_data(show_spinner=False)
def call_api(endpoint, data=None, files=None):
    url = f"{API_BASE_URL}{endpoint}"
    try:
        response = requests.post(url, data=data, files=files)
        response.raise_for_status()  # Raises error for bad responses
        return response.json(), None
    except requests.exceptions.RequestException as e:
        return None, str(e)

# -------------------------------------
# Main Application
# -------------------------------------
def main():
    st.title("Improved Basketball Tracking & Dataset App")
    
    # Sidebar for operation selection (Enhanced UI/UX)
    st.sidebar.header("Operations")
    operations = [
        "Track Video",
        "Create Dataset from Image",
        "Create Dataset from Video",
        "Finalize Dataset (JSON)",
        "Finalize Dataset (Zip)"
    ]
    operation = st.sidebar.selectbox("Select Operation", operations)

    # -------------------------------
    # Operation: Track Video
    # -------------------------------
    if operation == "Track Video":
        st.header("Track Video")
        video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
        if st.button("Run Tracking") and video_file:
            with st.spinner("Processing video..."):
                files = {"file": (video_file.name, video_file, video_file.type)}
                result, error = call_api("/track/", files=files)
            if error:
                st.error(f"Error: {error}")
            else:
                st.success(result.get("message", "Tracking completed"))
                st.json({
                    "player_positions": result.get("player_positions"),
                    "ball_positions": result.get("ball_positions")
                })

    # -------------------------------
    # Operation: Create Dataset from Image
    # -------------------------------
    elif operation == "Create Dataset from Image":
        st.header("Create Dataset from Image")
        base_name = st.text_input("Dataset Base Name", value="my_dataset")
        conf_threshold = st.number_input("Confidence Threshold", value=0.75, min_value=0.0, max_value=1.0, step=0.05)
        image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
        if st.button("Create Dataset from Image") and image_file:
            with st.spinner("Creating dataset from image..."):
                files = {"file": (image_file.name, image_file, image_file.type)}
                data = {"base_name": base_name, "conf_threshold": str(conf_threshold)}
                result, error = call_api("/dataset/create_image/", data=data, files=files)
            if error:
                st.error(f"Error: {error}")
            else:
                st.success(result.get("message", "Dataset created"))
                st.json(result)

    # -------------------------------
    # Operation: Create Dataset from Video
    # -------------------------------
    elif operation == "Create Dataset from Video":
        st.header("Create Dataset from Video")
        base_name = st.text_input("Dataset Base Name", value="my_video_dataset")
        conf_threshold = st.number_input("Confidence Threshold", value=0.75, min_value=0.0, max_value=1.0, step=0.05)
        video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
        if st.button("Create Dataset from Video") and video_file:
            with st.spinner("Processing video dataset..."):
                files = {"file": (video_file.name, video_file, video_file.type)}
                data = {"base_name": base_name, "conf_threshold": str(conf_threshold)}
                result, error = call_api("/dataset/create_video/", data=data, files=files)
            if error:
                st.error(f"Error: {error}")
            else:
                st.success(result.get("message", "Video dataset created"))
                st.json(result)

    # -------------------------------
    # Operation: Finalize Dataset (JSON)
    # -------------------------------
    elif operation == "Finalize Dataset (JSON)":
        st.header("Finalize Dataset (JSON)")
        base_name = st.text_input("Dataset Base Name", value="my_dataset")
        train_ratio = st.number_input("Train Ratio", value=0.7, min_value=0.0, max_value=1.0, step=0.1)
        valid_ratio = st.number_input("Validation Ratio", value=0.2, min_value=0.0, max_value=1.0, step=0.1)
        test_ratio = st.number_input("Test Ratio", value=0.1, min_value=0.0, max_value=1.0, step=0.1)
        if st.button("Finalize Dataset (JSON)"):
            data = {
                "base_name": base_name,
                "train_ratio": str(train_ratio),
                "valid_ratio": str(valid_ratio),
                "test_ratio": str(test_ratio)
            }
            with st.spinner("Finalizing dataset..."):
                result, error = call_api("/dataset/finalize/", data=data)
            if error:
                st.error(f"Error: {error}")
            else:
                st.success(result.get("message", "Dataset finalized"))
                st.json(result)

    # -------------------------------
    # Operation: Finalize Dataset (Zip)
    # -------------------------------
    elif operation == "Finalize Dataset (Zip)":
        st.header("Finalize Dataset (Zip)")
        base_name = st.text_input("Dataset Base Name", value="my_dataset")
        train_ratio = st.number_input("Train Ratio", value=0.7, min_value=0.0, max_value=1.0, step=0.1)
        valid_ratio = st.number_input("Validation Ratio", value=0.2, min_value=0.0, max_value=1.0, step=0.1)
        test_ratio = st.number_input("Test Ratio", value=0.1, min_value=0.0, max_value=1.0, step=0.1)
        if st.button("Finalize and Download Zip"):
            data = {
                "base_name": base_name,
                "train_ratio": str(train_ratio),
                "valid_ratio": str(valid_ratio),
                "test_ratio": str(test_ratio)
            }
            with st.spinner("Finalizing dataset and creating zip..."):
                try:
                    url = f"{API_BASE_URL}/dataset/finalize_zip"
                    response = requests.post(url, data=data)
                    response.raise_for_status()
                    zip_content = response.content
                    zip_filename = f"{base_name}_final_dataset.zip"
                    st.success("Dataset finalized. Download below:")
                    st.download_button(label="Download Zip", data=zip_content, file_name=zip_filename, mime="application/zip")
                except Exception as e:
                    st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
