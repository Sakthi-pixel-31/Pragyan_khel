import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile

st.set_page_config(layout="wide")
st.title("🎬 Cinematic Focus — Select Subject")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n-seg.pt")

model = load_model()

# ---------------- HELPERS ----------------
def apply_focus(frame, results, selected_id):
    if results.masks is None:
        return frame

    masks = results.masks.data.cpu().numpy()
    boxes = results.boxes.xyxy.cpu().numpy()

    if selected_id >= len(masks):
        return frame

    # Selected mask
    focus_mask = masks[selected_id]

    focus_mask = cv2.resize(
        focus_mask,
        (frame.shape[1], frame.shape[0])
    )

    focus_mask = (focus_mask > 0.5).astype(np.uint8)

    # Blur background
    blurred = cv2.GaussianBlur(frame, (55, 55), 0)

    mask3 = np.repeat(focus_mask[:, :, None], 3, axis=2)

    output = np.where(mask3 == 1, frame, blurred)

    return output.astype(np.uint8)


def draw_boxes(frame, results):
    if results.boxes is None:
        return frame, []

    boxes = results.boxes.xyxy.cpu().numpy()
    labels = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"ID {i}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        labels.append(f"Object {i}")

    return frame, labels


# ---------------- UI ----------------
mode = st.radio(
    "Choose Mode",
    ["Upload Video", "Use Webcam"]
)

selected_object = None

if mode == "Upload Video":

    uploaded_file = st.file_uploader(
        "Upload video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_file is not None:

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)

        frame_placeholder = st.empty()
        select_placeholder = st.empty()

        selected_index = 0
        selector_created = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 360))

            results = model(frame, verbose=False)[0]

            # Draw boxes for selection
            preview_frame, labels = draw_boxes(frame.copy(), results)

            # Create selector once objects appear
            if (not selector_created) and len(labels) > 0:
                selected_index = select_placeholder.selectbox(
                    "Select object to focus",
                    range(len(labels)),
                    format_func=lambda x: labels[x]
                )
                selector_created = True

            # Apply focus
            if selector_created:
                output_frame = apply_focus(frame, results, selected_index)
            else:
                output_frame = preview_frame

            frame_placeholder.image(
                cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB),
                use_container_width=True
            )

        cap.release()


else:
    st.info("⚠️ Webcam in Streamlit is limited. Use upload for best results.")