import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import cv2


@st.cache_resource
def load_model():
    return YOLO("best(5).pt")

model = load_model()

disease_classes = ['aphids','bacterial_wilt','black_fungus_groundnut','bud_rot','leaf_fungus','leaf_miner','red_hairy_caterpillar','root_rot','rosette','rust','stunt_virus','tikka','tobacco_caterpillar','white_grub'        
]

st.title("🌾Groundnut Pest & Disease Detection")
st.info("ℹ️ **Note:** This model can detect only these diseases: aphids,bacterial_wilt,black_fungus_groundnut,bud_rot,leaf_fungus,leaf_miner,red_hairy_caterpillar,root_rot,rosette,rust,stunt_virus,tikka,tobacco_caterpillar,white_grub")

# st.write("Upload an image to detect Pest & Disease")
uploaded_file = st.file_uploader("Upload an Image(Wheat)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image (Original)", use_column_width=True)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp_file.write(uploaded_file.read())
    temp_file.close()

    # Run YOLO inference
    st.write("🔍 Detecting diseases...")
    results = model.predict(source=temp_file.name, conf=0.25, save=False)

    # Show detection results
    for r in results:
        # Annotated image (BGR by default)
        im_array = r.plot()
        # Convert back to RGB so colors match original
        im_array = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)

        st.image(im_array, caption="Detected Diseases", use_column_width=True)

        # Extract detected class names
        detected = [disease_classes[int(c)] for c in r.boxes.cls.cpu().numpy()]
        st.subheader("🦠 Detected Diseases:")
        if detected:
            st.write(", ".join(set(detected)))
        else:
            st.write("✅ No disease detected!")

    # Cleanup temp file
    os.remove(temp_file.name)
