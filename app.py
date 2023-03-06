import streamlit as st
from PIL import Image
from model import predict

st.set_option("deprecation.showfileUploaderEncoding", False)
st.sidebar.title("Take a photo of your pets")
st.sidebar.write("Let me make a guess about breeds of your pets")
st.sidebar.write("")
img_source = st.sidebar.radio("How upload?",("select file","take a photo"))
if img_source == "select file":
    img_file = st.sidebar.file_uploader("Choose the photo",type=["png","jpg","jpeg"])
elif img_source == "take a photo":
    img_file = st.camera_input("Let's shoot")

if img_file is not None:
    with st.spinner("guessing..."):
        img = Image.open(img_file)
        st.image(img, caption="Target", width=480)
        st.write("")
        result = predict(img)
        a = result[0]
        b = result[1]
        c = result[2]
        st.write(f"First: {a}")
        st.write(f"Second: {b}")
        st.write(f"Third: {c}")

st.sidebar.caption("""Dataset : The Oxford-IIIT Pet Dataset\n
Licence is Creative Commons Attribution-ShareAlike 4.0 International License.""")