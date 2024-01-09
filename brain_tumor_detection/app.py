import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import requests 
from streamlit_lottie import st_lottie
import json 
import cv2
import base64
import time

from PIL import Image
from tensorflow.keras.models import load_model
import os
# Add background image
# st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)
@st.cache_data
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_base64_of_bin_file("/Users/lovishmiglani/Desktop/brain_tumor_detection/b_g1.jpg")
page_bg_img = f'''
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/png;base64,{img}");
    background-size: cover;
}}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

side_bar_img = get_base64_of_bin_file("/Users/lovishmiglani/Desktop/brain_tumor_detection/b_g1.jpg")
side_bar_bg_img = f'''
<style>
[data-testid="stSidebarContent"] {{
    background-image: url("data:image/png;base64,{side_bar_img}");
    background-size: cover;
}}
</style>
'''
st.markdown(side_bar_bg_img, unsafe_allow_html=True)
# Sidebar menu
with st.sidebar:
    choose = option_menu(menu_title="Main Menu",
                         options=["Home", "Brain Tumor?", "Brain Tumor detection", "Disclaimer"], 
        icons=['house', 'book', 'cloud-upload', 'gear'], 
        menu_icon="cast", 
        default_index= -4,
        styles={
            "container": {"padding": "0!important", "background-color": "rgb(49 51 63)","border-radius": "0.5rem;"},
            "icon": {"color": "orange", "font-size": "18px"},
            "bi-cast": {"color": "orange"},
            "nav-link": {
                "font-size": "15px",
                "text-align": "left",
                "color": "white",
                "margin": "0px",
                "--hover-color": "rgb(37 38 46)",
                "font-family": "'Georgia', sans-serif", 
                "weight": "bold",
            },
            "nav-link-selected": {"background-color": "#7a80df"},
            "menu-title": {
                "color": "rgb(255 255 255)",
                "font-family": "'Georgia', sans-serif",
                "weight": "bold",
            }
            },
        )
    # return choose
    
header_style = f"""
    <style>
        [data-testid="stHeader"] {{
        position: fixed;
        top: 0px;
        left: 0px;
        right: 0px;
        height:  2.875rem;
        background: rgb(216 187 186 / 0%);
        /* outline: none; */
        z-index: 999990;
        display: block;
        position: fixed;
        opacity: 0;
        animation: fadeIn 2s forwards;
        @keyframes fadeIn {{
            to {{
                opacity: 1;
            }}
        }}
        }}
    </style>
"""
st.markdown(header_style, unsafe_allow_html=True)

# Center page title
# url = "https://lottie.host/5bc15346-ea85-4e0e-bb0b-a591d4f1fdd6/U60wGTVp4k.json"  # Correct Lottie URL
# try:
#     response = requests.get(url)
#     response.raise_for_status()  # Raise an HTTPError for bad responses
#     lottie_json = response.json()
#     st_lottie(lottie_json, width=650, height=300)
#     resize_animation = """<style> 
#     div[data-testid="lottie"] {
#         width: 650px !important;
#         height: 300px !important;
#     }
#     </style>"""
# except requests.exceptions.RequestException as e:
#     print(f"Error fetching Lottie animation: {e}")
# response = requests.get(url)
# print(response.content)

model_path = '/Users/lovishmiglani/Desktop/brain_tumor_detection/BrainTumorDetec.h5'

# Check if the file exists
if os.path.exists(model_path):
    # Load the model
    model = load_model(model_path)
    print("Model loaded successfully.")
else:
    print(f"Error: Model file not found at {model_path}")
model = load_model(model_path)
# Load the trained model
# model_path = '/Users/lovishmiglani/Desktop/brain_tumor_detection/BrainTumorDetec.h5'
# model = load_model(model_path)
def make_prediction(img):
    # Assuming your model expects input images of shape (64, 64, 3)
    input_img = cv2.resize(img, (64, 64))
    input_img = np.expand_dims(input_img, axis=0)
    input_img = input_img / 255.0  # Normalize pixel values if needed

    # Use model.predict instead of model.predict_classes for probability output
    prediction = model.predict(input_img)

    # Assuming binary classification, convert to class label (0 or 1)
    pred_class = 1 if prediction > 0.5 else 0

    return pred_class


def show_result(img):
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    pred = make_prediction(img)
    if pred:
        st.write("Tumor Detected")
    else:
        st.write("No Tumor")

def main():
    st.title("Brain Tumor Detector")
    # Home page content
    if choose == "Home":
        st.write("Welcome to the Brain Tumor detection web app!")
        def animate_text(text):
            st.markdown(
                f"""
                <div class="text-animation">
                    <span>{text}</span>
                </div>
                 <style>
                    .text-animation span {{
                    animation: text-anim 9s steps({len(text)}) infinite;
                    white-space: nowrap;
                    overflow: hidden;
                    display: inline-block;
                    vertical-align: middle;
                    text-align: justify;
                    font-size: 20px;
                    color: black;
                    }}
            
                    @keyframes text-anim {{
                        from {{ width: 0; }}
                    to {{ width: 120%; }}
                    }}
                </style>
                """,
                unsafe_allow_html=True
            )

# Example text
        text = "OUR ADVANCED MACHINE LEARNING MODEL LEVERAGES CUTTING-EDGE TECHANALOGY TO DETECT BRAIN TUMORS WITH PRESICION AND EFFICIENCY." 
        # "Developed using state-of-the-art algorithms, our model analyzes medical images to identify potential tumors, aiding healthcare" 
        # "professionals in early diagnosis."

# Display the animated text
        animate_text(text)

        # st.image('B_T2.jpg', caption='Sunrise by the mountains')
        # col1, col2 = st.columns([1, 2], gap="small")
        # col1.image("B_t2.jpg")
        # col1.write("every one has to go")
        # col2.image("B_G.jpg")
        # col2.write("brain")
        # file_ = open("/Users/lovishmiglani/Desktop/brain_tumor_detection/GIF_.gif", "rb")
        # contents = file_.read()
        # data_url = base64.b64encode(contents).decode("utf-8")
        # file_.close()

        # st.markdown(
        #     f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
        #     unsafe_allow_html=True,
        # )
        page_home_style = f"""
        <style>
            [data-testid="StyledLinkIconContainer"] {{
            color: #31333f;
            font-family: 'Georgia', sans-serif;
            text-align: center;
            font-size: 72px;
            text-shadow: 7px 10px 12px #f5f5f5;
                    
            }}
        
            @keyframes fadeIn {{
                to {{
                    opacity: 1;
                }}
            }}
        </style>
        """
        

        st.markdown(page_home_style, unsafe_allow_html=True)
        st.markdown("""
        <style>
        .st-emotion-cache-nahz7x p {
            font-size: 30px;
            text-align: center;
            font-family: 'Georgia', sans-serif;
            word-break: break-word;
            opacity: 0;
            animation: fadeIn 2s forwards;
            text-shadow: -2px 7px 13px #f5f5f5;
        }
        </style>
        """, unsafe_allow_html=True)
        custom_css = f"""
        <style>
            [data-testid="stFileUploadDropzone"] {{
                display: flex;
                -webkit-box-align: center;
                align-items: center;
                padding: 1rem;
                background-color: rgb(18 19 41 / 0%);
                border-radius: 0.5rem;
                color: rgb(250, 250, 250);
            }}
        </style>
        """
        st.markdown(custom_css, unsafe_allow_html=True)          
        
        # page_p_style = f"""
        # <style>
        #     [data-testid="stMarkdownContainer"] {{
        #     text-align: center;
        #     font size: 38px;
        #     }}
        # </style>
        #  """
        # st.markdown(page_p_style, unsafe_allow_html=True)
        st.markdown("""
        <style>
        .st-emotion-cache-5rimss p {
            font-size: 30px;
            text-align: center;
            font-family: 'Georgia', sans-serif;
            opacity: 0;
            font-weight: bold;
            animation: fadeIn 2s forwards;
            text-shadow: 6px 8px 8px #ffffff;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)
    elif choose == "Brain Tumor?":
        page_brain_tumor_style = f"""
        <style>
            div[data-testid="StyledLinkIconContainer"] {{
            color: #31333f;
            font-family: 'Georgia', sans-serif;
            text-align: center;
            font-size: 72px;
            
            text-shadow: 7px 10px 12px #f5f5f5;
                    
            }}

        </style>
        """
        st.markdown(page_brain_tumor_style, unsafe_allow_html=True)
        def scroll_text(text):
            st.markdown(
                f"""
                <div style="width: 100%; overflow: hidden;">
                    <marquee behavior="scroll" direction="left" style="font-size: 30px;color: white;text-shadow: 5px 6px 5px #f5f5f5;font-family: 'Georgia', sans-serif;">
                    {text}
                    </marquee>
                </div>
                """,
                unsafe_allow_html=True
            )

# Example text
        long_text = (
            "Detecting brain tumors early saves lives, and advancements in machine learning empower precise diagnosis and timely intervention."
        )

# Display the scrolling text
        scroll_text(long_text)
        st.markdown("""
        <style>
        .st-emotion-cache-nahz7x p {
            font-family: "Source Code Pro", monospace;
            text-align: justify;
            white-space: break-spaces;
            font-size: 17px;
            font-weight: bold;
            color: #31333f;
            animation: fadeIn 2s forwards;
            text-shadow: 6px 8px 20px #ffffff;
        }
        </style>
        """, unsafe_allow_html=True)            
        st.write("Brain tumors are a complex and challenging medical condition characterized by the abnormal growth of cells within the brain. These tumors can arise from various types of brain tissue and may be either benign or malignant. The impact of a brain tumor on an individual's health can be profound, affecting cognitive functions, motor skills, and overall quality of life. Early detection and accurate diagnosis play a critical role in determining the most effective treatment strategy. Advanced imaging technologies, such as magnetic resonance imaging (MRI) and computed tomography (CT) scans, are instrumental in identifying the location, size, and nature of brain tumors. Treatment options range from surgery and radiation therapy to chemotherapy, often requiring a multidisciplinary approach. Ongoing research in the field aims to enhance our understanding of the underlying causes of brain tumors and develop more targeted and personalized treatment approaches to improve patient outcomes. Despite the complexities associated with brain tumors, advancements in medical science and technology continue to pave the way for improved diagnostic techniques and innovative therapeutic interventions.")

    elif choose == "Brain Tumor detection":
        page_brain_tumor_style = f"""
        <style>
            div[data-testid="StyledLinkIconContainer"] {{
            color: #31333f;
            font-family: 'Georgia', sans-serif;
            text-align: center;
            font-size: 72px;
            opacity: 0;
            animation: fadeIn 2s forwards;
            text-shadow: 7px 10px 12px #f5f5f5;
                    
            }}
        
            @keyframes fadeIn {{
                to {{
                    opacity: 1;
                }}
            }}
        </style>
        """
        st.markdown(page_brain_tumor_style, unsafe_allow_html=True)
        st.write("Drop Your Brain MRI Image")
        # file_ = open("/Users/lovishmiglani/Desktop/brain_tumor_detection/GIF_.gif", "rb")
        # contents = file_.read()
        # data_url = base64.b64encode(contents).decode("utf-8")
        # file_.close()

        # st.markdown(
        #     f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
        #     unsafe_allow_html=True,
        # )
        st.markdown("""
        <style>
        .st-emotion-cache-nahz7x p {
            font-size: 30px;
            text-align: center;
            font-family: 'Georgia', sans-serif;
            word-break: break-word;
            opacity: 0;
            font-weight: bold;
            animation: fadeIn 2s forwards;
            text-shadow: -2px 7px 13px #f5f5f5;
        }
        </style>
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])
        uploader_style = """
        <style>
        div[data-testid="stFileUploader"] {
            background-image: linear-gradient(to bottom, #6db3ffcf, #3701ff40);
            border-radius: 8px;
            box-shadow: 2px 2px 4px #9460b9;
            padding: 8px;
        }
        div[data-baseweb="file-uploader"] {
            align-items: center;
            justify-content: center;   
        } 
        label[data-baseweb="file-uploader-text"] {
            margin: 0 !important;
        }
        </style>
        """
        st.markdown(uploader_style, unsafe_allow_html=True)
        st.markdown("""
                <style>
                .st-emotion-cache-1erivf3 {
                    background-color: rgb(38 39 48 / 22%);
                }
                </style>
                 """, unsafe_allow_html=True)  
        st.markdown("""
                <style>
                .st-emotion-cache-5rimss p {
                    word-break: break-word;
                    font-size: 30px;
                    text-align: center;
                    font-family: 'Georgia', sans-serif;
                    opacity: 0;
                    font-weight: bold;
                    animation: fadeIn 2s forwards;
                    text-shadow: 6px 8px 8px #ffffff;
                    color: white;
                }
                </style>
                 """, unsafe_allow_html=True)    
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image = np.array(image)
            show_result(image)
    elif choose == "Disclaimer":
        page_brain_tumor_style = f"""
        <style>
            div[data-testid="StyledLinkIconContainer"] {{
            color: #31333f;
            font-family: 'Georgia', sans-serif;
            text-align: center;
            font-size: 72px;
            opacity: 0;
            animation: fadeIn 2s forwards;
            text-shadow: 7px 10px 12px #f5f5f5;
                    
            }}
        
            @keyframes fadeIn {{
                to {{
                    opacity: 1;
                }}
            }}
        </style>
        """
        st.markdown(page_brain_tumor_style, unsafe_allow_html=True)
        st.write("Disclaimer:")
        st.markdown("""
        <style>
        .st-emotion-cache-5rimss p {
            font-size: 30px;
            text-align: center;
            font-family: 'Georgia', sans-serif;
            opacity: 0;
            font-weight: bold;
            animation: fadeIn 2s forwards;
            text-shadow: 6px 8px 8px #ffffff;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)
        st.markdown("""
        <style>
        .st-emotion-cache-nahz7x p {
            font-size: 30px;
            text-align: center;
            font-family: 'Georgia', sans-serif;
            word-break: break-word;
            opacity: 0;
            font-weight: bold;
            animation: fadeIn 2s forwards;
            text-shadow: -2px 7px 13px #f5f5f5;
        }
        </style>
        """, unsafe_allow_html=True)

        st.text("The Brain Tumor Detection Model Presented Here Is Intended For Research And Educational Purposes Only. While The Model Has Demonstrated a High level Of accuracy In Its Predictions, It Should Not Be Bonsidered a Substitute For Professional Medical Advice, Diagnosis, Or Treatment. Medical Decisions Should Always be based on the evaluation of a qualified healthcare professional, who can take into account the individual's complete medical history and conduct a comprehensive examination. Users are cautioned against making any healthcare decisions solely based on the output of the model. The model's performance may vary in real-world scenarios, and its use should be complemented by standard medical procedures. Additionally, the model may not account for all possible factors influencing a medical condition. Users are encouraged to consult with healthcare professionals for personalized and accurate medical assessments. The developers and distributors of this model disclaim any responsibility for the misuse or misinterpretation of its results.")
        st.markdown("""
        <style>
        .st-emotion-cache-183lzff {
            font-family: "Source Code Pro", monospace;
            text-align: justify;
            white-space: break-spaces;
            font-size: 17px;
            font-weight: bold;
            color: #31333f;
            animation: fadeIn 2s forwards;
            text-shadow: 6px 8px 20px #ffffff;
        }
        </style>
        """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
