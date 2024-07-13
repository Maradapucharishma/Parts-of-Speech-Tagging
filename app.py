import streamlit as st
import pickle
from keras.utils import pad_sequences
from keras.models import load_model
import numpy as np

st.title("Parts of Speech Tagger")

with st.sidebar:
    pages = st.radio('Pages', ['Home', 'Individual Word'])

# Global variables to store input and POS results
if 'inp_list' not in st.session_state:
    st.session_state['inp_list'] = []
if 'pos_result_list' not in st.session_state:
    st.session_state['pos_result_list'] = []

def home():
    inp = st.text_input("Enter a string")
    submit_btn = st.button("Submit", key='submit_btn')  # Added key to force refresh
    st.markdown(
        """
        <style>
        .stButton>button {
            border-color: #000000; /* Darker border color */
        }
        </style>
        """, unsafe_allow_html=True)
    
    if submit_btn and inp:
        model = pickle.load(open(r"D:\streamlit\Deep Learning\POS Tag\model.pkl", "rb"))
        tk_x = pickle.load(open(r"D:\streamlit\Deep Learning\POS Tag\tk_x.pkl", "rb"))
        tk_y = pickle.load(open(r"D:\streamlit\Deep Learning\POS Tag\tk_y.pkl", "rb"))

        def pos_tags(inp):
            seq = tk_x.texts_to_sequences([inp])
            text = tk_x.sequences_to_texts(seq)[0].split()  # Get list of tokens
            st.markdown("<h3 style='font-size:24px'>Tokenized input:</h3>", unsafe_allow_html=True)
            st.markdown(", ".join(text), unsafe_allow_html=True)  # Display tokens separated by comma with space
            x = pad_sequences(seq, maxlen=271, padding='post')
            seqs = np.argmax(model.predict(x)[0], axis=1)[np.argmax(model.predict(x)[0], axis=1) != 0]
            pos = tk_y.sequences_to_texts([seqs])[0].split()  # Get list of POS tags
            return list(zip(text, pos))  # Combine tokens and POS tags

        pos_results = pos_tags(inp)
        st.markdown("<h3 style='font-size:24px'>POS TAGS:</h3>", unsafe_allow_html=True)
        for token, pos_tag in pos_results:
            st.markdown(f"<span style='font-size:18px'><strong>{token}</strong> - <strong>{pos_tag}</strong></span>", unsafe_allow_html=True)

        st.session_state['inp_list'] = inp.split()  # Store the input list in the session state
        st.session_state['pos_result_list'] = pos_results  # Store the POS results in the session state

def individual_word():
    if st.session_state['inp_list']:
        word = st.text_input("Enter a word from the sequence")
        sub_btn = st.button("Submit", key='sub_btn')  # Added key to force refresh
        st.markdown(
            """
            <style>
            .stButton>button {
                border-color: #000000; /* Darker border color */
            }
            </style>
            """, unsafe_allow_html=True)

        if word and sub_btn:
            if word in st.session_state['inp_list']:
                i = st.session_state['inp_list'].index(word)
                pos_tag = st.session_state['pos_result_list'][i][1]
                st.markdown(f"<h3 style='font-size:24px'>{word} - {pos_tag}</h3>", unsafe_allow_html=True)
            else:
                st.markdown("<h3 style='font-size:24px'>Enter a word that is in the sequence</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='font-size:24px'>Go to the Home page and input a sequence first</h3>", unsafe_allow_html=True)

if pages == 'Home':
    home()

if pages == 'Individual Word':
    individual_word()