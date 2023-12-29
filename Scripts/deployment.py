import streamlit as st
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import pipeline
import io

import pytube
import assemblyai as aai

aai.settings.api_key = f"59633312167e4650841167043bb6878c"

@st.cache_data()
def load_model():
    wow = pipeline("summarization", model="Heavycoder/snp_model")
    return wow

def download_audio_from_youtube(link):
    # Download the YouTube video
    video = pytube.YouTube(link)
    # Get the audio stream with the highest bitrate
    audio_streams = video.streams.filter(only_audio=True)
    # Choose the desired audio stream (e.g., the one with the highest bitrate)
    audio_stream = audio_streams.get_audio_only()
    # Download the audio stream
    audio_stream.download()
    # Get the file path of the downloaded audio
    file_path = audio_stream.default_filename
    return file_path

# def main():
#     st.title("Text Summarization with Pegasus")
    
#     # Text input
#     input_text = st.text_area("Enter the text to summarize", height=200)
    
#     # Summarize button
#     if st.button("Summarize"):
#         if input_text:
#             # # Load the model and tokenizer
#             # model, tokenizer = load_model()
            
#             # # Tokenize the input text
#             # inputs = tokenizer([input_text], truncation=True, padding="longest", max_length=4096, return_tensors="pt")
            
#             # # Generate the summary
#             # summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=150, early_stopping=True)
#             # summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#             gen_kwargs = {"length_penalty": 0.8, "num_beams":4, "max_length": 150}

#             pipe = load_model()
#             summary = pipe(input_text, **gen_kwargs)[0]["summary_text"]
            
#             # Display the summary
#             st.subheader("Summary")
#             st.write(summary)
#         else:
#             st.warning("Please enter some text to summarize.")

# if __name__ == "__main__":
#     main()


def main():
    st.title("Text Summarizer")

    # Create sidebar with tabs
    st.sidebar.title("Menu")
    tab = st.sidebar.radio("Select Option", ("Enter Text", "Upload MP3", "Insert YouTube Link"))



    if tab == "Enter Text":
        # Text input
        input_text = st.text_area("Enter the text to summarize", height=200)

        # Summarize button
        if st.button("Summarize"):
            if input_text:
                model = load_model()
                summary = model(input_text, max_length=150, num_beams=4, length_penalty=0.8, early_stopping=True)[0]["summary_text"]

                # Display the summary
                st.subheader("Summary")
                st.write(summary)
            else:
                st.warning("Please enter some text to summarize.")

    elif tab == "Upload MP3":
        # File upload
        mp3_file = st.file_uploader("Upload MP3 file", type=["mp3", "wav"])
        if mp3_file is not None:
            # Open a file on your local system
            with open("uploaded_file.mp3", "wb") as file:
                # Write the contents of the uploaded file into the local file
                file.write(mp3_file.read())

        if st.button('Summarize'):
            if mp3_file is not None:
                transcriber = aai.Transcriber()
                transcript = transcriber.transcribe("uploaded_file.mp3")
                if transcript.text:
                    model = load_model()
                    summary = model(transcript.text, max_length=150, num_beams=4, length_penalty=0.8, early_stopping=True)[0]["summary_text"]

                    # Display the summary
                    st.subheader("Summary")
                    st.write(summary)
                else:
                    st.warning("Empty Input")
            else:
                st.warning('mp3 file is empty!!')


    elif tab == "Insert YouTube Link":
        # Text input for link
        yt_link = st.text_input("Insert link for text summarization")

        if st.button("Summarize"):
            if yt_link:
                audio_file = download_audio_from_youtube(yt_link)
                if audio_file is not None:
                    transcriber = aai.Transcriber()
                    transcript = transcriber.transcribe(audio_file)
                    if transcript.text:
                        model = load_model()
                        summary = model(transcript.text, max_length=150, num_beams=4, length_penalty=0.8, early_stopping=True)[0]["summary_text"]

                        # Display the summary
                        st.subheader("Summary")
                        st.write(summary)
                    else:
                        st.warning("Empty input")
                else:
                    st.warning('mp3 file is empty!!')


if __name__ == "__main__":
    main()
