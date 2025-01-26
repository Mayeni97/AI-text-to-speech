import streamlit as st
from transformers import pipeline
import torch
from pathlib import Path
import os
from dotenv import load_dotenv
from datasets import load_dataset
import soundfile as sf
import numpy as np
import docx
import PyPDF2
import io

# Load environment variables
load_dotenv()

# Create necessary directories
output_dir = Path(os.getenv('OUTPUT_PATH', 'output'))
models_dir = Path(os.getenv('MODELS_PATH', 'models'))
output_dir.mkdir(exist_ok=True)
models_dir.mkdir(exist_ok=True)

class TextToSpeechApp:
    def __init__(self):
        self.load_models()
        
    def load_models(self):
        # Initialize the models
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
        # Load speaker embeddings from CMU Arctic dataset
        try:
            # Load the dataset with cache_dir set to models directory
            embeddings_dataset = load_dataset(
                "Matthijs/cmu-arctic-xvectors", 
                split="validation",
                cache_dir=models_dir
            )
            # Select a specific speaker embedding (index 7306 corresponds to a specific speaker)
            speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
            st.success("Successfully loaded speaker embeddings")
        except Exception as e:
            st.warning("Using default speaker embeddings due to loading error")
            # Use a default speaker embedding if loading fails
            speaker_embeddings = torch.zeros((1, 512))  # Default size for speecht5 speaker embeddings
        
        # Initialize TTS with speaker embeddings
        try:
            self.tts = pipeline("text-to-speech", model="microsoft/speecht5_tts")
            self.speaker_embeddings = speaker_embeddings
        except Exception as e:
            st.error(f"Failed to initialize text-to-speech model: {str(e)}")
            raise
        
    def process_text(self, text):
        # Summarize the text
        summary = self.summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
        
        # Convert summary to speech with speaker embeddings
        speech = self.tts(summary, forward_params={"speaker_embeddings": self.speaker_embeddings})
        
        return summary, speech
    
    def save_audio(self, speech, filename="output.wav"):
        # Save the audio file in the output directory
        filepath = output_dir / filename
        # Convert the audio data to the correct format and save it
        audio_data = speech["audio"]
        if isinstance(audio_data, np.ndarray):
            sf.write(str(filepath), audio_data, speech["sampling_rate"])
        else:
            audio_data.save(str(filepath))
        return str(filepath)

def main():
    st.title("AI Text-to-Speech Summarizer")
    
    # Initialize app
    app = TextToSpeechApp()
    
    # Create input options
    input_option = st.radio("Choose input method:", ["Text Input", "File Upload"])
    
    if input_option == "Text Input":
        text = st.text_area("Enter your text:", height=200)
    else:
        uploaded_file = st.file_uploader("Upload a file", type=['txt', 'pdf', 'doc', 'docx'])
        if uploaded_file:
            # Get the file extension
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            # Process different file types
            if file_extension == 'txt':
                text = uploaded_file.read().decode()
            elif file_extension == 'pdf':
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
                text = '\n'.join([page.extract_text() for page in pdf_reader.pages])
            elif file_extension in ['doc', 'docx']:
                doc = docx.Document(io.BytesIO(uploaded_file.read()))
                text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        else:
            text = ""
    
    if st.button("Process") and text:
        try:
            with st.spinner("Processing..."):
                # Process the text
                summary, speech = app.process_text(text)
                
                # Display summary
                st.subheader("Summary:")
                st.write(summary)
                
                # Save and play audio
                audio_file = app.save_audio(speech)
                
                # Display audio player
                st.subheader("Audio:")
                st.audio(audio_file)
                
                # Add download button
                with open(audio_file, "rb") as file:
                    st.download_button(
                        label="Download Audio",
                        data=file,
                        file_name="summary_audio.wav",
                        mime="audio/wav"
                    )
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()