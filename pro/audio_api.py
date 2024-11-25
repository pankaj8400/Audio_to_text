from fastapi import FastAPI, UploadFile, File
import tempfile
import shutil
import os
import concurrent.futures
import speech_recognition as sr
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.probability import FreqDist

# Ensure that NLTK data is downloaded and paths are correctly set
nltk.download('punkt')
nltk.download('stopwords')

app = FastAPI()

@app.get("/")
async def read_root():
    return {"Pankaj": "Welcome to My API!"}

def process_chunk(audio_chunk, results, start_time):
    recognizer = sr.Recognizer()
    try:
        # Convert audio to text
        text = recognizer.recognize_google(audio_chunk, language="en-US")
        results[start_time] = text
    except Exception as e:
        print(f"Error processing chunk: {e}")

def audio_to_text(audio_file_path, chunk_duration=60):
    recognizer = sr.Recognizer()
    results = {}

    with sr.AudioFile(audio_file_path) as source:
        total_duration = source.DURATION
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for i in range(0, int(total_duration), chunk_duration):
                # Calculate the offset for the chunk
                offset = i
                # Move to the offset and record the next chunk
                audio_chunk = recognizer.record(source, duration=chunk_duration, offset=offset)
                futures.append(executor.submit(process_chunk, audio_chunk, results, offset))
            
            # Wait for all chunks to be processed
            concurrent.futures.wait(futures)

    # Combine the results in the correct order
    arranged_text = ""
    for start_time in sorted(results.keys()):
        arranged_text += results[start_time] + " "

    return arranged_text

def summarize_text(text, num_sentences=3):
    if not text:
        return "No content to summarize."

    stop_words = set(stopwords.words("english"))
    words = [word for word in text.split() if word.lower() not in stop_words]
    text = ' '.join(words)

    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    if not sentences:
        return "No sentences found to summarize."

    word_freq = FreqDist(words)
    # Rank sentences based on word frequency within each sentence
    ranked_sentences = sorted(sentences, key=lambda s: sum(word_freq[word.lower()] for word in s.split() if word.lower() in word_freq), reverse=True)
    summary = ' '.join(ranked_sentences[:num_sentences])

    return summary

@app.post("/process_audio/")
async def process_audio(file: UploadFile = File(...)):
    temp_file_path = None
    try:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name

        # Close the file after copying to avoid file locking issues
        file.file.close()

        # Process the audio to text
        transcribed_text = audio_to_text(temp_file_path)
        
        # Summarize the text
        summarized_text = summarize_text(transcribed_text)

        return {"transcribed_text": transcribed_text, "summarized_text": summarized_text}
    finally:
        # Clean up temporary files
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except PermissionError:
                print(f"Could not delete file: {temp_file_path}")
