import cv2
import spacy
import tkinter as tk
from PIL import Image, ImageTk
from transformers import pipeline

# Step 1: Load spaCy for named entity recognition
nlp = spacy.load('en_core_web_sm')

# Step 2: Load Hugging Face Transformers for sentiment analysis
sentiment_analyzer = pipeline('sentiment-analysis')

# Step 3: Define a function for real-time face detection and sentiment analysis
def detect_faces_and_analyze_sentiment():
    _, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Perform face detection
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # Extract text from the detected face region
        face_region_text = pytesseract.image_to_string(gray_frame[y:y+h, x:x+w])
        
        # Perform sentiment analysis on the extracted text
        sentiment_results = sentiment_analyzer(face_region_text)
        
        # Extract named entities from the extracted text
        doc = nlp(face_region_text)
        named_entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        # Display sentiment analysis results and named entities on the GUI
        display_results_on_gui(sentiment_results, named_entities)
    
    # Update the live webcam feed
    photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)
    canvas.image = photo
    root.after(10, detect_faces_and_analyze_sentiment)

# Step 4: Define a function to display sentiment analysis results and named entities on the GUI
def display_results_on_gui(sentiment_results, named_entities):
    # Clear previous results
    text_var.set("")
    
    # Display sentiment analysis results
    for result in sentiment_results:
        text_var.set(text_var.get() + f"Sentiment: {result['label']} ({result['score']:.2f})\n")
    
    # Display named entities
    text_var.set(text_var.get() + "\nNamed Entities:\n")
    for entity in named_entities:
        text_var.set(text_var.get() + f"{entity[0]} ({entity[1]})\n")

# Step 5: Set up the OpenCV video capture and Tkinter GUI
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

root = tk.Tk()
root.title("Real-time Face Sentiment Analysis")

canvas = tk.Canvas(root, width=640, height=480)
canvas.pack()

text_var = tk.StringVar()
result_label = tk.Label(root, textvariable=text_var, font=("Helvetica", 12))
result_label.pack()

# Step 6: Start real-time face detection and sentiment analysis
detect_faces_and_analyze_sentiment()

# Step 7: Run the Tkinter main loop
root.mainloop()

# Release the video capture when the application is closed
cap.release()
