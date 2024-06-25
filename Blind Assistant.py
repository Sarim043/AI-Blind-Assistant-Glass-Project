import cv2
from PIL import Image
import numpy as np
import torch
import easyocr
import pyttsx3
import pyaudio
import vosk
import json
import time
from fuzzywuzzy import fuzz
from transformers import DetrImageProcessor, DetrForObjectDetection, pipeline
import winsound
import keyboard

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Initialize the DETR processor and model for object detection
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

# Initialize the Vosk model
model_vosk = vosk.Model("vosk-model-small-en-us-0.15")

# Initialize the audio stream
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=5000)

# Initialize the Image Captioning pipeline
captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

# Initialize the recognizer
recognizer = vosk.KaldiRecognizer(model_vosk, 16000)

# Function to play a tune message
def play_tune(reverse=False):
    # Define frequencies for the tune (you can adjust these as needed)
    frequencies = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88]
    durations = [200, 200, 200, 200, 200, 200, 200]  # Duration in milliseconds
    
    # Reverse the frequencies and durations if reverse flag is True
    if reverse:
        frequencies.reverse()
        durations.reverse()
    
    # Play each frequency with the corresponding duration
    for frequency, duration in zip(frequencies, durations):
        winsound.Beep(int(frequency), duration)
        time.sleep(0.1)  # Add a short delay between beeps

# Function to recognize "ON" and "OFF" commands
def listen_for_on_off():
    while True:
        stream.start_stream()
        data = stream.read(16000 * 5)  # Read data for 5 seconds
        stream.stop_stream()
        
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            text = result['text']
            print("Recognized Speech:", text)
            if fuzz.partial_ratio('Hello', text.lower()) > 50:
                speak("Opening")
                play_tune()
                return True
            elif fuzz.partial_ratio('turn off', text.lower()) > 50:
                speak("Closing")
                return False
        else:
            print("Sorry, I couldn't understand. Please speak clearly.")

# Function to recognize voice commands with enhanced fuzzy matching
def perform_voice_recognition(choices):
    while True:
        stream.start_stream()
        data = stream.read(16000 * 5)  # Read data for 5 seconds
        stream.stop_stream()
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            text = result['text']
            print("Recognized Speech:", text)
            best_match = max(choices, key=lambda choice: fuzz.partial_ratio(choice, text.lower()))
            if fuzz.partial_ratio(best_match, text.lower()) > 50:  # Adjusted threshold
                return best_match
            else:
                print("Sorry, I couldn't understand. Please try again.")
        else:
            print("Sorry, I couldn't understand. Please try again.")
            continue

# Function to perform object detection
def detect_objects(image):
    inputs = processor(images=image, return_tensors="pt")
    outputs = model.forward(inputs['pixel_values'])
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.8)[0]
    detected_objects = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        detected_objects.append({
            "label": model.config.id2label[label.item()],
            "confidence": round(score.item(), 3),
            "bounding_box": box
        })
    if detected_objects:
        return max(detected_objects, key=lambda x: x['confidence'])
    else:
        return None

# Function to perform Optical Character Recognition (OCR)
def perform_ocr(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = reader.readtext(rgb_image)
    print("Extracted Text:")
    for result in results:
        print(result[1])
    return [result[1] for result in results]

# Function to perform image captioning
def perform_image_captioning():
    cap = cv2.VideoCapture(0)  # Open the default camera
    if not cap.isOpened():
        return "Failed to open camera"
    
    ret, frame = cap.read()  # Capture frame-by-frame
    cap.release()  # Release the capture
    cv2.destroyAllWindows()  # Close the window
    
    if ret:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = captioner(image)
        print("Result:", result)  # Print the result to see its structure
        try:
            caption = result[0]['generated_text']  # Accessing the correct key
            return caption
        except (KeyError, IndexError):
            return "Failed to generate caption"
    else:
        return "Failed to capture image"

# Function to speak out text
def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('voice', engine.getProperty('voices')[0].id)
    engine.say(text)
    engine.runAndWait()

# Function to capture and display image
def capture_and_display_image():
    cap = cv2.VideoCapture(0)  # 0 represents the default webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        cv2.imshow("Camera Feed", frame)
        
        # Check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exit key pressed, closing...")
            break
        
        yield frame
        
        if cv2.getWindowProperty("Camera Feed", cv2.WND_PROP_VISIBLE) < 1:
            break  # Check if the window is closed
    
    cap.release()
    cv2.destroyAllWindows()


# Main loop to start and stop the program
while True:
    if listen_for_on_off():  # If user says "ON"
        
        # Main application loop
        speak("How are you? Hope you're doing well.")
        performing_captioning = False
        while True:
            speak("How can I help you?")
            speak("1. Object Detection")
            speak("2. Text Extraction")
            speak("3. Image Captioning")
            speak("4. Exit")
            # Recognize the choice with enhanced voice recognition
            choice = perform_voice_recognition(['one', 'two', 'three', 'four', 'do', 'who', 'oh', 'when', 'hurry','har'])
            if choice == 'one':
                speak("Opening camera for object detection. ")
                performing_captioning = False
                for frame in capture_and_display_image():
                    detected_object = detect_objects(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                    if detected_object:
                        speak(f"{detected_object['label']} detected")
                    if keyboard.is_pressed('q'):
                        break
            elif choice == 'two':
                speak("Opening camera for text extraction. ")
                performing_captioning = False
                for frame in capture_and_display_image():
                    extracted_text = perform_ocr(frame)
                    if extracted_text:
                        speak(" ".join(extracted_text))
                    if keyboard.is_pressed('q'):
                        break
            elif choice == 'three':
                speak("Performing image captioning.")
                performing_captioning = True
                while performing_captioning:
                    caption = perform_image_captioning()
                    speak(caption)
                    if keyboard.is_pressed('q'):
                        performing_captioning = False
            elif choice == 'four':
                speak("Exiting. Goodbye!")
                break
            else:
                speak("Invalid choice. Please try again.")
        
    else:  # If user says "OFF"
        speak("Goodbye!")
        play_tune(reverse=True)  # Play the tune in reverse
        break