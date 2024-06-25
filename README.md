
# Blind Assistant Glass Project 

## Aim:
The primary aim of this project is to develop a pair of smart glasses equipped with advanced image processing and voice interaction capabilities to assist visually impaired individuals. These glasses are designed to provide real-time information about the user's surroundings, including object detection, text reading, and scene description, thereby enhancing their ability to navigate and interact with their environment independently.

## Key Objectives:
### Enhance Mobility: 
Aid visually impaired users in recognizing objects and navigating through their surroundings safely and efficiently.
### Increase Independence: 
Enable users to perform everyday tasks without the need for constant human assistance by providing real-time information and feedback.
### Improve Interaction: 
Facilitate interaction with the environment by reading text from signs, documents, and other written materials aloud.
### Provide Scene Understanding: 
Offer descriptive summaries of the user's environment, helping them understand the context and details of their surroundings.

## Core Functionalities:
### Voice Command Recognition: 
The glasses can understand and respond to voice commands, allowing users to interact with the system hands-free.
### Object Detection: 
Using advanced machine learning models, the glasses can identify and name objects within the user's field of vision, alerting them to obstacles and important items.
### Optical Character Recognition (OCR): 
The system can read text from various surfaces and documents, converting it into speech so that users can understand written information.
### Image Captioning: 
The glasses provide descriptive captions of the scene in front of the user, helping them to comprehend complex environments and situations.
### Audio Feedback: 
Real-time auditory feedback ensures that users receive immediate and clear information about their surroundings and interactions.

## Detailed Implementation:
### Vosk: 
Converts spoken commands into text using offline speech recognition.
### FuzzyWuzzy: 
Enhances command recognition accuracy through fuzzy matching.
### OpenCV: 
Captures images from a camera embedded in the glasses and processes them for further analysis.
### PIL (Python Imaging Library): 
Converts images into formats compatible with various processing libraries.
### DetrImageProcessor & DetrForObjectDetection: 
Implements the DETR model to detect and label objects in real-time, providing crucial information about the user's surroundings.
### EasyOCR: 
Reads and extracts text from images, converting it into speech for the user to hear.
### Transformers: 
Uses the BLIP model to generate natural language descriptions of the visual scenes, aiding users in understanding complex environments.
### Pyttsx3: 
Converts text information into spoken words, providing auditory feedback to the user.
### Winsound: 
Plays audio signals to indicate various states and actions, enhancing user interaction.
### Keyboard: 
Detects keypresses for additional control options, although primarily intended for use with voice commands.
![Gemini_Generated_Image_panlwbpanlwbpanl](https://github.com/Sarim043/AI-Projects/assets/173757190/28bb2e1f-576b-4577-92f2-b30371d2ce69)
