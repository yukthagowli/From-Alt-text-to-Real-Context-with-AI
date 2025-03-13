# **From Alt Text to Real Context with AI**

This project is a Flask-based web application that generates captions for images using the **BLIP model** and processes the captions using the **Gemini API**. The generated captions are stored as embeddings in **Pinecone**, a vector database, for future retrieval or analysis.

---

## **Features**
- **Image Captioning**: Generate captions for uploaded images using the BLIP model.
- **Contextual Processing**: Use the Gemini API to process and enhance the generated captions.
- **Vector Storage**: Store caption embeddings in Pinecone for efficient retrieval.
- **Modern UI**: A sleek and interactive user interface with black and violet colors, 3D shapes, and smooth animations.

---

## **Technologies Used**
- **Python**: Backend logic and API development.
- **Flask**: Web framework for building the application.
- **BLIP Model**: Pre-trained model for image captioning.
- **Gemini API**: For processing and enhancing captions.
- **Pinecone**: Vector database for storing caption embeddings.
- **HTML/CSS/JavaScript**: Frontend design and interactivity.

---

## **Setup Instructions**

### **1. Prerequisites**
- Python 3.7 or higher.
- A **Gemini API key** from [Google Cloud Console](https://console.cloud.google.com/).
- A **Pinecone API key** from [Pinecone](https://www.pinecone.io/).

### **2. Clone the Repository**
```bash
git clone https://github.com/your-username/from-alt-text-to-real-context-with-ai.git
cd from-alt-text-to-real-context-with-ai
