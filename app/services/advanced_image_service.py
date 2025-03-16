import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from app.services.text_service import generate_context, enhance_context, analyze_sentiment
from app.services.image_service import image_processor
import logging
from config.gemini_config import get_gemini_client, GEMINI_CONFIG
import json
import io
import base64

logger = logging.getLogger(__name__)

class AdvancedImageProcessor:
    def __init__(self):
        self.image = None
        self.image_array = None
        self.color_clusters = 5  # Number of dominant colors to detect
        self.gemini = get_gemini_client()

    def load_image(self, image_path):
        """Load and prepare image for processing"""
        try:
            self.image = Image.open(image_path)
            # Convert image to RGB mode if it isn't already
            if self.image.mode != 'RGB':
                self.image = self.image.convert('RGB')
            self.image_array = np.array(self.image)
            return self.image, self.image_array
        except Exception as e:
            raise ValueError(f"Error loading image: {str(e)}")

    def generate_image_context(self):
        """Generate detailed image description using Gemini Vision"""
        try:
            if self.image is None:
                raise ValueError("No image loaded")
            
            model = self.gemini.GenerativeModel(GEMINI_CONFIG['vision_model'])
            
            prompt = """Analyze this image in detail and provide:
            1. A comprehensive description
            2. Key elements and their significance
            3. Notable visual characteristics
            4. Any relevant context or implications
            Be specific and detailed in your analysis."""
            
            response = model.generate_content([prompt, self.image])
            return response.text.strip()
        except Exception as e:
            raise ValueError(f"Error generating image context: {str(e)}")

    def generate_enhanced_text(self, base_description):
        """Generate enhanced description using Gemini"""
        try:
            model = self.gemini.GenerativeModel(GEMINI_CONFIG['text_model'])
            
            prompt = f"""Based on this image description, provide an enhanced, more detailed analysis:
            
            Original description: {base_description}
            
            Please include:
            1. Deeper contextual analysis
            2. Potential symbolism or significance
            3. Technical aspects of the image
            4. Cultural or historical relevance (if applicable)"""
            
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            raise ValueError(f"Error generating enhanced text: {str(e)}")

    def analyze_colors(self):
        """Analyze color distribution and dominant colors"""
        try:
            if self.image_array is None:
                raise ValueError("No image loaded")

            # Reshape the image array for color analysis
            pixels = self.image_array.reshape(-1, 3)
            
            # Create color histogram with improved visualization
            plt.figure(figsize=(8, 4))
            hist_data = np.mean(pixels, axis=0)
            plt.plot(range(3), hist_data, marker='o', color='#23cca2')  # Primary green
            plt.xticks(range(3), ['R', 'G', 'B'])
            plt.title('Color Distribution', color='#2c3e50')  # Dark text color
            plt.grid(True, alpha=0.3)
            hist_fig = plt.gcf()
            plt.close()

            # Find dominant colors using K-means
            kmeans = KMeans(n_clusters=self.color_clusters, random_state=42)
            kmeans.fit(pixels)
            colors = kmeans.cluster_centers_
            
            # Calculate color percentages
            labels = kmeans.labels_
            unique_labels, label_counts = np.unique(labels, return_counts=True)
            percentages = (label_counts / len(labels)) * 100
            
            # Sort colors by percentage
            sorted_indices = np.argsort(percentages)[::-1]
            colors = colors[sorted_indices]
            percentages = percentages[sorted_indices]
            
            # Create pie chart of dominant colors
            plt.figure(figsize=(6, 6))
            
            # Convert colors to RGB format for plotting
            rgb_colors = colors / 255.0
            
            # Create pie chart with percentage labels
            patches, texts, autotexts = plt.pie(percentages, 
                                              colors=rgb_colors, 
                                              autopct='%1.1f%%',
                                              labels=[f'Color {i+1}' for i in range(len(colors))])
            
            plt.title('Dominant Colors')
            
            # Format percentage texts
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(8)
            
            pie_fig = plt.gcf()
            plt.close()

            # Convert color data for JSON response
            color_data = {
                'bins': list(range(3)),  # R, G, B channels
                'distribution': hist_data.tolist(),  # Color distribution data
                'dominant_colors': colors.astype(int).tolist(),  # RGB values of dominant colors
                'percentages': percentages.tolist()  # Percentage of each dominant color
            }

            return hist_fig, pie_fig, color_data
        except Exception as e:
            logger.error(f"Color analysis error details: {str(e)}")
            raise ValueError(f"Error analyzing colors: {str(e)}")

    def sentiment_analysis(self, text):
        """Analyze sentiment using Gemini"""
        try:
            model = self.gemini.GenerativeModel(GEMINI_CONFIG['text_model'])
            
            prompt = f"""Analyze the sentiment of this text and provide:
            1. Overall sentiment category (Positive, Negative, or Neutral)
            2. Confidence score (0-1)
            3. Key emotional indicators
            
            Text: {text}
            
            You MUST return ONLY valid JSON with these exact keys:
            {{
                "category": "sentiment category",
                "score": confidence_score,
                "indicators": ["key", "emotional", "indicators"]
            }}
            
            Do not include any explanations, markdown formatting, or additional text before or after the JSON."""
            
            response = model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Log the response for debugging
            logger.debug(f"Gemini sentiment response: {response_text}")
            
            # Handle potential JSON parsing issues
            try:
                # Try to find JSON in the response if it's not properly formatted
                if not response_text.startswith('{'):
                    # Look for the first occurrence of '{'
                    json_start = response_text.find('{')
                    if json_start >= 0:
                        # Find the matching closing brace
                        json_end = response_text.rfind('}') + 1
                        if json_end > json_start:
                            response_text = response_text[json_start:json_end]
                
                sentiment_data = json.loads(response_text)
                
                # Validate the required keys exist
                required_keys = ['category', 'score', 'indicators']
                for key in required_keys:
                    if key not in sentiment_data:
                        raise ValueError(f"Missing required key '{key}' in sentiment response")
                
                # Ensure indicators is a list
                if not isinstance(sentiment_data['indicators'], list):
                    sentiment_data['indicators'] = [sentiment_data['indicators']]
                
                return pd.DataFrame([{
                    'Sentiment': sentiment_data['category'],
                    'Confidence': sentiment_data['score'],
                    'Indicators': ', '.join(sentiment_data['indicators'])
                }])
            except json.JSONDecodeError as json_err:
                # Fallback to a default sentiment analysis if JSON parsing fails
                logger.error(f"JSON parsing error: {str(json_err)}. Response: {response_text}")
                return pd.DataFrame([{
                    'Sentiment': 'Neutral',
                    'Confidence': 0.5,
                    'Indicators': 'Error parsing sentiment'
                }])
        except Exception as e:
            logger.error(f"Sentiment analysis error: {str(e)}")
            # Return a default sentiment rather than raising an error
            return pd.DataFrame([{
                'Sentiment': 'Neutral',
                'Confidence': 0.5,
                'Indicators': f'Error: {str(e)}'
            }]) 