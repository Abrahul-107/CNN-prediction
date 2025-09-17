#!/usr/bin/env python3
"""
PDF First Page Classification - Production Inference Script

This script provides production-ready inference for PDF page classification.
It can handle different model types and provides comprehensive prediction results.

Key Features:
- Support for multiple model architectures (transfer, hybrid, simple)
- Robust model loading with fallback strategies
- Batch processing capabilities
- Comprehensive prediction reporting
- Excel/CSV output formats
- Error handling and recovery

Technical Approach:
1. Load trained model with proper custom objects
2. Extract and preprocess PDF pages
3. Generate predictions (with OCR for hybrid models)
4. Optimize classification thresholds
5. Export results with confidence scores
"""

import os
import pandas as pd
import numpy as np
from pdf2image import convert_from_path
import cv2
import pytesseract
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input, Model
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FirstPagePredictor:
    def __init__(self, model_path, image_size=(224, 224), model_type='transfer', confidence_threshold=0.5):
        """
        Initialize PDF first page predictor.
        
        Args:
            model_path (str): Path to trained model file (.h5)
            image_size (tuple): Input image dimensions (must match training)
            model_type (str): Model architecture type ('transfer', 'hybrid', 'simple')
            confidence_threshold (float): Classification threshold (0.0-1.0)
        
        Model Type Selection:
        - 'transfer': Transfer learning model (MobileNetV2/ResNet50 based)
        - 'hybrid': CNN + OCR text features combination
        - 'simple': Basic CNN trained from scratch
        
        Technical Notes:
        - Image size must match the size used during training
        - Model type determines input format and preprocessing requirements
        - Confidence threshold can be optimized based on validation results
        """
        self.image_size = image_size
        self.model_path = model_path
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.model = None
        
        print(f"Initializing PDF First Page Predictor:")
        print(f"- Model path: {model_path}")
        print(f"- Image size: {image_size}")
        print(f"- Model type: {model_type}")
        print(f"- Confidence threshold: {confidence_threshold}")
        
        self.load_model()

    def focal_loss(self, alpha=0.75, gamma=2.0):
        """
        Recreate focal loss function for model loading compatibility.
        
        Args:
            alpha (float): Class weighting parameter
            gamma (float): Focusing parameter
        
        Returns:
            function: Focal loss function for Keras
        
        Purpose:
        When models are trained with custom loss functions (like focal loss),
        these functions must be available during model loading. This recreates
        the focal loss used during training to ensure compatibility.
        """
        def focal_loss_fixed(y_true, y_pred):
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
            
            alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
            p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
            focal_loss = -alpha_t * tf.pow((1 - p_t), gamma) * tf.math.log(p_t)
            
            return tf.reduce_mean(focal_loss)
        
        return focal_loss_fixed

    def create_model_architecture(self):
        """
        Recreate model architecture for weights-only loading.
        
        Returns:
            keras.Model: Model architecture matching training configuration
        
        Architecture Recreation Strategy:
        When model loading fails due to custom objects or version incompatibility,
        we recreate the exact architecture used during training and load only
        the learned weights. This provides better compatibility across environments.
        """
        if self.model_type == 'transfer':
            return self.create_transfer_model()
        elif self.model_type == 'hybrid':
            return self.create_hybrid_model()
        elif self.model_type == 'simple':
            return self.create_simple_model()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def create_transfer_model(self):
        """Create transfer learning model architecture."""
        try:
            from tensorflow.keras.applications import MobileNetV2
            base_model = MobileNetV2(weights='imagenet', include_top=False, 
                                   input_shape=(*self.image_size, 3))
        except:
            from tensorflow.keras.applications import ResNet50
            base_model = ResNet50(weights='imagenet', include_top=False,
                                input_shape=(*self.image_size, 3))
        
        base_model.trainable = True
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.6),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
        return model

    def create_hybrid_model(self):
        """Create hybrid CNN + OCR model architecture."""
        # Image input branch
        image_input = Input(shape=(*self.image_size, 3), name='image_input')
        x = layers.Conv2D(32, (5, 5), activation='relu')(image_input)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        
        # Text features input branch  
        text_input = Input(shape=(2,), name='text_input')  # 2 OCR features
        t = layers.Dense(32, activation='relu')(text_input)
        t = layers.Dropout(0.2)(t)
        
        # Combine branches
        combined = layers.concatenate([x, t])
        combined = layers.Dense(64, activation='relu')(combined)
        combined = layers.Dropout(0.3)(combined)
        combined = layers.Dense(32, activation='relu')(combined)
        combined = layers.Dropout(0.2)(combined)
        output = layers.Dense(1, activation='sigmoid')(combined)
        
        model = Model(inputs=[image_input, text_input], outputs=output)
        return model

    def create_simple_model(self):
        """Create simple CNN architecture."""
        model = keras.Sequential([
            layers.Conv2D(32, (5, 5), activation='relu', input_shape=(*self.image_size, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
        return model

    def load_model(self):
        """
        Load trained model with multiple fallback strategies.
        
        Loading Strategies:
        1. Direct loading with custom objects (for focal loss compatibility)
        2. Architecture recreation + weights loading
        3. Loading without compilation (for inference only)
        
        Error Handling:
        Each strategy is tried sequentially. If all fail, detailed error
        information is provided to help with troubleshooting.
        """
        print("Loading trained model...")
        
        # Strategy 1: Load with custom objects (handles focal loss)
        try:
            custom_objects = {
                'focal_loss_fixed': self.focal_loss(),
            }
            self.model = keras.models.load_model(self.model_path, custom_objects=custom_objects)
            print("✓ Model loaded successfully (with custom objects)")
            return
        except Exception as e1:
            print(f"Strategy 1 failed: {str(e1)[:100]}...")
        
        # Strategy 2: Recreate architecture and load weights
        try:
            print("Attempting to recreate architecture and load weights...")
            self.model = self.create_model_architecture()
            self.model.load_weights(self.model_path)
            print("✓ Model weights loaded successfully")
            return
        except Exception as e2:
            print(f"Strategy 2 failed: {str(e2)[:100]}...")
        
        # Strategy 3: Load without compilation (inference only)
        try:
            print("Attempting to load without compilation...")
            self.model = keras.models.load_model(self.model_path, compile=False)
            print("✓ Model loaded without compilation")
            return
        except Exception as e3:
            print(f"Strategy 3 failed: {str(e3)[:100]}...")
        
        # All strategies failed
        raise RuntimeError(
            f"Failed to load model from {self.model_path}. "
            f"Check:\n"
            f"1. File exists and is accessible\n"
            f"2. Model type matches training configuration\n"
            f"3. TensorFlow version compatibility\n"
            f"4. Required dependencies are installed"
        )

    def clean_annotated_image(self, image):
        """
        Remove colored annotations from document images.
        
        Args:
            image (numpy.ndarray): RGB image array
        
        Returns:
            numpy.ndarray: Cleaned image with annotations removed
        
        Annotation Removal Process:
        1. Convert RGB to HSV for better color detection
        2. Create masks for yellow highlights, green circles, blue numbers
        3. Combine masks with morphological operations
        4. Replace detected regions with white background
        
        This preprocessing step is crucial for inference as it matches
        the annotation cleaning performed during training.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Define color ranges for different annotation types
        color_ranges = [
            ([20, 100, 100], [90, 255, 255]),   # Yellow highlights
            ([40, 50, 50], [80, 255, 255]),     # Green circles
            ([100, 50, 50], [130, 255, 255])    # Blue numbers
        ]
        
        # Create combined annotation mask
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in color_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Apply morphological dilation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
        
        # Replace annotations with white background
        cleaned_image = image.copy()
        cleaned_image[combined_mask > 0] = [255, 255, 255]
        
        return cleaned_image

    def extract_pages_from_pdf(self, pdf_path, output_dir="inference_pages"):
        """
        Extract PDF pages as individual image files.
        
        Args:
            pdf_path (str): Path to input PDF file
            output_dir (str): Directory for extracted images
        
        Returns:
            list: Paths to extracted image files
        
        Extraction Configuration:
        - 200 DPI: Balances quality with processing speed
        - JPEG format: Efficient storage with good quality
        - 90% quality: Minimal compression artifacts
        
        Error Handling:
        Provides clear error messages and installation instructions
        if PDF extraction dependencies are missing.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"Extracting pages from PDF: {pdf_path}")
        try:
            pages = convert_from_path(pdf_path, dpi=200)
        except Exception as e:
            print(f"PDF extraction failed: {e}")
            print("Required dependency: poppler-utils")
            print("Install with:")
            print("  macOS: brew install poppler")
            print("  Ubuntu: sudo apt-get install poppler-utils")
            print("  Windows: Download from poppler website")
            raise

        page_paths = []
        for i, page in enumerate(pages, 1):
            page_path = os.path.join(output_dir, f"page_{i:03d}.jpg")
            page.save(page_path, 'JPEG', quality=90)
            page_paths.append(page_path)

        print(f"Extracted {len(pages)} pages to {output_dir}")
        return page_paths

    def preprocess_image(self, image_path):
        """
        Preprocess image for model input (matches training preprocessing).
        
        Args:
            image_path (str): Path to image file
        
        Returns:
            numpy.ndarray: Preprocessed image ready for prediction
        
        Preprocessing Pipeline:
        1. Load image (OpenCV loads as BGR)
        2. Convert BGR to RGB (model expects RGB)
        3. Clean annotations to match training preprocessing
        4. Resize to target dimensions
        5. Normalize pixel values to [0,1] range
        
        Critical: This preprocessing must exactly match the preprocessing
        used during training to ensure consistent model performance.
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image from {image_path}")

            # Convert BGR (OpenCV) to RGB (model expects)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Clean annotations (matches training preprocessing)
            image = self.clean_annotated_image(image)
            
            # Resize to model input dimensions
            image = cv2.resize(image, self.image_size)
            
            # Normalize to [0,1] range
            image = image.astype('float32') / 255.0
            
            return image
            
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
            raise

    def extract_text_features(self, image):
        """
        Extract text features using OCR (for hybrid models only).
        
        Args:
            image (numpy.ndarray): Preprocessed image array
        
        Returns:
            numpy.ndarray: Text feature vector
        
        OCR Feature Extraction:
        1. Convert normalized image back to uint8 for OCR
        2. Run Tesseract OCR with German+English models
        3. Extract layout-based features (no keyword matching)
        4. Normalize features to [0,1] range
        
        Features Extracted:
        - First line length: Indicates header/title presence
        - Text density: Overall amount of text on page
        
        Error Handling:
        Returns zero-filled feature vector if OCR fails, allowing
        graceful degradation without stopping prediction process.
        """
        try:
            # Convert back to uint8 for OCR processing
            img_uint8 = (image * 255).astype('uint8')
            
            # Extract text using Tesseract
            text = pytesseract.image_to_string(img_uint8, lang='deu+eng')
            
            # Ensure text is string type
            if not isinstance(text, str):
                text = str(text)
            
            features = []
            
            # Feature 1: First line length (normalized)
            lines = text.split('\n')
            if lines and isinstance(lines[0], str):
                first_line_length = len(lines[0].strip())
            else:
                first_line_length = 0
            features.append(min(first_line_length / 50.0, 1.0))
            
            # Feature 2: Text density (normalized)
            total_text_length = len(text.replace('\n', '').replace(' ', ''))
            text_density = min(total_text_length / 1000.0, 1.0)
            features.append(text_density)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            print(f"OCR extraction failed: {e}")
            print("Required dependency: tesseract-ocr")
            print("Install with:")
            print("  macOS: brew install tesseract tesseract-lang-deu")
            print("  Ubuntu: sudo apt-get install tesseract-ocr tesseract-ocr-deu")
            
            # Return zero features on OCR failure
            return np.zeros(2, dtype=np.float32)

    def predict_pdf(self, pdf_path, output_excel_path="predictions.xlsx", 
                   use_optimal_threshold=False):
        """
        Generate predictions for all pages in a PDF document.
        
        Args:
            pdf_path (str): Path to input PDF file
            output_excel_path (str): Path for output Excel file
            use_optimal_threshold (bool): Use confidence threshold or 0.5
        
        Returns:
            pandas.DataFrame: Prediction results with confidence scores
        
        Prediction Process:
        1. Extract all PDF pages as images
        2. Preprocess each page (resize, normalize, clean annotations)
        3. Extract text features (for hybrid models)
        4. Generate probability predictions using trained model
        5. Apply classification threshold to get binary predictions
        6. Compile results with confidence scores and metadata
        7. Export to Excel/CSV format with detailed statistics
        
        Output Format:
        - pagenumber: Page number in original PDF
        - possibility_of_first_page: Binary prediction (0 or 1)
        - confidence_score: Model probability output (0.0-1.0)
        """
        print(f"Processing PDF: {pdf_path}")
        print(f"Model type: {self.model_type}")
        print(f"Confidence threshold: {self.confidence_threshold}")

        # Extract all pages from PDF
        page_paths = self.extract_pages_from_pdf(pdf_path)
        total_pages = len(page_paths)
        
        # Initialize results container
        predictions = []
        
        print(f"Generating predictions for {total_pages} pages...")
        
        # Process each page
        for i, page_path in enumerate(page_paths, 1):
            try:
                # Preprocess image
                image = self.preprocess_image(page_path)
                
                # Generate prediction based on model type
                if self.model_type == 'hybrid':
                    # Hybrid model: requires both image and text features
                    text_features = self.extract_text_features(image)
                    
                    # Prepare batch inputs
                    image_batch = np.expand_dims(image, axis=0)
                    text_batch = np.expand_dims(text_features, axis=0)
                    
                    # Get probability prediction
                    prediction_prob = self.model.predict([image_batch, text_batch], verbose=0)[0][0]
                    
                else:
                    # Transfer/simple models: image input only
                    image_batch = np.expand_dims(image, axis=0)
                    prediction_prob = self.model.predict(image_batch, verbose=0)[0][0]

                # Apply classification threshold
                threshold = self.confidence_threshold if use_optimal_threshold else 0.5
                prediction_binary = 1 if prediction_prob > threshold else 0

                # Store result
                predictions.append({
                    'pagenumber': i,
                    'possibility_of_first_page': prediction_binary,
                    'confidence_score': float(prediction_prob)
                })

                # Progress indicator
                if i % 20 == 0 or i == total_pages:
                    print(f"Processed {i}/{total_pages} pages ({i/total_pages*100:.1f}%)")

            except Exception as e:
                print(f"Error processing page {i}: {e}")
                # Add default prediction for failed pages
                predictions.append({
                    'pagenumber': i,
                    'possibility_of_first_page': 0,  # Conservative default
                    'confidence_score': 0.0
                })

        # Create results DataFrame
        results_df = pd.DataFrame(predictions)
        
        # Add metadata columns
        results_df['pdf_filename'] = os.path.basename(pdf_path)
        results_df['model_type'] = self.model_type
        results_df['threshold_used'] = threshold
        results_df['prediction_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Save results to Excel
        self.save_predictions(results_df, output_excel_path)
        
        # Print comprehensive summary
        self.print_prediction_summary(results_df, pdf_path)

        return results_df

    def save_predictions(self, results_df, output_path):
        """
        Save prediction results to Excel or CSV format.
        
        Args:
            results_df (pandas.DataFrame): Prediction results
            output_path (str): Output file path
        
        File Format Handling:
        - Attempts Excel format first (.xlsx)
        - Falls back to CSV if Excel writing fails
        - Includes comprehensive metadata in both formats
        """
        try:
            # Try Excel format first
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Main results
                results_df.to_excel(writer, sheet_name='Predictions', index=False)
                
                # Summary statistics
                summary_data = {
                    'Metric': ['Total Pages', 'Predicted First Pages', 'Predicted Non-First Pages', 
                              'Average Confidence', 'Min Confidence', 'Max Confidence'],
                    'Value': [
                        len(results_df),
                        results_df['possibility_of_first_page'].sum(),
                        len(results_df) - results_df['possibility_of_first_page'].sum(),
                        results_df['confidence_score'].mean(),
                        results_df['confidence_score'].min(),
                        results_df['confidence_score'].max()
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            print(f"Predictions saved to {output_path}")
            
        except Exception as e:
            print(f"Excel save failed: {e}")
            # Fallback to CSV
            csv_path = output_path.replace('.xlsx', '.csv')
            results_df.to_csv(csv_path, index=False)
            print(f"Saved as CSV: {csv_path}")

    def print_prediction_summary(self, results_df, pdf_path):
        """
        Print comprehensive prediction summary to console.
        
        Args:
            results_df (pandas.DataFrame): Prediction results
            pdf_path (str): Path to processed PDF
        
        Summary Statistics:
        - Overall prediction counts and percentages
        - Confidence score distribution
        - Pages identified as first pages with confidence levels
        - Performance indicators and recommendations
        """
        total_pages = len(results_df)
        first_pages = results_df['possibility_of_first_page'].sum()
        non_first_pages = total_pages - first_pages
        avg_confidence = results_df['confidence_score'].mean()
        
        print(f"\n{'='*60}")
        print("PREDICTION SUMMARY")
        print(f"{'='*60}")
        print(f"PDF File: {os.path.basename(pdf_path)}")
        print(f"Model Type: {self.model_type}")
        print(f"Classification Threshold: {self.confidence_threshold}")
        print(f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\nPage Analysis:")
        print(f"- Total pages processed: {total_pages}")
        print(f"- Pages predicted as FIRST pages: {first_pages} ({first_pages/total_pages*100:.1f}%)")
        print(f"- Pages predicted as NON-FIRST pages: {non_first_pages} ({non_first_pages/total_pages*100:.1f}%)")
        
        print(f"\nConfidence Statistics:")
        print(f"- Average confidence: {avg_confidence:.4f}")
        print(f"- Minimum confidence: {results_df['confidence_score'].min():.4f}")
        print(f"- Maximum confidence: {results_df['confidence_score'].max():.4f}")
        print(f"- Standard deviation: {results_df['confidence_score'].std():.4f}")

        # Show first page predictions with confidence
        first_page_results = results_df[results_df['possibility_of_first_page'] == 1]
        if len(first_page_results) > 0:
            print(f"\nPages Identified as FIRST PAGES:")
            print(f"{'Page':<6} {'Confidence':<12} {'Assessment':<15}")
            print("-" * 35)
            
            for _, row in first_page_results.iterrows():
                page_num = int(row['pagenumber'])
                confidence = row['confidence_score']
                
                # Confidence assessment
                if confidence > 0.8:
                    assessment = "High Confidence"
                elif confidence > 0.6:
                    assessment = "Medium Confidence"
                else:
                    assessment = "Low Confidence"
                
                print(f"{page_num:<6} {confidence:<12.4f} {assessment:<15}")
        else:
            print(f"\n⚠️  No pages were identified as first pages")
            print("Consider adjusting the confidence threshold or checking model performance")

        # Recommendations
        print(f"\nRecommendations:")
        if avg_confidence < 0.6:
            print("⚠️  Low average confidence - consider retraining model or checking data quality")
        elif avg_confidence > 0.8:
            print("✅ High average confidence - predictions are reliable")
        else:
            print("✅ Moderate confidence - predictions are reasonably reliable")
        
        if first_pages == 0:
            print("⚠️  No first pages detected - check threshold or model performance")
        elif first_pages > total_pages * 0.5:
            print("⚠️  High proportion of first pages detected - verify results")

    def batch_predict(self, pdf_list, output_dir="batch_predictions", 
                     use_optimal_threshold=False):
        """
        Process multiple PDF files in batch mode.
        
        Args:
            pdf_list (list): List of PDF file paths
            output_dir (str): Directory for batch prediction results
            use_optimal_threshold (bool): Use optimized threshold
        
        Returns:
            pandas.DataFrame: Batch processing summary
        
        Batch Processing Benefits:
        - Efficiently process multiple documents
        - Consistent preprocessing and prediction parameters
        - Consolidated reporting across all documents
        - Error isolation (one failed PDF doesn't stop others)
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"\n{'='*60}")
        print(f"BATCH PROCESSING {len(pdf_list)} PDF FILES")
        print(f"{'='*60}")

        batch_summary = []
        successful_predictions = 0
        
        for i, pdf_path in enumerate(pdf_list, 1):
            try:
                pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
                output_excel = os.path.join(output_dir, f"{pdf_name}_predictions.xlsx")

                print(f"\n[{i}/{len(pdf_list)}] Processing: {pdf_name}")
                print("-" * 50)

                # Generate predictions for this PDF
                results_df = self.predict_pdf(
                    pdf_path, 
                    output_excel, 
                    use_optimal_threshold=use_optimal_threshold
                )

                # Add to batch summary
                batch_summary.append({
                    'pdf_filename': pdf_name,
                    'pdf_path': pdf_path,
                    'total_pages': len(results_df),
                    'first_pages_found': results_df['possibility_of_first_page'].sum(),
                    'avg_confidence': results_df['confidence_score'].mean(),
                    'min_confidence': results_df['confidence_score'].min(),
                    'max_confidence': results_df['confidence_score'].max(),
                    'output_file': output_excel,
                    'processing_status': 'Success'
                })
                
                successful_predictions += 1
                print(f"✅ Successfully processed {pdf_name}")

            except Exception as e:
                print(f"❌ Error processing {pdf_path}: {e}")
                batch_summary.append({
                    'pdf_filename': os.path.basename(pdf_path),
                    'pdf_path': pdf_path,
                    'total_pages': 0,
                    'first_pages_found': 0,
                    'avg_confidence': 0.0,
                    'min_confidence': 0.0,
                    'max_confidence': 0.0,
                    'output_file': 'Failed',
                    'processing_status': f'Error: {str(e)[:50]}...'
                })

        # Create batch summary
        summary_df = pd.DataFrame(batch_summary)
        
        # Save batch summary
        summary_path = os.path.join(output_dir, "batch_processing_summary.xlsx")
        try:
            summary_df.to_excel(summary_path, index=False)
            print(f"\n📊 Batch summary saved to {summary_path}")
        except:
            csv_summary_path = os.path.join(output_dir, "batch_processing_summary.csv")
            summary_df.to_csv(csv_summary_path, index=False)
            print(f"\n📊 Batch summary saved to {csv_summary_path}")

        # Print batch results
        print(f"\n{'='*60}")
        print("BATCH PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total PDFs processed: {len(pdf_list)}")
        print(f"Successful predictions: {successful_predictions}")
        print(f"Failed predictions: {len(pdf_list) - successful_predictions}")
        print(f"Success rate: {successful_predictions/len(pdf_list)*100:.1f}%")
        
        if successful_predictions > 0:
            successful_results = summary_df[summary_df['processing_status'] == 'Success']
            print(f"\nAggregate Statistics:")
            print(f"- Total pages processed: {successful_results['total_pages'].sum()}")
            print(f"- Total first pages found: {successful_results['first_pages_found'].sum()}")
            print(f"- Average confidence: {successful_results['avg_confidence'].mean():.4f}")

        return summary_df


def main():
    """
    Main function demonstrating inference script usage.
    
    This function shows how to use the FirstPagePredictor class for:
    1. Single PDF prediction
    2. Batch PDF processing
    3. Different model types and configurations
    4. Error handling and troubleshooting
    """
    print("PDF First Page Classification - Inference Script")
    print("=" * 60)
    
    # Configuration
    MODEL_PATH = "best_model_fold_1.h5"  # Update with your model path
    MODEL_TYPE = "hybrid"                       # Options: 'transfer', 'hybrid', 'simple'
    IMAGE_SIZE = (224, 224)                       # Must match training configuration
    CONFIDENCE_THRESHOLD = 0.5                    # Adjust based on validation results
    
    # Input files
    PDF_PATH = "Sample Invoices.pdf"              # Single PDF for testing
    OUTPUT_PATH = "first_page_predictions.xlsx"   # Output file
    
    # Batch processing (optional)
    PDF_LIST = [
        "Sample Invoices.pdf",
        # Add more PDF files here for batch processing
    ]
    
    print(f"Configuration:")
    print(f"- Model path: {MODEL_PATH}")
    print(f"- Model type: {MODEL_TYPE}")
    print(f"- Image size: {IMAGE_SIZE}")
    print(f"- Confidence threshold: {CONFIDENCE_THRESHOLD}")
    
    if os.path.exists(MODEL_PATH):
        try:
            # Initialize predictor
            predictor = FirstPagePredictor(
                model_path=MODEL_PATH,
                model_type=MODEL_TYPE,
                image_size=IMAGE_SIZE,
                confidence_threshold=CONFIDENCE_THRESHOLD
            )
            
            # Single PDF prediction
            print(f"\n{'='*40}")
            print("SINGLE PDF PREDICTION")
            print(f"{'='*40}")
            
            if os.path.exists(PDF_PATH):
                results = predictor.predict_pdf(
                    pdf_path=PDF_PATH,
                    output_excel_path=OUTPUT_PATH,
                    use_optimal_threshold=True
                )
                print(f"✅ Single PDF prediction completed!")
                print(f"Results saved to: {OUTPUT_PATH}")
            else:
                print(f"⚠️  PDF file not found: {PDF_PATH}")
            
            # Batch processing (if multiple PDFs available)
            existing_pdfs = [pdf for pdf in PDF_LIST if os.path.exists(pdf)]
            if len(existing_pdfs) > 1:
                print(f"\n{'='*40}")
                print("BATCH PROCESSING")
                print(f"{'='*40}")
                
                batch_summary = predictor.batch_predict(
                    pdf_list=existing_pdfs,
                    output_dir="batch_predictions",
                    use_optimal_threshold=True
                )
                print(f"✅ Batch processing completed!")
                print(f"Results saved to: batch_predictions/")
            
            print(f"\n🎉 Inference completed successfully!")
            
        except Exception as e:
            print(f"❌ Error during inference: {e}")
            print(f"\n🔧 Troubleshooting Guide:")
            print(f"1. Verify model file exists: {MODEL_PATH}")
            print(f"2. Check MODEL_TYPE matches training: {MODEL_TYPE}")
            print(f"3. Ensure dependencies are installed:")
            print(f"   - For all models: tensorflow, opencv-python, pdf2image")
            print(f"   - For hybrid models: pytesseract, tesseract-ocr")
            print(f"4. Verify PDF files are accessible and not corrupted")
            print(f"5. Check available memory and disk space")
            
    else:
        print(f"❌ Model file not found: {MODEL_PATH}")
        print(f"\nTo use this inference script:")
        print(f"1. Train a model using train_basic.py or train_kfold.py")
        print(f"2. Update MODEL_PATH to point to your trained model")
        print(f"3. Set MODEL_TYPE to match your trained model")
        print(f"4. Adjust CONFIDENCE_THRESHOLD based on validation results")
        
        print(f"\nExample usage:")
        print(f"predictor = FirstPagePredictor('my_model.h5', model_type='transfer')")
        print(f"results = predictor.predict_pdf('document.pdf', 'output.xlsx')")


if __name__ == "__main__":
    main()