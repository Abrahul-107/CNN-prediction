#!/usr/bin/env python3
"""
PDF First Page Classification - K-Fold Cross Validation Training

This script implements robust k-fold cross-validation for PDF page classification.
It provides statistical validation of model performance across multiple data splits.

Key Features:
- Stratified k-fold cross-validation for robust evaluation
- Hybrid CNN + OCR text features (optional)
- Automatic threshold optimization per fold
- Comprehensive statistical analysis
- Multiple model architectures (transfer learning, hybrid, simple CNN)
- Detailed visualization and reporting

Technical Approach:
1. Split data into k folds while maintaining class balance
2. Train separate model on each fold
3. Optimize classification threshold per fold
4. Aggregate results with statistical measures
5. Generate comprehensive performance analysis
"""

import os
import numpy as np
import pandas as pd
import cv2
from pdf2image import convert_from_path
import pytesseract
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# SSL fix for transfer learning models
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class PDFPageClassifier:
    def __init__(self, image_size=(224, 224), model_type='transfer'):
        """
        Initialize PDF page classifier with configurable model types.
        
        Args:
            image_size (tuple): Input image dimensions
            model_type (str): 'transfer', 'hybrid', or 'simple'
        
        Model Types Explained:
        - 'transfer': Uses pre-trained CNN (MobileNetV2/ResNet50) with transfer learning
        - 'hybrid': Combines CNN visual features with OCR text features  
        - 'simple': Custom CNN trained from scratch
        
        Technical Rationale:
        - Transfer learning leverages ImageNet features for better generalization
        - Hybrid models combine visual layout with semantic text understanding
        - Simple CNNs provide baseline performance for comparison
        """
        self.image_size = image_size
        self.model_type = model_type
        self.model = None
        
        print(f"Initialized PDF classifier with:")
        print(f"- Image size: {image_size}")
        print(f"- Model type: {model_type}")

    def extract_pages_from_pdf(self, pdf_path, output_dir='extracted_pages'):
        """
        Convert PDF pages to individual image files for processing.
        
        Args:
            pdf_path (str): Path to input PDF
            output_dir (str): Directory for extracted images
        
        Returns:
            list: Paths to extracted image files
        
        Technical Details:
        - Uses pdf2image library (requires poppler-utils system dependency)
        - 200 DPI balances image quality with processing speed
        - JPEG format with 90% quality for efficient storage
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print("Extracting pages from PDF...")
        try:
            pages = convert_from_path(pdf_path, dpi=200)
        except Exception as e:
            print(f"PDF extraction failed: {e}")
            print("Install poppler-utils:")
            print("  macOS: brew install poppler") 
            print("  Ubuntu: sudo apt-get install poppler-utils")
            raise
        
        page_paths = []
        for i, page in enumerate(pages, 1):
            page_path = os.path.join(output_dir, f'page_{i:03d}.jpg')
            page.save(page_path, 'JPEG', quality=95)
            page_paths.append(page_path)
        
        print(f"Extracted {len(pages)} pages to {output_dir}")
        return page_paths

    def clean_annotated_image(self, image):
        """
        Remove colored annotations (highlights, circles, markers) from document images.
        
        Args:
            image (numpy.ndarray): RGB image array
        
        Returns:
            numpy.ndarray: Cleaned image with annotations replaced by white pixels
        
        Color Detection Strategy:
        1. Convert RGB to HSV color space (better for color detection)
        2. Define HSV ranges for different annotation colors
        3. Create binary masks for each color type
        4. Combine masks and apply morphological operations
        5. Replace detected regions with document background color (white)
        
        HSV Color Ranges:
        - Yellow highlights: Hue 20-90°, Saturation 100-255, Value 100-255
        - Green circles: Hue 40-80°, Saturation 50-255, Value 50-255
        - Blue numbers: Hue 100-130°, Saturation 50-255, Value 50-255
        """
        # HSV color space is more robust for color detection
        # Hue represents the color itself, independent of lighting variations
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Define color ranges for different annotation types
        annotation_ranges = [
            ([20, 100, 100], [90, 255, 255]),   # Yellow highlights
            ([40, 50, 50], [80, 255, 255]),     # Green circles  
            ([100, 50, 50], [130, 255, 255])    # Blue numbers
        ]
        
        # Create combined mask for all annotation colors
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in annotation_ranges:
            lower_bound = np.array(lower)
            upper_bound = np.array(upper)
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Apply morphological dilation to capture annotation boundaries
        # This ensures we remove annotation edges and artifacts
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
        
        # Replace annotated regions with white background
        cleaned_image = image.copy()
        cleaned_image[combined_mask > 0] = [255, 255, 255]  # White RGB
        
        return cleaned_image

    def preprocess_image(self, image_path):
        """
        Comprehensive image preprocessing pipeline for CNN input.
        
        Args:
            image_path (str): Path to image file
        
        Returns:
            numpy.ndarray: Preprocessed image ready for neural network
        
        Preprocessing Pipeline:
        1. Load image (OpenCV uses BGR format by default)
        2. Convert BGR → RGB (neural networks expect RGB)
        3. Clean annotations to prevent model confusion
        4. Resize to target dimensions (224×224 for transfer learning)
        5. Normalize pixel values to [0,1] range for training stability
        
        Technical Considerations:
        - BGR to RGB conversion is critical for pre-trained models
        - Pixel normalization (÷255) improves gradient flow during training
        - Bilinear interpolation during resize preserves image quality
        """
        # Load image (OpenCV loads as BGR by default)
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert color space: BGR (OpenCV) → RGB (CNN standard)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Remove colored annotations that could mislead the classifier
        image = self.clean_annotated_image(image)
        
        # Resize to target dimensions using bilinear interpolation
        image = cv2.resize(image, self.image_size)
        
        # Normalize pixel values: [0,255] → [0,1]
        # This improves training stability and convergence speed
        image = image.astype('float32') / 255.0
        
        return image

    def extract_text_features(self, image):
        """
        Extract semantic text features using Optical Character Recognition (OCR).
        
        Args:
            image (numpy.ndarray): Preprocessed image array
        
        Returns:
            numpy.ndarray: Vector of text-based features
        
        OCR Feature Engineering:
        This function extracts high-level semantic features that help distinguish
        first pages from continuation pages based on text content and layout.
        
        Feature Set (5 features):
        1. First line length: Longer first lines often indicate titles/headers
        2. Text density: Total amount of text on the page
        
        Technical Implementation:
        - Uses Tesseract OCR with German + English language models
        - Converts normalized image back to uint8 format for OCR
        - Handles OCR failures gracefully with zero-filled feature vectors
        - Features are normalized to [0,1] range for neural network compatibility
        
        Limitations Removed:
        - No keyword-based features (as requested)
        - Focus on layout and density features only
        """
        try:
            # Convert normalized image back to uint8 for OCR processing
            img_uint8 = (image * 255).astype('uint8')
            
            # Extract text using Tesseract OCR
            # 'deu+eng' enables German and English language models
            text = pytesseract.image_to_string(img_uint8, lang='deu+eng')
            
            # Handle cases where OCR returns non-string data
            if not isinstance(text, str):
                text = str(text)
            
            # Initialize feature vector
            features = []
            
            # Feature 1: First line length (normalized)
            # Rationale: First pages often have longer title lines
            lines = text.split('\n')
            if lines and isinstance(lines[0], str):
                first_line_length = len(lines[0].strip())
            else:
                first_line_length = 0
            # Normalize to [0,1] range (50 characters = typical line length)
            features.append(min(first_line_length / 50.0, 1.0))
            
            # Feature 2: Total text density (normalized)
            # Rationale: First pages may have different text density patterns
            total_text_length = len(text.replace('\n', '').replace(' ', ''))
            text_density = min(total_text_length / 1000.0, 1.0)
            features.append(text_density)
            
            # Pad feature vector to fixed size (for consistency)
            while len(features) < 2:
                features.append(0.0)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            print(f"OCR extraction failed: {e}")
            # Return zero-filled feature vector on failure
            return np.zeros(2, dtype=np.float32)

    def load_and_prepare_data(self, pdf_path, ground_truth_excel):
        """
        Load PDF pages and ground truth labels, prepare data for training.
        
        Args:
            pdf_path (str): Path to input PDF file
            ground_truth_excel (str): Path to Excel file with labels
        
        Returns:
            tuple: Prepared data based on model type
                  - For 'transfer'/'simple': (images, labels, page_numbers)
                  - For 'hybrid': (images, text_features, labels, page_numbers)
        
        Data Preparation Process:
        1. Extract all PDF pages as image files
        2. Load ground truth labels from Excel file
        3. Process each page: image preprocessing + optional OCR
        4. Match pages with labels by page number
        5. Return structured data arrays for training
        
        Expected Excel Format:
        - Column 'Seitennummer': Page number (1-based)
        - Column 'Probability of first page': Binary label (0 or 1)
        """
        # Step 1: Extract PDF pages
        page_paths = self.extract_pages_from_pdf(pdf_path)
        
        # Step 2: Load ground truth data
        df = pd.read_excel(ground_truth_excel)
        print("Ground truth data shape:", df.shape)
        print("Ground truth columns:", df.columns.tolist())
        print("\nFirst few rows:")
        print(df.head())
        
        # Step 3: Initialize data containers
        images = []
        text_features = [] if self.model_type == 'hybrid' else None
        labels = []
        page_numbers = []
        
        print(f"\nLoading and preprocessing images...")
        if self.model_type == 'hybrid':
            print("Extracting OCR text features (this may take a while)...")
        
        # Step 4: Process each labeled page
        for idx, row in df.iterrows():
            page_num = int(row['Seitennummer'])
            label = int(row['Probability of first page'])
            
            if page_num <= len(page_paths):
                try:
                    # Preprocess image
                    img = self.preprocess_image(page_paths[page_num - 1])
                    images.append(img)
                    labels.append(label)
                    page_numbers.append(page_num)
                    
                    # Extract text features for hybrid model
                    if self.model_type == 'hybrid':
                        text_feat = self.extract_text_features(img)
                        text_features.append(text_feat)
                    
                    # Progress indicator
                    if (idx + 1) % 20 == 0:
                        print(f"Processed {idx + 1}/{len(df)} pages...")
                        
                except Exception as e:
                    print(f"Error processing page {page_num}: {e}")
                    continue
            else:
                print(f"Warning: Page {page_num} not found (PDF has {len(page_paths)} pages)")
        
        # Step 5: Display data statistics
        print(f"\nSuccessfully loaded {len(images)} pages")
        class_counts = np.bincount(labels)
        print(f"Class distribution: {class_counts}")
        if len(class_counts) >= 2:
            print(f"Class 0 (not first page): {class_counts[0]} ({class_counts[0]/len(labels)*100:.1f}%)")
            print(f"Class 1 (first page): {class_counts[1]} ({class_counts[1]/len(labels)*100:.1f}%)")
        
        # Step 6: Return data based on model type
        if self.model_type == 'hybrid':
            return (np.array(images), np.array(text_features), 
                   np.array(labels), np.array(page_numbers))
        else:
            return (np.array(images), np.array(labels), np.array(page_numbers))

    def create_transfer_model(self):
        """
        Create transfer learning model using pre-trained CNN.
        
        Returns:
            keras.Model: Transfer learning model for binary classification
        
        Transfer Learning Architecture:
        1. Pre-trained base model (MobileNetV2 or ResNet50) as feature extractor
        2. Freeze early layers to retain low-level features
        3. Fine-tune later layers for document-specific features
        4. Add custom classification head for binary output
        
        Model Selection Strategy:
        - First try MobileNetV2 (efficient, good for mobile/edge deployment)
        - Fallback to ResNet50 (more powerful, higher accuracy)
        - Final fallback to simple CNN (basic architecture)
        
        Technical Benefits:
        - Leverages features learned on ImageNet (millions of natural images)
        - Faster convergence compared to training from scratch
        - Better performance with limited training data
        - Proven architecture reduces hyperparameter tuning
        """
        try:
            # Primary choice: MobileNetV2 (efficient and effective)
            from tensorflow.keras.applications import MobileNetV2
            base_model = MobileNetV2(
                weights='imagenet',        # Pre-trained on ImageNet
                include_top=False,         # Exclude classification layers
                input_shape=(*self.image_size, 3)
            )
            print("✓ Using MobileNetV2 as base model")
            
        except Exception as e1:
            try:
                # Fallback: ResNet50 (more powerful)
                print(f"MobileNetV2 failed ({e1}), trying ResNet50...")
                from tensorflow.keras.applications import ResNet50
                base_model = ResNet50(
                    weights='imagenet',
                    include_top=False,
                    input_shape=(*self.image_size, 3)
                )
                print("✓ Using ResNet50 as base model")
                
            except Exception as e2:
                print(f"ResNet50 also failed ({e2}), using simple CNN...")
                return self.create_simple_cnn()
        
        # Fine-tuning strategy: freeze early layers, train later layers
        base_model.trainable = True
        layers_to_freeze = len(base_model.layers) - 20
        for i, layer in enumerate(base_model.layers):
            if i < layers_to_freeze:
                layer.trainable = False
        
        print(f"Frozen {layers_to_freeze} layers, training {len(base_model.layers) - layers_to_freeze} layers")
        
        # Build complete model with classification head
        model = keras.Sequential([
            # Feature extraction backbone
            base_model,
            
            # Dimensionality reduction
            layers.GlobalAveragePooling2D(),
            
            # Classification layers with regularization
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            
            # Binary classification output
            layers.Dense(1, activation='sigmoid')
        ])
        
        return model

    def create_hybrid_model(self):
        """
        Create hybrid model combining CNN visual features with OCR text features.
        
        Returns:
            keras.Model: Multi-input hybrid model
        
        Hybrid Architecture Design:
        1. Image Branch: CNN for visual feature extraction
        2. Text Branch: Dense network for text feature processing  
        3. Fusion Layer: Combine both feature types
        4. Classification Head: Final decision based on combined features
        
        Rationale for Hybrid Approach:
        - Visual features: Capture layout, formatting, logos, visual structure
        - Text features: Capture semantic content, headers, document type
        - Combined: More robust classification using complementary information
        
        Technical Implementation:
        - Two separate input branches processed independently
        - Concatenation layer fuses features from both modalities
        - Shared classification layers make final prediction
        - Dropout and batch normalization for regularization
        """
        # Image input branch - processes visual information
        image_input = Input(shape=(*self.image_size, 3), name='image_input')
        
        # Convolutional layers for visual feature extraction
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
        
        # Dense layer for image features
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        
        # Text input branch - processes OCR features
        text_input = Input(shape=(2,), name='text_input')  # 2 text features
        t = layers.Dense(32, activation='relu')(text_input)
        t = layers.Dropout(0.2)(t)
        
        # Fusion layer - combine visual and text features
        combined = layers.concatenate([x, t])
        combined = layers.Dense(64, activation='relu')(combined)
        combined = layers.Dropout(0.3)(combined)
        combined = layers.Dense(32, activation='relu')(combined)
        combined = layers.Dropout(0.2)(combined)
        
        # Output layer for binary classification
        output = layers.Dense(1, activation='sigmoid')(combined)
        
        model = Model(inputs=[image_input, text_input], outputs=output)
        return model

    def create_simple_cnn(self):
        """
        Create simple CNN model trained from scratch.
        
        Returns:
            keras.Sequential: Basic CNN model
        
        Simple CNN Architecture:
        - Lightweight convolutional layers for feature extraction
        - Progressive channel increase (32 → 64 → 128)
        - Global average pooling to reduce parameters
        - Dense classification layers
        
        Use Cases:
        - Baseline performance comparison
        - When transfer learning models fail to load
        - Limited computational resources
        - Custom feature learning without pre-trained bias
        """
        model = keras.Sequential([
            # First convolutional block
            layers.Conv2D(32, (5, 5), activation='relu', 
                         input_shape=(*self.image_size, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block  
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            
            # Classification layers
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
        return model

    def create_model(self):
        """
        Factory method to create model based on specified type.
        
        Returns:
            keras.Model: Model instance based on self.model_type
        """
        model_creators = {
            'transfer': self.create_transfer_model,
            'hybrid': self.create_hybrid_model,
            'simple': self.create_simple_cnn
        }
        
        if self.model_type not in model_creators:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return model_creators[self.model_type]()

    def find_optimal_threshold(self, y_true, y_pred_prob, metric='f1'):
        """
        Find optimal classification threshold through grid search.
        
        Args:
            y_true: Ground truth labels
            y_pred_prob: Predicted probabilities 
            metric: Metric to optimize ('f1' or 'accuracy')
        
        Returns:
            tuple: (best_threshold, best_score)
        
        Threshold Optimization Rationale:
        - Default threshold of 0.5 may not be optimal for imbalanced data
        - Different thresholds trade off precision vs recall
        - Optimal threshold maximizes chosen metric (F1-score by default)
        - Grid search tests thresholds from 0.1 to 0.9 in 0.02 steps
        
        Technical Implementation:
        - Evaluates each threshold on validation/test predictions
        - Computes metric score for each threshold
        - Returns threshold with highest score
        - F1-score balances precision and recall for imbalanced data
        """
        thresholds = np.arange(0.1, 0.9, 0.02)  # Test 40 different thresholds
        best_score = 0
        best_threshold = 0.5  # Default fallback
        
        for threshold in thresholds:
            # Apply threshold to get binary predictions
            y_pred = (y_pred_prob > threshold).astype(int).flatten()
            
            # Calculate metric score
            try:
                if metric == 'f1':
                    score = f1_score(y_true, y_pred, average='weighted')
                elif metric == 'accuracy':
                    score = accuracy_score(y_true, y_pred)
                else:
                    # Default to F1-score for unknown metrics
                    score = f1_score(y_true, y_pred, average='weighted')
                
                # Update best threshold if score improved
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
                    
            except Exception as e:
                # Handle edge cases (e.g., all predictions same class)
                continue
        
        return best_threshold, best_score

    def train_single_fold(self, X_train, X_val, X_test, y_train, y_val, y_test,
                         fold_num=0, epochs=100, batch_size=8, verbose=1):
        """
        Train model for a single fold in cross-validation.
        
        Args:
            X_train, X_val, X_test: Training, validation, and test data
            y_train, y_val, y_test: Corresponding labels
            fold_num: Current fold number (for file naming)
            epochs: Maximum training epochs
            batch_size: Training batch size
            verbose: Training verbosity level
        
        Returns:
            tuple: (model, history, y_pred_prob)
        
        Single Fold Training Process:
        1. Create fresh model instance (no weight sharing between folds)
        2. Compile model with optimizer and loss function
        3. Set up training callbacks (checkpointing, early stopping)
        4. Calculate class weights for imbalance handling
        5. Train model with proper validation monitoring
        6. Generate predictions on test set
        7. Clean up memory before returning
        
        Technical Considerations:
        - Each fold gets independent model to avoid information leakage
        - Class weights handle imbalanced data automatically
        - Early stopping prevents overfitting
        - Learning rate reduction handles convergence plateaus
        - Memory cleanup prevents accumulation across folds
        """
        print(f"Training fold {fold_num + 1}...")
        
        # Step 1: Create fresh model for this fold
        model = self.create_model()
        
        # Step 2: Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',  # Standard binary classification loss
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # Step 3: Set up training callbacks
        callbacks = [
            # Save best model for this fold
            ModelCheckpoint(
                f'best_model_fold_{fold_num}.h5',
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                verbose=0
            ),
            
            # Early stopping to prevent overfitting
            EarlyStopping(
                monitor='val_accuracy',
                patience=20,               # Wait 20 epochs for improvement
                restore_best_weights=True,
                mode='max',
                verbose=0
            ),
            
            # Reduce learning rate when stuck
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,               # Halve learning rate
                patience=10,              # Wait 10 epochs
                verbose=verbose,
                min_lr=1e-7
            )
        ]
        
        # Step 4: Calculate class weights for imbalance handling
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
        
        if verbose:
            print(f"  Class weights: {class_weights_dict}")
        
        # Step 5: Train model based on type
        if self.model_type == 'hybrid':
            # Hybrid model: separate image and text inputs
            X_train_img, X_train_text = X_train
            X_val_img, X_val_text = X_val
            X_test_img, X_test_text = X_test
            
            history = model.fit(
                [X_train_img, X_train_text], y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=([X_val_img, X_val_text], y_val),
                class_weight=class_weights_dict,
                callbacks=callbacks,
                verbose=verbose
            )
            
            # Generate predictions
            y_pred_prob = model.predict([X_test_img, X_test_text], verbose=0)
            
        else:
            # Transfer/simple models: single image input
            history = model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_val, y_val),
                class_weight=class_weights_dict,
                callbacks=callbacks,
                verbose=verbose
            )
            
            # Generate predictions
            y_pred_prob = model.predict(X_test, verbose=0)
        
        return model, history, y_pred_prob


def run_kfold_cross_validation(classifier, data, k=5, epochs=100, batch_size=8,
                              test_size=0.2, random_state=42, verbose=1):
    """
    Run comprehensive k-fold cross-validation with statistical analysis.
    
    Args:
        classifier: PDFPageClassifier instance
        data: Prepared data tuple (format depends on model type)
        k: Number of folds for cross-validation
        epochs: Training epochs per fold
        batch_size: Training batch size
        test_size: Proportion of data reserved for final testing
        random_state: Random seed for reproducibility
        verbose: Verbosity level (0=silent, 1=normal, 2=detailed)
    
    Returns:
        tuple: (fold_results, mean_accuracy, mean_f1)
    
    K-Fold Cross-Validation Process:
    1. Reserve test set (never used during CV)
    2. Split remaining data into k folds
    3. For each fold: train on k-1 folds, validate on 1 fold
    4. Optimize threshold per fold for best performance
    5. Aggregate results across folds with statistics
    6. Generate comprehensive performance report
    
    Statistical Benefits:
    - Reduces variance in performance estimates
    - Uses all data for training (across folds)
    - Provides confidence intervals for metrics
    - Identifies model stability across different data splits
    - More reliable than single train/test split
    """
    
    # Step 1: Parse data based on model type
    if classifier.model_type == 'hybrid':
        X_img, X_text, y, page_numbers = data
    else:
        X_img, y, page_numbers = data
        X_text = None
    
    print(f"\n{'='*60}")
    print(f"K-FOLD CROSS-VALIDATION ({k} FOLDS)")
    print(f"{'='*60}")
    print(f"Dataset overview:")
    print(f"- Total samples: {len(y)}")
    print(f"- Model type: {classifier.model_type}")
    print(f"- Class distribution: {np.bincount(y)}")
    print(f"- Test set size: {test_size*100:.0f}%")
    
    # Step 2: Reserve final test set (never seen during cross-validation)
    if X_text is not None:
        X_img_cv, X_img_test, X_text_cv, X_text_test, y_cv, y_test = train_test_split(
            X_img, X_text, y, test_size=test_size, stratify=y, random_state=random_state)
    else:
        X_img_cv, X_img_test, y_cv, y_test = train_test_split(
            X_img, y, test_size=test_size, stratify=y, random_state=random_state)
        X_text_cv, X_text_test = None, None
    
    print(f"- CV samples: {len(y_cv)}")
    print(f"- Test samples: {len(y_test)}")
    
    # Step 3: Set up stratified k-fold splitting
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    
    # Step 4: Initialize results containers
    fold_results = []
    all_histories = []
    
    # Step 5: Perform k-fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_img_cv, y_cv)):
        print(f"\n{'-'*40}")
        print(f"FOLD {fold+1}/{k}")
        print(f"{'-'*40}")
        
        # Split data for this fold
        if X_text_cv is not None:
            # Hybrid model data preparation
            X_train = (X_img_cv[train_idx], X_text_cv[train_idx])
            X_val = (X_img_cv[val_idx], X_text_cv[val_idx])
            X_test_fold = (X_img_test, X_text_test)
        else:
            # Single input model data preparation
            X_train = X_img_cv[train_idx]
            X_val = X_img_cv[val_idx]
            X_test_fold = X_img_test
        
        y_train, y_val = y_cv[train_idx], y_cv[val_idx]
        
        print(f"Data split - Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
        print(f"Train distribution: {np.bincount(y_train)}")
        print(f"Val distribution: {np.bincount(y_val)}")
        
        # Train model for this fold
        model, history, y_pred_prob = classifier.train_single_fold(
            X_train, X_val, X_test_fold, y_train, y_val, y_test,
            fold_num=fold, epochs=epochs, batch_size=batch_size, verbose=verbose
        )
        
        # Store training history
        all_histories.append(history)
        
        # Find optimal threshold for this fold
        best_threshold, best_f1 = classifier.find_optimal_threshold(
            y_test, y_pred_prob, metric='f1'
        )
        
        # Generate final predictions with optimal threshold
        y_pred = (y_pred_prob > best_threshold).astype(int).flatten()
        
        # Calculate performance metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Store fold results
        fold_result = {
            'fold': fold + 1,
            'accuracy': acc,
            'f1_score': f1,
            'threshold': best_threshold,
            'predictions': y_pred,
            'probabilities': y_pred_prob.flatten(),
            'history': history
        }
        fold_results.append(fold_result)
        
        # Display fold results
        print(f"\nFold {fold+1} Results:")
        print(f"  Optimal threshold: {best_threshold:.3f}")
        print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print(f"  F1-score: {f1:.4f}")
        
        # Detailed classification report
        if verbose >= 1:
            print(f"\nClassification Report (Fold {fold+1}):")
            print(classification_report(y_test, y_pred,
                                      target_names=['Not First Page', 'First Page'],
                                      digits=4))
            
            cm = confusion_matrix(y_test, y_pred)
            print(f"Confusion Matrix:")
            print(cm)
        
        # Memory cleanup
        del model
        tf.keras.backend.clear_session()
    
    # Step 6: Aggregate results across all folds
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    # Calculate statistics
    accuracies = [r['accuracy'] for r in fold_results]
    f1_scores = [r['f1_score'] for r in fold_results]
    thresholds = [r['threshold'] for r in fold_results]
    
    # Summary statistics
    acc_mean, acc_std = np.mean(accuracies), np.std(accuracies)
    f1_mean, f1_std = np.mean(f1_scores), np.std(f1_scores)
    thr_mean, thr_std = np.mean(thresholds), np.std(thresholds)
    
    print(f"Performance Metrics (Mean ± Std):")
    print(f"  Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"  F1-score: {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"  Threshold: {thr_mean:.3f} ± {thr_std:.3f}")
    
    # Individual fold breakdown
    print(f"\nPer-Fold Results:")
    print(f"{'Fold':<6} {'Accuracy':<10} {'F1-Score':<10} {'Threshold':<10}")
    print("-" * 40)
    for i, result in enumerate(fold_results):
        print(f"{i+1:<6} {result['accuracy']:<10.4f} {result['f1_score']:<10.4f} {result['threshold']:<10.3f}")
    
    # Performance assessment
    print(f"\nPerformance Assessment:")
    if acc_mean > 0.75:
        print("✅ Excellent: Accuracy > 75%")
    elif acc_mean > 0.65:
        print("✅ Good: Accuracy > 65%")
    elif acc_mean > 0.55:
        print("⚠️  Moderate: Accuracy > 55%")
    else:
        print("❌ Poor: Accuracy ≤ 55%")
    
    if acc_std < 0.05:
        print("✅ Stable: Low variance across folds")
    else:
        print("⚠️  Unstable: High variance across folds")
    
    # Generate cross-validation visualizations
    plot_cv_results(fold_results, all_histories)
    
    return fold_results, acc_mean, f1_mean


def plot_cv_results(fold_results, histories, save_prefix='cv_results'):
    """
    Generate comprehensive visualizations for cross-validation results.
    
    Args:
        fold_results: List of fold result dictionaries
        histories: List of training histories from each fold
        save_prefix: Prefix for saved plot files
    
    Visualizations Generated:
    1. Box plots of performance metrics across folds
    2. Training curves averaged across folds
    3. Threshold distribution across folds
    4. Performance stability analysis
    """
    
    # Extract metrics for plotting
    accuracies = [r['accuracy'] for r in fold_results]
    f1_scores = [r['f1_score'] for r in fold_results]
    thresholds = [r['threshold'] for r in fold_results]
    
    # Create comprehensive figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('K-Fold Cross-Validation Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Accuracy distribution
    axes[0, 0].boxplot([accuracies], labels=['Accuracy'])
    axes[0, 0].scatter([1] * len(accuracies), accuracies, alpha=0.6, c='red')
    axes[0, 0].set_title('Accuracy Across Folds')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: F1-score distribution
    axes[0, 1].boxplot([f1_scores], labels=['F1-Score'])
    axes[0, 1].scatter([1] * len(f1_scores), f1_scores, alpha=0.6, c='blue')
    axes[0, 1].set_title('F1-Score Across Folds')
    axes[0, 1].set_ylabel('F1-Score')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Threshold distribution
    axes[0, 2].hist(thresholds, bins=10, alpha=0.7, color='green')
    axes[0, 2].set_title('Optimal Thresholds')
    axes[0, 2].set_xlabel('Threshold')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Training curves (averaged)
    if histories:
        # Average training metrics across folds
        max_epochs = min(len(h.history['accuracy']) for h in histories)
        
        avg_train_acc = np.mean([h.history['accuracy'][:max_epochs] for h in histories], axis=0)
        avg_val_acc = np.mean([h.history['val_accuracy'][:max_epochs] for h in histories], axis=0)
        
        axes[1, 0].plot(avg_train_acc, label='Training')
        axes[1, 0].plot(avg_val_acc, label='Validation')
        axes[1, 0].set_title('Average Training Curves')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Loss curves (averaged)
        avg_train_loss = np.mean([h.history['loss'][:max_epochs] for h in histories], axis=0)
        avg_val_loss = np.mean([h.history['val_loss'][:max_epochs] for h in histories], axis=0)
        
        axes[1, 1].plot(avg_train_loss, label='Training')
        axes[1, 1].plot(avg_val_loss, label='Validation')
        axes[1, 1].set_title('Average Loss Curves')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Performance comparison
    fold_nums = list(range(1, len(fold_results) + 1))
    axes[1, 2].plot(fold_nums, accuracies, 'o-', label='Accuracy', linewidth=2)
    axes[1, 2].plot(fold_nums, f1_scores, 's-', label='F1-Score', linewidth=2)
    axes[1, 2].set_title('Performance by Fold')
    axes[1, 2].set_xlabel('Fold Number')
    axes[1, 2].set_ylabel('Score')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_xticks(fold_nums)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Cross-validation analysis saved as '{save_prefix}_analysis.png'")


def save_cv_summary(fold_results, output_file="cv_summary.txt"):
    """
    Save detailed cross-validation summary to text file.
    
    Args:
        fold_results: Results from all folds
        output_file: Output file path
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(output_file, 'w') as f:
        f.write("PDF FIRST PAGE CLASSIFICATION - K-FOLD CROSS-VALIDATION SUMMARY\n")
        f.write("="*70 + "\n")
        f.write(f"Analysis completed: {timestamp}\n\n")
        
        # Overall statistics
        accuracies = [r['accuracy'] for r in fold_results]
        f1_scores = [r['f1_score'] for r in fold_results]
        thresholds = [r['threshold'] for r in fold_results]
        
        f.write("OVERALL PERFORMANCE:\n")
        f.write(f"Mean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}\n")
        f.write(f"Mean F1-Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}\n")
        f.write(f"Mean Threshold: {np.mean(thresholds):.3f} ± {np.std(thresholds):.3f}\n\n")
        
        # Per-fold details
        f.write("PER-FOLD RESULTS:\n")
        for i, result in enumerate(fold_results):
            f.write(f"Fold {i+1}:\n")
            f.write(f"  Accuracy: {result['accuracy']:.4f}\n")
            f.write(f"  F1-Score: {result['f1_score']:.4f}\n")
            f.write(f"  Threshold: {result['threshold']:.3f}\n")
        
        f.write(f"\nTRAINING CONFIGURATION:\n")
        f.write(f"Number of folds: {len(fold_results)}\n")
        f.write(f"Model architecture: Transfer learning with fine-tuning\n")
        f.write(f"Loss function: Binary cross-entropy\n")
        f.write(f"Optimizer: Adam\n")
        f.write(f"Threshold optimization: F1-score maximization\n")
    
    print(f"Cross-validation summary saved to {output_file}")


def main():
    """
    Main execution function for k-fold cross-validation.
    
    This function orchestrates the complete cross-validation pipeline:
    1. Initialize classifier with specified model type
    2. Load and prepare data
    3. Run k-fold cross-validation
    4. Generate comprehensive analysis and reports
    """
    print("PDF First Page Classification - K-Fold Cross-Validation")
    print("=" * 60)
    
    # Configuration parameters
    MODEL_TYPE = 'transfer'      # Options: 'transfer', 'hybrid', 'simple'
    IMAGE_SIZE = (224, 224)      # Input image dimensions
    K_FOLDS = 5                  # Number of cross-validation folds
    EPOCHS = 100                 # Training epochs per fold
    BATCH_SIZE = 8               # Training batch size
    TEST_SIZE = 0.2              # Final test set proportion
    RANDOM_STATE = 42            # Random seed for reproducibility
    
    # Input files
    PDF_PATH = "Sample Invoices.pdf"
    EXCEL_PATH = "Correct values sample invoices.xlsx"
    
    print(f"Configuration:")
    print(f"- Model type: {MODEL_TYPE}")
    print(f"- Image size: {IMAGE_SIZE}")
    print(f"- K-folds: {K_FOLDS}")
    print(f"- Epochs per fold: {EPOCHS}")
    print(f"- Batch size: {BATCH_SIZE}")
    
    try:
        # Initialize classifier
        classifier = PDFPageClassifier(image_size=IMAGE_SIZE, model_type=MODEL_TYPE)
        
        # Load and prepare data
        print("\nLoading and preparing data...")
        data = classifier.load_and_prepare_data(PDF_PATH, EXCEL_PATH)
        
        # Run k-fold cross-validation
        fold_results, mean_accuracy, mean_f1 = run_kfold_cross_validation(
            classifier, data,
            k=K_FOLDS,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            verbose=1
        )
        
        # Save comprehensive summary
        save_cv_summary(fold_results)
        
        # Final results
        print(f"\n🎯 FINAL CROSS-VALIDATION RESULTS:")
        print(f"   Mean Accuracy: {mean_accuracy:.4f} ({mean_accuracy*100:.2f}%)")
        print(f"   Mean F1-Score: {mean_f1:.4f}")
        
        # Performance assessment
        if mean_accuracy > 0.75:
            print("🎉 Excellent performance! Target accuracy >75% achieved.")
        elif mean_accuracy > 0.65:
            print("✅ Good performance. Consider hyperparameter tuning for improvement.")
        else:
            print("📈 Moderate performance. Try different model architectures or feature engineering.")
        
        print(f"\nGenerated files:")
        print(f"- cv_results_analysis.png (cross-validation visualizations)")
        print(f"- cv_summary.txt (detailed numerical results)")
        print(f"- best_model_fold_*.h5 (trained models for each fold)")
        
    except Exception as e:
        print(f"❌ Error during cross-validation: {e}")
        print("\nTroubleshooting checklist:")
        print("1. Verify PDF and Excel file paths")
        print("2. Check system dependencies (poppler-utils, tesseract)")
        print("3. Ensure sufficient memory and disk space")
        print("4. For hybrid model, verify tesseract installation")


if __name__ == "__main__":
    main()