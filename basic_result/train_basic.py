#!/usr/bin/env python3
"""
PDF First Page Classification - Basic Training Script

This script implements a transfer learning approach using MobileNetV2/ResNet50
to classify PDF pages as either "first page" or "continuation page".

Key Features:
- Transfer learning with pre-trained CNN models
- Data augmentation for document images
- Class imbalance handling with focal loss and class weights
- Comprehensive visualization and metrics tracking
- Annotation cleaning (removes highlights and markup)

Technical Approach:
1. Extract PDF pages as images (200 DPI)
2. Clean annotations (yellow highlights, colored circles)
3. Apply transfer learning with MobileNetV2
4. Use focal loss to handle class imbalance
5. Generate comprehensive training visualizations
"""

import os
import pandas as pd
import numpy as np
from pdf2image import convert_from_path
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# SSL fix for downloading pre-trained models
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class PDFPageClassifier:
    def __init__(self, image_size=(224, 224)):
        """
        Initialize the PDF page classifier.
        
        Args:
            image_size (tuple): Target size for input images (width, height)
                              224x224 is optimal for transfer learning models
        
        Technical Notes:
        - 224x224 is the standard input size for ImageNet pre-trained models
        - Smaller sizes (e.g., 128x128) reduce training time but may lose important details
        - Larger sizes (e.g., 512x512) capture more detail but require more memory
        """
        self.image_size = image_size
        self.model = None
        self.training_history = None

    def extract_pages_from_pdf(self, pdf_path, output_dir="extracted_pages"):
        """
        Convert PDF pages to individual JPEG images.
        
        Args:
            pdf_path (str): Path to input PDF file
            output_dir (str): Directory to save extracted page images
        
        Returns:
            list: Paths to extracted page image files
        
        Technical Details:
        - Uses pdf2image library which requires poppler-utils
        - 200 DPI provides good balance between quality and file size
        - Higher DPI (300+) captures more text detail but increases processing time
        - Images saved as JPEG with 90% quality to balance size vs quality
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("Extracting pages from PDF...")
        try:
            # Convert PDF to images at 200 DPI
            # DPI (Dots Per Inch) controls image resolution
            # 200 DPI: Good balance of quality vs processing speed
            # 150 DPI: Faster but may lose text clarity
            # 300 DPI: Higher quality but slower processing
            pages = convert_from_path(pdf_path, dpi=200)
        except Exception as e:
            print(f"Error extracting PDF: {e}")
            print("Make sure poppler-utils is installed:")
            print("  macOS: brew install poppler")
            print("  Ubuntu: sudo apt-get install poppler-utils")
            raise

        page_paths = []
        for i, page in enumerate(pages, 1):
            page_path = os.path.join(output_dir, f"page_{i:03d}.jpg")
            # Save with 90% JPEG quality - good compression with minimal quality loss
            page.save(page_path, 'JPEG', quality=95)
            page_paths.append(page_path)

        print(f"Extracted {len(pages)} pages to {output_dir}")
        return page_paths

    def clean_annotated_image(self, image):
        """
        Remove colored annotations that could interfere with classification.
        
        Args:
            image (numpy.ndarray): RGB image array
        
        Returns:
            numpy.ndarray: Image with annotations replaced by white pixels
        
        Technical Approach:
        1. Convert RGB to HSV color space for better color detection
        2. Create masks for different annotation colors using HSV ranges
        3. Combine masks using bitwise operations
        4. Dilate masks to capture annotation boundaries
        5. Replace detected regions with white background
        
        Color Ranges (HSV):
        - Yellow highlights: H=20-90, S=100-255, V=100-255
        - Green circles: H=40-80, S=50-255, V=50-255
        - Blue numbers: H=100-130, S=50-255, V=50-255
        """
        # Convert RGB to HSV color space
        # HSV (Hue, Saturation, Value) is better for color-based segmentation
        # because hue represents the color itself, independent of lighting
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Create mask for yellow highlights
        # HSV values are more robust to lighting variations than RGB
        lower_yellow = np.array([20, 100, 100])  # Lower bound for yellow in HSV
        upper_yellow = np.array([90, 255, 255])  # Upper bound for yellow in HSV
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Create mask for green circles (annotation markers)
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Create mask for blue numbers
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Combine all annotation masks using bitwise OR
        # This creates a single mask covering all annotation types
        annotation_mask = cv2.bitwise_or(yellow_mask, green_mask)
        annotation_mask = cv2.bitwise_or(annotation_mask, blue_mask)
        
        # Dilate mask to capture annotation boundaries
        # Morphological dilation expands white regions to catch annotation edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        annotation_mask = cv2.dilate(annotation_mask, kernel, iterations=1)
        
        # Replace annotated areas with white background
        cleaned_image = image.copy()
        cleaned_image[annotation_mask > 0] = [255, 255, 255]  # White RGB values
        
        return cleaned_image

    def preprocess_image(self, image_path):
        """
        Load and preprocess image for CNN input.
        
        Args:
            image_path (str): Path to image file
        
        Returns:
            numpy.ndarray: Preprocessed image ready for CNN (normalized, resized)
        
        Preprocessing Steps:
        1. Load image using OpenCV (loads as BGR by default)
        2. Convert BGR to RGB (CNNs expect RGB format)
        3. Clean annotations to avoid model confusion
        4. Resize to target dimensions (224x224 for transfer learning)
        5. Normalize pixel values to [0,1] range (CNNs work better with normalized data)
        
        Technical Notes:
        - OpenCV loads images as BGR, but CNNs expect RGB
        - Normalization (dividing by 255) converts 0-255 pixel values to 0-1 range
        - This helps with training stability and convergence
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert from BGR (OpenCV default) to RGB (CNN expected)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Remove colored annotations that could confuse the model
        image = self.clean_annotated_image(image)
        
        # Resize to target dimensions required by the CNN
        # Interpolation method INTER_LINEAR provides good quality
        image = cv2.resize(image, self.image_size)
        
        # Normalize pixel values from [0, 255] to [0, 1]
        # This improves training stability and convergence speed
        image = image.astype('float32') / 255.0
        
        return image

    def create_data_generator(self):
        """
        Create data augmentation generator for document images.
        
        Returns:
            ImageDataGenerator: Configured augmentation generator
        
        Data Augmentation Rationale:
        - Artificially expands training dataset
        - Improves model generalization
        - Helps prevent overfitting
        
        Document-Specific Augmentation:
        - Minimal rotation (1°): Documents are usually upright
        - Small shifts (2%): Slight positional variations
        - Subtle zoom (3%): Minor scale changes
        - Brightness variation (90-110%): Different lighting conditions
        - NO horizontal flip: Would create invalid document orientations
        
        Technical Notes:
        - fill_mode='constant' fills empty pixels with constant value (0)
        - Conservative augmentation prevents unrealistic document variations
        """
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        return ImageDataGenerator(
            rotation_range=1,              # Max rotation: ±1 degree
            width_shift_range=0.02,        # Max horizontal shift: 2% of image width
            height_shift_range=0.02,       # Max vertical shift: 2% of image height
            zoom_range=0.03,              # Zoom range: 97-103% of original size
            brightness_range=[0.9, 1.1],   # Brightness: 90-110% of original
            fill_mode='constant',          # Fill empty pixels with constant value
            horizontal_flip=False          # Never flip documents horizontally
        )

    def load_and_prepare_data(self, pdf_path, ground_truth_excel):
        """
        Load PDF and ground truth labels, prepare training data.
        
        Args:
            pdf_path (str): Path to PDF file
            ground_truth_excel (str): Path to Excel file with labels
        
        Returns:
            tuple: (images, labels, page_numbers) as numpy arrays
        
        Ground Truth Format:
        The Excel file should contain:
        - 'Seitennummer': Page number (1-based indexing)
        - 'Probability of first page': Binary label (0 or 1)
        
        Data Processing:
        1. Extract all PDF pages as images
        2. Load ground truth labels from Excel
        3. Match pages with labels by page number
        4. Preprocess each image (clean, resize, normalize)
        5. Create numpy arrays for efficient training
        """
        # Extract PDF pages to individual image files
        page_paths = self.extract_pages_from_pdf(pdf_path)

        # Load ground truth labels from Excel file
        df = pd.read_excel(ground_truth_excel)
        print("Ground truth data shape:", df.shape)
        print("Ground truth columns:", df.columns.tolist())
        print("\nFirst few rows:")
        print(df.head())

        # Initialize lists to store processed data
        images = []
        labels = []
        page_numbers = []

        print("\nLoading and preprocessing images...")
        for idx, row in df.iterrows():
            page_num = int(row['Seitennummer'])  # Page number (1-based)
            label = int(row['Probability of first page'])  # Binary label

            # Check if page exists in extracted images
            if page_num <= len(page_paths):
                image_path = page_paths[page_num - 1]  # Convert to 0-based index
                try:
                    # Preprocess image for CNN input
                    image = self.preprocess_image(image_path)
                    images.append(image)
                    labels.append(label)
                    page_numbers.append(page_num)
                except Exception as e:
                    print(f"Error processing page {page_num}: {e}")
                    continue
            else:
                print(f"Warning: Page {page_num} not found in PDF (PDF has {len(page_paths)} pages)")

        # Display data statistics
        print(f"\nSuccessfully loaded {len(images)} pages")
        class_counts = np.bincount(labels)
        print(f"Class distribution: {class_counts}")
        print(f"Class 0 (not first page): {class_counts[0]} pages ({class_counts[0]/len(labels)*100:.1f}%)")
        print(f"Class 1 (first page): {class_counts[1]} pages ({class_counts[1]/len(labels)*100:.1f}%)")

        return np.array(images), np.array(labels), np.array(page_numbers)

    def create_transfer_cnn_model(self):
        """
        Create transfer learning model using pre-trained CNN.
        
        Returns:
            keras.Model: Compiled CNN model for binary classification
        
        Transfer Learning Approach:
        1. Use pre-trained model (MobileNetV2 or ResNet50) as feature extractor
        2. Freeze most layers to retain learned features
        3. Fine-tune last few layers on document data
        4. Add custom classification head for binary classification
        
        Model Architecture:
        - Base: MobileNetV2 (efficient) or ResNet50 (powerful)
        - Feature extraction: GlobalAveragePooling2D
        - Classification layers: Dense layers with dropout for regularization
        - Output: Single sigmoid unit for binary classification
        
        Technical Benefits:
        - Leverages features learned on ImageNet (1M+ images)
        - Requires less training data than training from scratch
        - Often achieves better performance than custom CNNs
        - Faster training due to pre-trained weights
        """
        try:
            # Try MobileNetV2 first (efficient and lightweight)
            from tensorflow.keras.applications import MobileNetV2
            
            base_model = MobileNetV2(
                weights='imagenet',        # Use pre-trained ImageNet weights
                include_top=False,         # Exclude final classification layers
                input_shape=(*self.image_size, 3)  # RGB images
            )
            print("✓ Using MobileNetV2 as base model")
        except Exception as e:
            # Fallback to ResNet50 if MobileNetV2 fails
            print(f"MobileNetV2 failed: {e}")
            print("Trying ResNet50 instead...")
            from tensorflow.keras.applications import ResNet50
            
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.image_size, 3)
            )
            print("✓ Using ResNet50 as base model")
        
        # Fine-tuning strategy: freeze early layers, train later layers
        base_model.trainable = True
        # Freeze all layers except the last 30
        # This allows the model to adapt to document images while preserving
        # low-level feature detectors learned on ImageNet
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        
        # Build complete model with classification head
        model = keras.Sequential([
            # Pre-trained feature extractor
            base_model,
            
            # Global average pooling reduces spatial dimensions
            # Converts feature maps to fixed-size feature vectors
            layers.GlobalAveragePooling2D(),
            
            # Dense layers for classification with dropout for regularization
            layers.Dropout(0.6),                    # High dropout to prevent overfitting
            layers.Dense(512, activation='relu'),   # First classification layer
            layers.BatchNormalization(),            # Normalize layer inputs
            layers.Dropout(0.4),
            layers.Dense(128, activation='relu'),   # Second classification layer
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),    # Final hidden layer
            layers.Dropout(0.2),
            
            # Output layer for binary classification
            layers.Dense(1, activation='sigmoid')   # Sigmoid outputs probability [0,1]
        ])
        
        return model

    def focal_loss(self, alpha=0.75, gamma=2.0):
        """
        Implement focal loss for handling class imbalance.
        
        Args:
            alpha (float): Weighting factor for rare class (typically 0.25-0.75)
            gamma (float): Focusing parameter (typically 2.0)
        
        Returns:
            function: Focal loss function for Keras
        
        Focal Loss Theory:
        - Addresses class imbalance by down-weighting easy examples
        - Focuses training on hard, misclassified examples
        - Formula: FL(p_t) = -α(1-p_t)^γ log(p_t)
        
        Parameters:
        - α (alpha): Balances positive/negative examples
        - γ (gamma): Controls how much to down-weight easy examples
        
        Benefits over standard cross-entropy:
        - Better handles severe class imbalance
        - Improves performance on minority class
        - Reduces influence of easy negative examples
        """
        def focal_loss_fixed(y_true, y_pred):
            # Prevent log(0) by clipping predictions
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
            
            # Calculate focal loss components
            alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
            p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
            
            # Focal loss formula
            focal_loss = -alpha_t * tf.pow((1 - p_t), gamma) * tf.math.log(p_t)
            
            return tf.reduce_mean(focal_loss)
        
        return focal_loss_fixed

    def plot_training_metrics(self, history, save_path='training_metrics.png'):
        """
        Create comprehensive visualization of training metrics.
        
        Args:
            history: Keras training history object
            save_path (str): Path to save the plot
        
        Visualizations:
        1. Training vs Validation Accuracy
        2. Training vs Validation Loss
        3. Precision over epochs
        4. Recall over epochs
        
        Technical Notes:
        - Helps identify overfitting (train >> validation performance)
        - Shows training convergence and stability
        - Useful for hyperparameter tuning
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('PDF First Page Classification - Training Metrics', fontsize=16, fontweight='bold')
        
        # Plot 1: Accuracy
        axes[0, 0].plot(history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        axes[0, 0].plot(history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        axes[0, 0].set_title('Model Accuracy Over Time', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Loss
        axes[0, 1].plot(history.history['loss'], 'b-', label='Training Loss', linewidth=2)
        axes[0, 1].plot(history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0, 1].set_title('Model Loss Over Time', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Precision
        axes[1, 0].plot(history.history['precision'], 'g-', label='Training Precision', linewidth=2)
        axes[1, 0].plot(history.history['val_precision'], 'orange', label='Validation Precision', linewidth=2)
        axes[1, 0].set_title('Model Precision Over Time', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Recall
        axes[1, 1].plot(history.history['recall'], 'purple', label='Training Recall', linewidth=2)
        axes[1, 1].plot(history.history['val_recall'], 'brown', label='Validation Recall', linewidth=2)
        axes[1, 1].set_title('Model Recall Over Time', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Training metrics saved to {save_path}")

    def plot_confusion_matrix(self, y_true, y_pred, save_path='confusion_matrix.png'):
        """
        Create detailed confusion matrix visualization.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path (str): Path to save the plot
        
        Confusion Matrix Interpretation:
        - True Negatives (TN): Correctly identified non-first pages
        - False Positives (FP): Non-first pages incorrectly labeled as first
        - False Negatives (FN): First pages incorrectly labeled as non-first
        - True Positives (TP): Correctly identified first pages
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not First Page', 'First Page'],
                   yticklabels=['Not First Page', 'First Page'])
        plt.title('Confusion Matrix - PDF First Page Classification', fontweight='bold')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # Add performance metrics to the plot
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        plt.figtext(0.02, 0.02, f'Accuracy: {accuracy:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}', 
                   fontsize=12, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Confusion matrix saved to {save_path}")

    def train_model(self, X, y, page_numbers=None, test_size=0.15, validation_split=0.15, 
                   epochs=100, batch_size=8, stratify=True, random_state=42):
        """
        Train the PDF page classification model.
        
        Args:
            X: Image data (numpy array)
            y: Labels (numpy array)  
            page_numbers: Page numbers for tracking (optional)
            test_size: Proportion of data for testing
            validation_split: Proportion of training data for validation
            epochs: Maximum number of training epochs
            batch_size: Number of samples per training batch
            stratify: Whether to maintain class distribution in splits
            random_state: Random seed for reproducible results
        
        Returns:
            tuple: (history, X_test, y_test, y_pred, y_pred_prob)
        
        Training Process:
        1. Split data into train/validation/test sets
        2. Calculate class weights to handle imbalance
        3. Create and compile transfer learning model
        4. Set up training callbacks (checkpointing, early stopping)
        5. Train model with data augmentation
        6. Evaluate on test set and generate visualizations
        """
        total_samples = len(X)
        print(f"\n{'='*60}")
        print(f"TRAINING PDF FIRST PAGE CLASSIFIER")
        print(f"{'='*60}")
        print(f"Total samples: {total_samples}")
        print(f"Using {int((1-test_size)*100)}/{int(test_size*100)} train/test split")

        # Stratified train/test split to maintain class balance
        if stratify and len(np.unique(y)) > 1:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                stratify=y,  # Maintain class distribution
                random_state=random_state,
                shuffle=True
            )

            if page_numbers is not None:
                _, page_test = train_test_split(
                    page_numbers, 
                    test_size=test_size, 
                    stratify=y,
                    random_state=random_state,
                    shuffle=True
                )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=random_state,
                shuffle=True
            )

            if page_numbers is not None:
                _, page_test = train_test_split(
                    page_numbers, 
                    test_size=test_size, 
                    random_state=random_state,
                    shuffle=True
                )

        # Split training data into train/validation
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train, y_train,
            test_size=validation_split,
            stratify=y_train,
            random_state=random_state
        )

        print(f"\nData split completed:")
        print(f"Training samples: {len(X_train_final)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Training class distribution: {np.bincount(y_train_final.astype(int))}")
        print(f"Validation class distribution: {np.bincount(y_val.astype(int))}")
        print(f"Test class distribution: {np.bincount(y_test.astype(int))}")

        # Calculate class weights to handle imbalance
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(y_train_final), 
            y=y_train_final
        )
        class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
        
        print(f"\nClass weights: {class_weights_dict}")
        print("Explanation: Higher weights for minority class to balance training")

        # Create and compile model
        print("\nBuilding transfer learning model...")
        self.model = self.create_transfer_cnn_model()
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # Conservative learning rate
            loss=self.focal_loss(alpha=0.75, gamma=2.0),           # Focal loss for imbalance
            metrics=['accuracy', 'precision', 'recall']
        )

        print("\nModel Architecture:")
        self.model.summary()

        # Set up training callbacks
        callbacks = [
            # Save best model based on validation accuracy
            ModelCheckpoint(
                'best_first_page_classifier.h5',
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                mode='max',
                verbose=1
            ),
            
            # Stop early if no improvement
            EarlyStopping(
                monitor='val_accuracy',
                patience=20,  # Wait 20 epochs before stopping
                restore_best_weights=True,
                verbose=1,
                mode='max'
            ),
            
            # Reduce learning rate when stuck
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,      # Reduce by half
                patience=10,     # Wait 10 epochs
                min_lr=1e-7,     # Minimum learning rate
                verbose=1
            )
        ]

        # Set up data generators
        datagen = self.create_data_generator()
        val_datagen = keras.preprocessing.image.ImageDataGenerator()  # No augmentation for validation
        
        print(f"\nStarting training with {epochs} epochs...")
        print("Training configuration:")
        print(f"- Batch size: {batch_size}")
        print(f"- Data augmentation: Enabled for training data")
        print(f"- Loss function: Focal loss (α=0.75, γ=2.0)")
        print(f"- Optimizer: Adam (lr=0.0001)")
        
        # Train the model
        history = self.model.fit(
            datagen.flow(X_train_final, y_train_final, batch_size=batch_size),
            steps_per_epoch=max(1, len(X_train_final) // batch_size * 2),
            epochs=epochs,
            validation_data=val_datagen.flow(X_val, y_val, batch_size=batch_size),
            validation_steps=max(1, len(X_val) // batch_size),
            class_weight=class_weights_dict,
            callbacks=callbacks,
            verbose=1
        )
        
        # Store training history
        self.training_history = history

        # Evaluate on test set
        print(f"\n{'='*50}")
        print("EVALUATING MODEL ON TEST SET")
        print(f"{'='*50}")
        test_loss, test_accuracy, test_precision, test_recall = self.model.evaluate(X_test, y_test, verbose=0)

        print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")

        # Calculate F1 score
        if test_precision + test_recall > 0:
            f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall)
        else:
            f1_score = 0.0
        print(f"Test F1-Score: {f1_score:.4f}")

        # Generate predictions
        print("\nGenerating detailed predictions...")
        y_pred_prob = self.model.predict(X_test, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()

        # Classification report
        print(f"\n{'='*50}")
        print("DETAILED CLASSIFICATION REPORT")
        print(f"{'='*50}")
        print(classification_report(y_test, y_pred, target_names=['Not First Page', 'First Page']))

        # Confusion matrix analysis
        print(f"\n{'='*50}")
        print("CONFUSION MATRIX ANALYSIS")
        print(f"{'='*50}")
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(cm)
        print("\nInterpretation:")
        print(f"True Negatives (correct non-first pages): {cm[0,0]}")
        print(f"False Positives (incorrect first page predictions): {cm[0,1]}")
        print(f"False Negatives (missed first pages): {cm[1,0]}")
        print(f"True Positives (correct first page predictions): {cm[1,1]}")

        # Sample predictions
        if page_numbers is not None:
            print(f"\n{'='*50}")
            print("SAMPLE PREDICTIONS")
            print(f"{'='*50}")

            n_samples = min(10, len(X_test))
            print(f"{'Page':<6} {'Actual':<8} {'Predicted':<10} {'Confidence':<12} {'Correct':<8}")
            print("-" * 50)

            for i in range(n_samples):
                page_num = page_test[i] if page_numbers is not None else f"Test_{i+1}"
                actual = int(y_test[i])
                predicted = int(y_pred[i])
                confidence = float(y_pred_prob[i][0])
                correct = "✓" if actual == predicted else "✗"

                print(f"{page_num:<6} {actual:<8} {predicted:<10} {confidence:<12.4f} {correct:<8}")

        # Generate visualizations
        print(f"\n{'='*50}")
        print("GENERATING VISUALIZATIONS")
        print(f"{'='*50}")
        
        # Training metrics plot
        self.plot_training_metrics(history)
        
        # Confusion matrix plot
        self.plot_confusion_matrix(y_test, y_pred)
        
        # Save training summary
        self.save_training_summary(history, {
            'test_accuracy': test_accuracy,
            'test_precision': test_precision, 
            'test_recall': test_recall,
            'test_f1': f1_score
        })

        return history, X_test, y_test, y_pred, y_pred_prob

    def save_training_summary(self, history, test_metrics, output_file="training_summary.txt"):
        """
        Save comprehensive training summary to file.
        
        Args:
            history: Keras training history
            test_metrics: Dictionary of test metrics
            output_file: Output file path
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(output_file, 'w') as f:
            f.write("PDF FIRST PAGE CLASSIFICATION - TRAINING SUMMARY\n")
            f.write("="*60 + "\n")
            f.write(f"Training completed: {timestamp}\n\n")
            
            f.write("MODEL CONFIGURATION:\n")
            f.write(f"Image Size: {self.image_size}\n")
            f.write(f"Architecture: Transfer Learning (MobileNetV2/ResNet50)\n")
            f.write(f"Total Epochs: {len(history.history['accuracy'])}\n")
            f.write(f"Loss Function: Focal Loss (α=0.75, γ=2.0)\n")
            f.write(f"Optimizer: Adam (lr=0.0001)\n\n")
            
            f.write("TRAINING RESULTS:\n")
            f.write(f"Best Training Accuracy: {max(history.history['accuracy']):.4f}\n")
            f.write(f"Best Validation Accuracy: {max(history.history['val_accuracy']):.4f}\n")
            f.write(f"Final Training Loss: {history.history['loss'][-1]:.4f}\n")
            f.write(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}\n\n")
            
            f.write("TEST SET PERFORMANCE:\n")
            f.write(f"Test Accuracy: {test_metrics['test_accuracy']:.4f}\n")
            f.write(f"Test Precision: {test_metrics['test_precision']:.4f}\n")
            f.write(f"Test Recall: {test_metrics['test_recall']:.4f}\n")
            f.write(f"Test F1-Score: {test_metrics['test_f1']:.4f}\n\n")
            
            f.write("OBSERVATIONS:\n")
            f.write("- Transfer learning with pre-trained CNN models\n")
            f.write("- Focal loss used to handle class imbalance\n") 
            f.write("- Data augmentation applied to training data\n")
            f.write("- Model saved as 'best_first_page_classifier.h5'\n")

        print(f"Training summary saved to {output_file}")

# Main execution
if __name__ == "__main__":
    print("PDF First Page Classification - Basic Training")
    print("=" * 60)
    
    # Configuration
    IMAGE_SIZE = (224, 224)  # Standard size for transfer learning
    TEST_SIZE = 0.15         # 15% for testing
    VALIDATION_SPLIT = 0.15  # 15% of training data for validation
    EPOCHS = 100             # Maximum training epochs
    BATCH_SIZE = 8           # Small batch size for stability
    RANDOM_STATE = 42        # For reproducible results
    
    # File paths
    PDF_PATH = "Sample Invoices.pdf"
    EXCEL_PATH = "Correct values sample invoices.xlsx"
    
    try:
        # Initialize classifier
        classifier = PDFPageClassifier(image_size=IMAGE_SIZE)
        
        # Load and prepare data
        print("Loading data...")
        X, y, page_numbers = classifier.load_and_prepare_data(PDF_PATH, EXCEL_PATH)
        
        # Train model
        history, X_test, y_test, y_pred, y_pred_prob = classifier.train_model(
            X, y, page_numbers,
            test_size=TEST_SIZE,
            validation_split=VALIDATION_SPLIT,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            stratify=True,
            random_state=RANDOM_STATE
        )
        
        print(f"\n🎉 Training completed successfully!")
        print("Generated files:")
        print("- best_first_page_classifier.h5 (trained model)")
        print("- training_metrics.png (training curves)")
        print("- confusion_matrix.png (performance visualization)")
        print("- training_summary.txt (detailed results)")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check file paths for PDF and Excel files")
        print("2. Ensure all dependencies are installed")
        print("3. Verify sufficient disk space and memory")
        print("4. Check PDF extraction requirements (poppler-utils)")