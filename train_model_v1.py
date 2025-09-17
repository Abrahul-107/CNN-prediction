import os
import pandas as pd
import numpy as np
from pdf2image import convert_from_path
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class PDFPageClassifier:
    def __init__(self, image_size=(224, 224)):  # Changed back to 224
        self.image_size = image_size
        self.model = None

    def extract_pages_from_pdf(self, pdf_path, output_dir="extracted_pages"):
        """Extract pages from PDF as images"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("Extracting pages from PDF...")
        pages = convert_from_path(pdf_path, dpi=200)  # Reduced DPI for speed

        page_paths = []
        for i, page in enumerate(pages, 1):
            page_path = os.path.join(output_dir, f"page_{i:03d}.jpg")
            page.save(page_path, 'JPEG', quality=90)  # Added quality setting
            page_paths.append(page_path)

        print(f"Extracted {len(pages)} pages to {output_dir}")
        return page_paths

    def clean_annotated_image(self, image):
        """Remove yellow highlights and colored annotations"""
        # Convert to different color spaces for better detection
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Create mask for yellow highlights
        # Yellow in HSV: H=60±30, S=100-255, V=100-255
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([90, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Create mask for green circles (light green)
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Create mask for blue numbers
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Combine all annotation masks
        annotation_mask = cv2.bitwise_or(yellow_mask, green_mask)
        annotation_mask = cv2.bitwise_or(annotation_mask, blue_mask)
        
        # Dilate mask to capture annotation boundaries
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        annotation_mask = cv2.dilate(annotation_mask, kernel, iterations=1)
        
        # Replace annotated areas with white background
        cleaned_image = image.copy()
        cleaned_image[annotation_mask > 0] = [255, 255, 255]  # White background
        
        return cleaned_image

    def preprocess_image(self, image_path):
        """Enhanced preprocessing with annotation removal"""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # CLEAN ANNOTATIONS FIRST
        image = self.clean_annotated_image(image)
        
        # Then resize and normalize
        image = cv2.resize(image, self.image_size)
        image = image.astype('float32') / 255.0
        return image

    
    def create_data_generator(self):
        """Create OPTIMIZED data augmentation for documents"""
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        return ImageDataGenerator(
            rotation_range=1,              # REDUCED - documents shouldn't rotate much
            width_shift_range=0.02,        # REDUCED - smaller shifts
            height_shift_range=0.02,       # REDUCED - smaller shifts
            zoom_range=0.03,              # REDUCED - slight zoom only
            brightness_range=[0.9, 1.1],   # REDUCED - subtle brightness
            fill_mode='constant',          # Changed to constant
            horizontal_flip=False          # Never flip documents
        )

    def load_and_prepare_data(self, pdf_path, ground_truth_excel):
        """Load PDF pages and ground truth data"""
        # Extract pages from PDF
        page_paths = self.extract_pages_from_pdf(pdf_path)

        # Load ground truth
        df = pd.read_excel(ground_truth_excel)
        print("Ground truth data shape:", df.shape)
        print("Ground truth columns:", df.columns.tolist())
        print("\nFirst few rows:")
        print(df.head())

        # Prepare image data
        images = []
        labels = []
        page_numbers = []

        print("\nLoading and preprocessing images...")
        for idx, row in df.iterrows():
            page_num = int(row['Seitennummer'])
            label = int(row['Probability of first page'])

            if page_num <= len(page_paths):
                image_path = page_paths[page_num - 1]  # Convert to 0-based index
                try:
                    image = self.preprocess_image(image_path)
                    images.append(image)
                    labels.append(label)
                    page_numbers.append(page_num)
                except Exception as e:
                    print(f"Error processing page {page_num}: {e}")
                    continue
            else:
                print(f"Warning: Page {page_num} not found in PDF (PDF has {len(page_paths)} pages)")

        print(f"\nSuccessfully loaded {len(images)} pages")
        print(f"Class distribution: {np.bincount(labels)}")
        print(f"Class 0 (not first page): {np.sum(np.array(labels) == 0)} pages")
        print(f"Class 1 (first page): {np.sum(np.array(labels) == 1)} pages")

        return np.array(images), np.array(labels), np.array(page_numbers)

    def create_transfer_cnn_model(self):
        """Transfer learning model - most effective for your case"""
        try:
            from tensorflow.keras.applications import MobileNetV2
            
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.image_size, 3)
            )
        except Exception as e:
            print(f"MobileNetV2 download failed: {e}")
            print("Using ResNet50 instead...")
            from tensorflow.keras.applications import ResNet50
            
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.image_size, 3)
            )
        
        # Fine-tune last layers
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
    
    def extreme_focal_loss(self, alpha=0.9, gamma=4.0):
        """Fixed focal loss for TensorFlow 2.x"""
        def focal_loss_fixed(y_true, y_pred):
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
            
            alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
            p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
            
            # FIX: Use tf.math.log instead of tf.log
            focal_loss = -alpha_t * tf.pow((1 - p_t), gamma) * tf.math.log(p_t)
            
            return tf.reduce_mean(focal_loss)
        return focal_loss_fixed

    def train_model(self, X, y, page_numbers=None, test_size=0.2, validation_split=0.2, 
                   epochs=50, batch_size=16, stratify=True, random_state=42):
        """FIXED train_model with proper validation"""
        
        total_samples = len(X)
        print(f"\nTotal samples: {total_samples}")
        print(f"Using {int((1-test_size)*100)}/{int(test_size*100)} train/test split")

        # Stratified train/test split
        if stratify and len(np.unique(y)) > 1:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                stratify=y,
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

        # PROPER TRAIN/VALIDATION SPLIT
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

        # Class weights
        from sklearn.utils.class_weight import compute_class_weight
        
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(y_train_final), 
            y=y_train_final
        )
        class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
        
        print(f"Class weights: {class_weights_dict}")
        
        # USE TRANSFER LEARNING MODEL
        self.model = self.create_transfer_cnn_model()
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),  # INCREASED from 0.0001
            loss=self.extreme_focal_loss(alpha=0.95, gamma=5.0),
            metrics=['accuracy', 'precision', 'recall']
        )

        print("\nModel Architecture:")
        self.model.summary()

        # Setup callbacks
        model_checkpoint = ModelCheckpoint(
            'best_first_page_classifier.h5',
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        )

        early_stopping = EarlyStopping(
            monitor='val_accuracy',  # Changed to val_accuracy
            patience=15,             # Increased patience
            restore_best_weights=True,
            verbose=1,
            mode='max'
        )

        # Reduce learning rate callback
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1
        )

        # CREATE DATA GENERATOR
        datagen = self.create_data_generator()
        
        # FIXED validation setup
        val_datagen = keras.preprocessing.image.ImageDataGenerator()  # No augmentation for validation
        
        # Train with PROPER validation
        print(f"\nStarting training with {epochs} epochs and data augmentation...")
        history = self.model.fit(
            datagen.flow(X_train_final, y_train_final, batch_size=batch_size),
            steps_per_epoch=max(1, len(X_train_final) // batch_size * 2),  # INCREASED steps
            epochs=epochs,
            validation_data=val_datagen.flow(X_val, y_val, batch_size=batch_size),  # PROPER validation
            validation_steps=max(1, len(X_val) // batch_size),
            class_weight=class_weights_dict,
            callbacks=[model_checkpoint, early_stopping, reduce_lr],
            verbose=1
        )

        # Rest of your evaluation code remains the same...
        # Evaluate on test set
        print("\nEvaluating model on test set...")
        test_loss, test_accuracy, test_precision, test_recall = self.model.evaluate(X_test, y_test, verbose=0)

        print(f"\n{'='*50}")
        print(f"FINAL TEST RESULTS")
        print(f"{'='*50}")
        print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")

        # Calculate F1 score
        if test_precision + test_recall > 0:
            f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall)
        else:
            f1_score = 0.0
        print(f"Test F1-Score: {f1_score:.4f}")

        # Detailed evaluation
        print("\nGenerating detailed predictions...")
        y_pred_prob = self.model.predict(X_test, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()

        print("\n" + "="*50)
        print("CLASSIFICATION REPORT")
        print("="*50)
        print(classification_report(y_test, y_pred, target_names=['Not First Page', 'First Page']))

        print("\n" + "="*50)
        print("CONFUSION MATRIX")
        print("="*50)
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        print("\nConfusion Matrix Interpretation:")
        print(f"True Negatives (correctly identified non-first pages): {cm[0,0]}")
        print(f"False Positives (incorrectly identified as first pages): {cm[0,1]}")
        print(f"False Negatives (missed first pages): {cm[1,0]}")
        print(f"True Positives (correctly identified first pages): {cm[1,1]}")

        # Show sample predictions
        if page_numbers is not None:
            print("\n" + "="*50)
            print("SAMPLE PREDICTIONS ON TEST SET")
            print("="*50)

            n_samples = min(10, len(X_test))
            sample_indices = np.arange(n_samples)

            print(f"{'Page':<6} {'Actual':<8} {'Predicted':<10} {'Confidence':<12} {'Correct':<8}")
            print("-" * 50)

            for i in sample_indices:
                page_num = page_test[i] if page_numbers is not None else f"Test_{i+1}"
                actual = int(y_test[i])
                predicted = int(y_pred[i])
                confidence = float(y_pred_prob[i][0])
                correct = "✓" if actual == predicted else "✗"

                print(f"{page_num:<6} {actual:<8} {predicted:<10} {confidence:<12.4f} {correct:<8}")

        return history, X_test, y_test, y_pred, y_pred_prob

    # Your other methods remain the same...

# USAGE
if __name__ == "__main__":
    classifier = PDFPageClassifier(image_size=(224, 224))
    
    X, y, page_numbers = classifier.load_and_prepare_data("Sample Invoices.pdf", "Correct values sample invoices.xlsx")
    
    # Train with optimized parameters for 75%+ accuracy
    history, X_test, y_test, y_pred, y_pred_prob = classifier.train_model(
        X, y, page_numbers,
        test_size=0.15,        # Use more training data
        validation_split=0.15, # Smaller validation
        epochs=150,            # More epochs
        batch_size=8,          # Smaller batch for stability
        stratify=True,
        random_state=42
    )
