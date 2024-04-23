
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import local_binary_pattern # Zik Addition
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

defective_folder = r"C:\Users\Nkosie\Downloads\Academic Weapons\Y3\B1\MLA\Assi\CONCRETE CRACKS\Defective"
defectless_folder = r"C:\Users\Nkosie\Downloads\Academic Weapons\Y3\B1\MLA\Assi\CONCRETE CRACKS\Defectless"
random_folder = r"C:\Users\Nkosie\Downloads\Academic Weapons\Y3\B1\MLA\Assi\CONCRETE CRACKS\Random" # Zik Addition

# Feature Extraction methods! ____________________
def canny_feature_extraction(image, threshold1=100, threshold2=200):
    # Apply Canny edge detection
    edges = cv2.Canny(image, threshold1=threshold1, threshold2=threshold2)

    # Flatten the edge image to obtain a feature vector
    feature_vector = edges.flatten()

    return feature_vector

def hog_feature_extraction(image):
    # Initialize HOG descriptor
    hog = cv2.HOGDescriptor()

    # Compute HOG features
    hog_features = hog.compute(image)

    # Flatten the feature vector
    feature_vector = hog_features.flatten()

    return feature_vector

def lbp_feature_extraction(image, radius=1, n_points=8):
    # Compute LBP features
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')

    # Flatten the LBP image to obtain a feature vector
    feature_vector = lbp.ravel()

    return feature_vector
#---------------------------------------------------

def contrast_stretching(image):
    # Calculate minimum and maximum pixel values
    min_val = np.min(image)
    max_val = np.max(image)

    # Perform contrast stretching
    stretched_image = ((image - min_val) / (max_val - min_val)) * 255
    stretched_image = stretched_image.astype(np.uint8)

    return stretched_image


def image_enhancement(image):
    # Contrast improvement
    image_eq = cv2.equalizeHist(image)
    # image_eq = contrast_stretching(image)

    # denoising
    gaussian_blur = cv2.GaussianBlur(image_eq, (5, 5), 0)
    # image_blur =  bilateral_filter(image_eq, 9, 75, 75)
    # median_blur = cv2.medianBlur(image, kernel_size)
    # mean_blur = cv2.blur(image, (kernel_size, kernel_size))
    # filtered_image = cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    return gaussian_blur

# The Magic
def load_and_preprocess_images(folder, label):
    images = []
    labels = []
    features = []
    c = 0
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale
        c = c + 1
        if img is not None:
            resized_img = cv2.resize(img, (200, 200))
            enh_img = image_enhancement(resized_img)   # Enhancing Resized image
            fet_vec = hog_feature_extraction(enh_img)  # Extracting Feature Vector from Enhanced image
            
            images.append(enh_img)
            labels.append(label)
            features.append(fet_vec)
    # print(c)
    return images, labels, features


def display_sidebyside(defect_no, defectless_number):
    # assign images
    defectless_img = defectless_images[defectless_number]
    defective_img = defective_images[defect_no]

    # assign labels
    defectless_label = defectless_labels[defectless_number]
    defective_label = defective_labels[defect_no]

    # Create a figure and axis objects
    fig, axes = plt.subplots(1, 2, figsize=(5, 5))

    # Display the first image with its label
    axes[0].imshow(defective_img, cmap='gray')
    axes[0].set_title(defective_label)
    axes[0].axis('off')

    # Display the second image with its label
    axes[1].imshow(defectless_img, cmap='gray')
    axes[1].set_title(defectless_label)
    axes[1].axis('off')

    # Adjust layout and display the images
    plt.tight_layout()
    plt.show()
    
def display_random_w_prediction(Model, random_no, prediction): # Zik Addition
    # assign images
    random_img = random_images[random_no]

    # assign labels

    if prediction == 0 :
        random_lbl = ' Prediction: Defectless'
        
    elif prediction == 1 :
        random_lbl = ' Prediction: Defective'
        
    else:
        random_lbl = ': No Prediction'


    # Create a figure and axis objects
    fig, axes = plt.subplots(1, 2, figsize=(5, 5))

    # Display the first image with its label
    axes[0].imshow(random_img, cmap='gray')
    axes[0].set_title(Model + random_lbl)
    axes[0].axis('off')

    # Adjust layout and display the images
    plt.tight_layout()
    plt.show()


defective_images, defective_labels, defective_features = load_and_preprocess_images(defective_folder, 1)
defectless_images, defectless_labels, defectless_features = load_and_preprocess_images(defectless_folder, 0)
random_images, random_labels, random_features = load_and_preprocess_images(random_folder, 2)  # Zik Addition

# display_sidebyside(2,2)

# Combine data.
X = np.array(defective_features + defectless_features)
y = defective_labels + defectless_labels
Z = random_features

# Normalize features.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Z_scaled = scaler.fit_transform(Z)  # Zik Addition

# print(X_scaled.shape)

# Split dataset into training and testing.
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Create and train the MLPClassifier model ------------------------
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', max_iter=500, random_state=42)
mlp.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = mlp.predict(X_test)

print("MP Classifier Metrics:")

# Calculate Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Initialize & train Random Forest classifier ---------------------
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\nRandom Forest Classifier Metrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Initialize & Train the SVM classifier --------------------------------------------------
svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(X_train, y_train)

# Make predictions
y_pred = svm_classifier.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\nSVM Classifier Metrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Zik Additions
# Use Predictors 
# print(Z_scaled.shape)

random_resultMlp = mlp.predict(Z_scaled)
random_resultRF = rf_classifier.predict(Z_scaled)
random_resultSVM = svm_classifier.predict(Z_scaled)

# 0 - 9 in code. 1 - 10 in folder cause Arry Index
n = 3
display_random_w_prediction('MLP', n, random_resultMlp[n])
display_random_w_prediction('RF', n, random_resultRF[n])
display_random_w_prediction('SVM', n, random_resultSVM[n])