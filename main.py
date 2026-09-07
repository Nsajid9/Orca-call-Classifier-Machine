import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns

class OrcaNet:
    @staticmethod
    def extract_features(image_path, vector_size=32):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            return np.zeros(vector_size * 64)
        image = cv2.resize(image, (224,224))
        try:
            alg = cv2.KAZE_create()
            kps = alg.detect(image)
            
            # Handling case with no keypoints
            if not kps:
                return np.zeros(vector_size * 64)
                
            kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
            kps, dsc = alg.compute(image, kps)
            
            if dsc is None:
                return np.zeros(vector_size * 64)
                
            dsc = dsc.flatten()
            needed_size = (vector_size * 64)
            if dsc.size < needed_size:
                dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
        except cv2.error as e:
            print('Error: ', e)
            return np.zeros(vector_size * 64)
            
        return dsc

    @staticmethod
    def classifier(n_estimators=100, random_state=42):
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        return rf

if __name__ == "__main__":
    positive = []
    negative = []
    num_calls = 0
    num_no_calls = 0

    train_dir = "Spectrograms/train/"
    class_names = ['call', 'no_call']

    print("Extracting features from spectrograms...")
    for c in class_names:
        class_dir = os.path.join(train_dir, c)
        if not os.path.exists(class_dir):
            print(f"Directory {class_dir} does not exist.")
            continue
            
        images = [x for x in os.listdir(class_dir) if x.lower().endswith('png')]
        for image in images:
            img_path = os.path.join(class_dir, image)
            feat_vector = OrcaNet.extract_features(img_path, 64)
            if c == 'call':
                positive.append(feat_vector)
                num_calls += 1
            else:
                negative.append(feat_vector)
                num_no_calls += 1 

    print(f"Processed {num_calls} call images and {num_no_calls} no_call images.")
    
    if num_calls + num_no_calls == 0:
        print("No images found. Ensure you ran generate_dataset.py first.")
        exit(1)

    x = np.concatenate((positive, negative), axis=0)
    z = np.zeros(num_calls)
    o = np.ones(num_no_calls)
    y = np.concatenate((z, o), axis=0).reshape(x.shape[0], 1)

    print("x shape:", x.shape)
    print("y shape:", y.shape)

    # Train/Test Split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=42)
    
    print("Training Random Forest Classifier...")
    model = OrcaNet.classifier()
    model.fit(x_train, y_train.ravel())
    
    score = model.score(x_test, y_test.ravel())
    print("Random Forest Classification Test Score:", score)

    # Predictions and Confusion Matrix
    print("Evaluating and creating confusion matrix...")
    y_prediction = model.predict(x_test)
    y_true = y_test.ravel()

    cm = confusion_matrix(y_true, y_prediction)

    f, ax = plt.subplots(figsize=(6,6))
    sns.heatmap(cm, annot=True, linewidths=0.4, linecolor="red", fmt=".0f", ax=ax)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")
    
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    print(f"Saved confusion matrix to {cm_path}")
    print("Done!")
