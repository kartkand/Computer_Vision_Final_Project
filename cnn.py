import pandas as pd
import pickle
import os
import shutil
import cv2
from PIL import Image, ImageEnhance 

def main():
    # Delete all augmented images from previous run
    imageNames = os.listdir("train")
    for imageName in imageNames:
        if len(imageName) <= 12:
            print(imageName)
            os.remove("train/"+imageName)

    # If we don't already having an image to label mapping, make one
    if os.path.isfile('imageToLabel.pickle') == False:
        # Go through and find out which images have cancer and which ones don't
        df = pd.read_csv('train_labels.csv')
        imageToLabel = {}
        for index, row in df.iterrows():
            imageToLabel[row['id']+".tif"] = row['label']
        
        #Save the mapping of image to label using pickle
        with open('imageToLabel.pickle','wb') as f:
            pickle.dump(imageToLabel,f)
    else:
        with open('imageToLabel.pickle','rb') as f:
            imageToLabel = pickle.load(f)
    
    # We count how many of the images have cancer spread and how many don't
    # From this, we see that there is an imbalance in the training data.
    # We ideally want the ratio to be as close to 1.  We'll fix this next
    cancerPresentCount = 0
    cancerNotPresentCount = 0
    for image in imageToLabel:
        if imageToLabel[image] == 0:
            cancerNotPresentCount += 1
        else:
            cancerPresentCount += 1
    print(cancerPresentCount,cancerNotPresentCount)

    # Create new images that contain cancer
    index = 0
    if cancerPresentCount/cancerNotPresentCount < 1:
        imageNames = os.listdir("train")
        for imageName in imageNames:
            if imageToLabel[imageName] == 1:
                path = "train/" + imageName
                img = Image.open(path)
                enhancer = ImageEnhance.Brightness(img)
                enhanced_im = enhancer.enhance(1.8)
                enhanced_im.save("train/"+str(index)+".tif")
                imageToLabel[str(index)+".tif"] = 1
                index = index + 1
                cancerPresentCount = cancerPresentCount + 1
                print(cancerPresentCount/cancerNotPresentCount)
            if cancerPresentCount/cancerNotPresentCount >= 1:
                break
    
    #By now, we have an even balance in the two classes for training

if __name__ == "__main__":
    main()
