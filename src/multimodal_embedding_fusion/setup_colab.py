from google.colab import drive
import os

def setup_colab():
    drive.mount('/content/drive')
    
    if not os.path.exists('Nepali_Multi_Modal'):
        !git clone https://github.com/Anil-Banjade/Nepali_Multi_Modal.git
    
    os.chdir('Nepali_Multi_Modal')

    if not os.path.exists('flickr8k-nepali-dataset'):
        !kaggle datasets download -d bipeshrajsubedi/flickr8k-nepali-dataset
        !unzip flickr8k-nepali-dataset.zip
    
    !pip install -r requirements.txt

if __name__ == "__main__":
    setup_colab()


# from setup_colab import setup_colab
# setup_colab()

# from main import main
# main()