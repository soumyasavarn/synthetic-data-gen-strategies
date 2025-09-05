# synthetic-data-gen-strategies
<<<<<<< Updated upstream
Built CTGAN for tabular data and Pix2Pix for image-to-comic translation in PyTorch as part of the Advanced ML Lab (Jan–Apr 2025).

Improved Iris dataset accuracy by 3.33% through data augmentation and deployed a Streamlit app for real-time synthesis.

=======
This is a course project under DA312 Advanced ML Laboratory.
Walkthrough:
The file `app.py` is a frontend interface using streamlit which provides us options to input .csv or .jpg/.png type of files. It detects automatically whether we want to generate image or tabular data. 
Case 1: If we are trying to generate tabular data we will get two choose between vanilla GAN and CTGAN and no. of epoch and how many samples we want to generate. The model will be trained on the fly. (Note: In this case no pre-trained model is used)
Case 2: If we are trying to generate image then I am using a pre-trained model `generator_120.pth`(which was trained on Kaggle on dataset: https://www.kaggle.com/datasets/defileroff/comic-faces-paired-synthetic-v2 on GPU P100 for approx 10 hours). 120 is the number of epochs.

Running the app:
First, you need to install dependencies using `pip install -r requirements.txt`.
Then run the `app.py` using `python -m streamlit run app.py`.

Note: `testing_data` folder contains some testing data (image as well as csv). Also, it is highly recommended for image generation to use this data only as GANs could not generalise well in this limited training environment. They are prone to overfit on the domain.

TL;DR: Built CTGAN for tabular data and Pix2Pix for image-to-comic translation in PyTorch as part of the Advanced ML Lab (Jan–Apr 2025). Improved Iris dataset accuracy by 3.33% through data augmentation and deployed a Streamlit app for real-time synthesis.
>>>>>>> Stashed changes
Report: https://drive.google.com/file/d/1Y6Hx2FJsqoxIJaOXLrbAghCdvE5GtLds/view
