Personality Prediction with Machine Learning
This project aims to predict personality types based on text data using machine learning techniques. It utilizes the Myers-Briggs Type Indicator (MBTI) dataset, which consists of posts from various social media platforms.

Installation
Clone the repository to your local machine:
git clone <repository_url>
Navigate to the project directory:
cd personality-prediction
Install the required Python packages:
pip install -r requirements.txt

Deployment
After installing the required packages, ensure that you have the necessary data files. You can unzip the included dataset.
Place the downloaded dataset (mbti_1.csv) in the project directory.
Update the file paths in the Python script (predict_personality.py) to point to the correct location of the dataset.
Run the Python script to train the machine learning model and start the GUI:
python MBTIpred.py
Once the GUI is running, you can enter text in the provided entry field and click "Submit" to predict the personality type. The predicted MBTI type and confidence level will be displayed.

Notes
Ensure that you have Python installed on your system before running the application.
The project uses libraries such as scikit-learn, NLTK, and appJar. These dependencies are specified in the requirements.txt file and will be installed during setup.
