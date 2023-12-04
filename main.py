import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def IsRetinopathy(grade):
    return 1 if grade>0 else 0
def get_pixel_values(image_path,resize_shape=(25, 25)):
    img = cv2.imread(image_path)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(gray_image, resize_shape)
    return resized_img.flatten()
df = pd.read_csv("./B. Disease Grading/2. Groundtruths/training.csv")
x_test_list =[]
test_df = pd.read_csv("./B. Disease Grading/2. Groundtruths/test.csv")
# print(df.columns )
# print(df.shape)
# print("test shape",test_df.shape)
# selected_data = df.iloc[0:, 3:]
# count_null = df["Unnamed: 11"].isnull().sum()
test_df=test_df.iloc[:,:2]
# print("testdfcol",test_df.columns)
df = df.iloc[:,:2]
# print(df.columns )
df["binary"]=df["Retinopathy grade"].apply(lambda x: IsRetinopathy(x))
test_df["binary"]=test_df["Retinopathy grade"].apply(lambda x: IsRetinopathy(x))




frequency_counts = df["Retinopathy grade"].value_counts()
test_frequency_counts = test_df["Retinopathy grade"].value_counts()

bin_freq = df["binary"].value_counts()
test_bin_freq = test_df["binary"].value_counts()
  

#creating empty data frame for storing index values
df_pixel_values = pd.DataFrame()
test_df_pixel_values = pd.DataFrame()
pixel_values_list =[]
test_pixel_values_list =[]
#creating separate df for images name
images_name = df['Image name']
test_images_name = test_df['Image name']

for image in images_name:
    print("train doing for ",image)
    image_path=f"./B. Disease Grading/1. Original Images/a. Training Set/{image}.jpg"
    pixel_values=get_pixel_values(image_path)
    pixel_values_list.append(pixel_values)
for image in test_images_name:
    print("test for image",image)
    image_path=f"./B. Disease Grading/1. Original Images/b. Testing Set/{image}.jpg"
    pixel_values=get_pixel_values(image_path)
    test_pixel_values_list.append(pixel_values)
df_pixel_values = pd.DataFrame(pixel_values_list)
test_df_pixel_values=pd.DataFrame(test_pixel_values_list)
print("this is shape")
print(df_pixel_values.shape)
print(test_df_pixel_values.shape)
print(df_pixel_values.head())
print(test_df_pixel_values.head())

X_train = df_pixel_values
y_train  = df["binary"]
x_test = test_df_pixel_values
y_test = test_df["binary"]




logistic_model = LogisticRegression(random_state=1)

logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

print("Logistic regression")
print(f'Accuracy: {accuracy}')
print('\nClassification Report:\n', classification_report_str)

decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train, y_train)
y_pred = decision_tree_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)


print("decision tree")
print(f'Accuracy: {accuracy}')

print('\nClassification Report:\n', classification_report_str)

random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)
y_pred = random_forest_model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

classification_report_str = classification_report(y_test, y_pred)


print("random forest")

print(f'Accuracy: {accuracy}')

print('\nClassification Report:\n', classification_report_str)