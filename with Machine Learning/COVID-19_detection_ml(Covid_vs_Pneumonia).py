import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage.feature import hog, greycomatrix, greycoprops
import cv2, pathlib
from sklearn import svm, tree, ensemble, linear_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix,roc_curve

files_path = 'resimler/COVID_DATASET/'
files_path = pathlib.Path(files_path)
images_dict = {
    'covid': list(files_path.glob('COVID/*')),
    'pneumonia': list(files_path.glob('PNEUMONIA/*')),

}


labels_dict = {
    'covid': 0,
    'pneumonia': 1,
}

IMAGE_SHAPE = (224, 224)
X, y = [], []
for name, images in images_dict.items():
    for image in images:
        img = cv2.imread(str(image), 1)
        resized_img = cv2.resize(img, dsize=IMAGE_SHAPE)
        blurred_img = cv2.GaussianBlur(resized_img, ksize=(3, 3), sigmaX=0.5, sigmaY=0.7,
                                       borderType=cv2.BORDER_CONSTANT)
        hsv_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2HSV)
        h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
        value = 10
        limit = 255 - value
        v[v > limit] = 255
        v[v <= limit] += value
        hsv_img_new = cv2.merge((h, s, v))
        img_brightness = cv2.cvtColor(hsv_img_new, cv2.COLOR_HSV2RGB)
        img_brightness_gray = cv2.cvtColor(img_brightness, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        img_hist_gray = clahe.apply(img_brightness_gray)
        X.append(img_hist_gray)
        y.append(labels_dict[name])

dataset = pd.DataFrame()
for image in X:
    df = pd.DataFrame()
    i = 0
    glcm_features = ['energy','correlation','homogeneity','contrast']
    for distance in np.arange(1,6,2):
        for angle in np.arange(0,(3*np.pi/4)+0.1,np.pi/4):
            i += 1
            for feature in glcm_features:
                GLCM = greycomatrix(image,[distance],[angle])
                df['GLCM_{}_{}'.format(feature,i)] = greycoprops(GLCM, prop=feature)[0]

    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), visualize=True)
    df['HOG_mean'] = np.mean(fd)
    df['HOG_std'] = np.std(fd)

    ksize = 5
    psi = 0
    j = 0
    for theta in np.arange(np.pi / 4, 2 * np.pi, np.pi / 4):
        for sigma in (1, 3, 5, 7):
            for lamda in (np.pi / 4, np.pi, np.pi / 4):
                for gamma in (0.5, 0.9):
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, psi)
                    fimg = cv2.filter2D(image, cv2.CV_8UC3, kernel)
                    fimg_mean = np.mean(fimg)
                    df['Gabor_{}'.format(j+1)] = fimg_mean
                    j += 1


    dataset = dataset.append(df)

scaler = MinMaxScaler()
scaler.fit(dataset)
dataset = scaler.transform(dataset)

X_train, X_test, y_train, y_test = train_test_split(dataset,y,test_size=0.2,random_state=42)

model_params = {
    'Support Vector Machines': {
        'model':svm.SVC(gamma='auto'),
        'params': {
            'C':[1,10,20,30],
            'kernel':['linear','rbf']
        }
    },
    'Random Forest': {
        'model':ensemble.RandomForestClassifier(),
        'params': {
            'n_estimators': [10,30,50,100],
            'criterion': ['gini','entropy']
        }
    },
    'Decision Tree': {
        'model': tree.DecisionTreeClassifier(),
        'params': {
            'criterion': ['gini', 'entropy']
        }
    },
    'Logistic Regression': {
        'model': linear_model.LogisticRegression(solver='liblinear'),
        'params': {
            'C': [1, 10, 20, 30]
        }
    }
}

# 115-163 satırları arası tablo oluşturmak için.

df_results = pd.DataFrame()
best_scores = []
for model_name, mp in model_params.items():
    clf = GridSearchCV(mp['model'],mp['params'],cv=5)
    clf.fit(X_train,y_train)
    best_scores.append({
        'model':clf.best_estimator_,
        'best_score':clf.best_score_,
        'best_params':clf.best_params_
    })
    clf_result = pd.DataFrame(clf.cv_results_)
    clf_result['Model Adı'] = model_name
    df_results = df_results.append(clf_result)

df_best_results = pd.DataFrame(best_scores)


df_final_results = pd.DataFrame()
for model_name in model_params:
    df_results_model = df_results[df_results['Model Adı'] == model_name]
    label_names = ['Covid_Precision', 'Covid_Recall', 'Covid_F1-score',
                   'Pneumonia_Precision', 'Pneumonia_Recall', 'Pneumonia_F1-score']

    test_scores = []
    cr_s = {}
    m = 0

    if model_name == 'Support Vector Machines':
        for i,j in zip(df_results_model['param_C'],df_results_model['param_kernel']):
            model = svm.SVC(C=i,kernel=j,gamma='auto')
            model.fit(X_train,y_train)
            score = model.score(X_test, y_test)
            test_scores.append(score)
            y_pred = model.predict(X_test)
            cr = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            cr_s[str(m)] = np.array(list(
                map(lambda k: [cr[str(k)]['precision'], cr[str(k)]['recall'], cr[str(k)]['f1-score']],
                    range(0, 2)))).flatten()
            m += 1

        df_results_model['Test Doğruluk'] = test_scores

    elif model_name == 'Random Forest':
        for i, j in zip(df_results_model['param_n_estimators'], df_results_model['param_criterion']):
            model = ensemble.RandomForestClassifier(n_estimators=i,criterion=j)
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            test_scores.append(score)
            y_pred = model.predict(X_test)
            cr = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            cr_s[str(m)] = np.array(list(
                map(lambda k: [cr[str(k)]['precision'], cr[str(k)]['recall'], cr[str(k)]['f1-score']],
                    range(0, 2)))).flatten()
            m += 1

        df_results_model['Test Doğruluk'] = test_scores

    elif model_name == 'Decision Tree':
        for i in (df_results_model['param_criterion']):
            model = tree.DecisionTreeClassifier(criterion=i)
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            test_scores.append(score)
            y_pred = model.predict(X_test)
            cr = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            cr_s[str(m)] = np.array(list(
                map(lambda k: [cr[str(k)]['precision'], cr[str(k)]['recall'], cr[str(k)]['f1-score']],
                    range(0, 2)))).flatten()
            m += 1

        df_results_model['Test Doğruluk'] = test_scores

    else:
        for i in (df_results_model['param_C']):
            model = linear_model.LogisticRegression(C=i,solver='liblinear')
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            test_scores.append(score)
            y_pred = model.predict(X_test)
            cr = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            cr_s[str(m)] = np.array(list(map(lambda k: [cr[str(k)]['precision'], cr[str(k)]['recall'], cr[str(k)]['f1-score']],range(0, 2)))).flatten()
            m += 1

        df_results_model['Test Doğruluk'] = test_scores

    for i in range(len(label_names)):
        column_vals = []
        for j in range(df_results_model.shape[0]):
            column_val = cr_s[str(j)][i]
            column_vals.append(column_val)
        df_results_model[label_names[i]] = column_vals

    df_final_results = df_final_results.append(df_results_model)

df_final_results2 = df_final_results.loc[:,'Model Adı':'Pneumonia_F1-score'].drop(['param_criterion','param_n_estimators'],axis='columns')
df_final_results2.insert(1,column='Parametreler',value=df_final_results['params'])
df_final_results2.insert(2,column='Eğitim Doğruluk',value=df_final_results['mean_test_score'])
df_final_results2.to_excel('Covid_vs_Pneumonia.xlsx')

best_model_df = df_final_results2[df_final_results2['Test Doğruluk'] == np.max(df_final_results2['Test Doğruluk'])]

best_model = ensemble.RandomForestClassifier(criterion='entropy',n_estimators=50)
best_model.fit(X_train,y_train)
y_pred_best = best_model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test,y_pred_best)
df_cm = pd.DataFrame(cm,index=['Covid-19','Zatürre'],columns=['Covid-19','Zatürre'])
import seaborn as sns
sns.heatmap(df_cm, annot=True, fmt='d')
plt.title('Karışıklık Matrisi', fontsize=20)
plt.xlabel('Tahmin', fontsize=15)
plt.ylabel('Gerçek', fontsize=15)
plt.tight_layout()
plt.show()


fpr,tpr,_ = roc_curve(y_test,y_pred_best)
plt.figure(figsize=(16,9))
plt.plot(fpr,tpr)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title('ROC Eğrisi', fontsize=20)
plt.xlabel('Yalancı Pozitif Oranı', fontsize=15)
plt.ylabel('Doğru Pozitif Oranı', fontsize=15)
plt.show()