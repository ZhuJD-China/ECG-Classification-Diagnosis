# ECG-Classification-Diagnosis

## Introduction

​      Arrhythmia is a common cardiovascular syndrome. The correct identification of arrhythmia is of great significance for the prevention of cardiovascular disease. Electrocardiogram (ECG) is a kind of medical monitoring technology which reflects cardiac activity. The electrocardiogram was used to observe whether the ECG signal was abnormal or not, and whether there were abnormal cardiac beats, so as to prevent or diagnose cardiovascular disease in advance. In clinical examination, due to the influence of power frequency, EMG and other interference signals

1. wavelet transform algorithm is used to filter ECG signal.
	
   <div align="center">
	<img src="https://github.com/ZhuJD-China/ECG-Classification-Diagnosis/blob/master/ECG%E5%AE%9E%E9%AA%8C%E6%8A%A5%E5%91%8Av1.1/1.png" alt="Editor" width="100">
</div>

<div align="center">
	<img src="https://github.com/ZhuJD-China/ECG-Classification-Diagnosis/blob/master/ECG%E5%AE%9E%E9%AA%8C%E6%8A%A5%E5%91%8Av1.1/2.png" alt="Editor" width="100">
</div>
   
   The wavelet basis function is obtained by transforming the basic wavelet through scale factors a and b.
   
   <div align="center">
	<img src="https://github.com/ZhuJD-China/ECG-Classification-Diagnosis/blob/master/ECG%E5%AE%9E%E9%AA%8C%E6%8A%A5%E5%91%8Av1.1/3.png" alt="Editor" width="100">
</div>

​       According to its definition, continuous wavelet transform is essentially an integral wavelet transform：

   <div align="center">
	<img src="https://github.com/ZhuJD-China/ECG-Classification-Diagnosis/blob/master/ECG%E5%AE%9E%E9%AA%8C%E6%8A%A5%E5%91%8Av1.1/4.png" alt="Editor" width="100">
</div>

2. build a deep learning based arrhythmia diagnosis model.

## DataSet

​     MIT-BIH arrhythmia database is an MIT ECG database based on international standards, expert diagnosis and annotation, and it is also a standard ECG database widely recognized and used by the academic community. This database is an important data source for the research of automatic arrhythmia diagnosis algorithm in this paper.

https://www.physionet.org/content/mitdb/1.0.0/

## Arrhythmia analysis

<div align="center">
	<img src="https://github.com/ZhuJD-China/ECG-Classification-Diagnosis/blob/master/ECG%E5%AE%9E%E9%AA%8C%E6%8A%A5%E5%91%8Av1.1/%E4%B8%80%E4%B8%AA%E5%AE%8C%E6%95%B4%E7%9A%84%E5%BF%83%E7%94%B5%E6%B3%A2%E5%BD%A2.png" alt="Editor" width="500">
</div>

* P wave: P wave is usually considered to be the beginning part of the whole ECG cycle, which represents the specific situation of the potential change during the depolarization of the atrial muscle. The front and back represent right atrial activation and left atrial activation, respectively. The length of the interval is about 0.08s-0.11s, and the amplitude does not exceed 0.25mV.
* P-R interval: The PR interval represents the conduction time required for the excitement of the sinus node to cause the ventricular muscle to start to excite, so it is also called the atrioventricular conduction time. The length of the interval is generally 0.12s-0.2.s.
*  QRS complex: QRS complex is a wave group consisting of Q wave, R wave and S wave. Among them, Q wave is the first downward wave whose duration is generally not less than 0.04s, the high sharp protruding wave behind the Q wave is the R wave, and the subsequent downward wave is the S wave. The QRS complex graph contains relatively rich information, and this waveform is often used as the main tool and basis for distinguishing the type of arrhythmia.
*  T wave: T wave mainly represents the potential fluctuation changes during the repolarization of rapid ventricular motion. The waveform direction of the T wave is generally the same as the QRS complex, and the interval length is about 0.02s-0.25s.
*  QT interval: QT interval represents the whole process of ventricular muscle depolarization and ventricular muscle repolarization. The length of the interval is approximately 0.43s-0.44s.
*  U wave: U wave appears 0.02s-0.04s after T wave, with low amplitude. The direction of the U wave is generally the same as the direction of the T wave, and the amplitude is about half of the T wave.

## Model

![](https://github.com/ZhuJD-China/ECG-Classification-Diagnosis/blob/master/ECG%E5%AE%9E%E9%AA%8C%E6%8A%A5%E5%91%8Av1.1/471595_1_En_23_Fig4_HTML.png)



`val_loss: 0.0326 - val_accuracy: 0.9933`

![](https://github.com/ZhuJD-China/ECG-Classification-Diagnosis/blob/master/ECG%E5%AE%9E%E9%AA%8C%E6%8A%A5%E5%91%8Av1.1/cnn_lstm_acc.png)

![](https://github.com/ZhuJD-China/ECG-Classification-Diagnosis/blob/master/ECG%E5%AE%9E%E9%AA%8C%E6%8A%A5%E5%91%8Av1.1/cnn_lstm_loss.png)
## Common evaluation indicators for classification models are Accuracy, Precision, Recall and F1-score

$$
\text { Accuracy }=\frac{T P+T N}{T P+T N+F P+F N}
$$

$$
\text { Precision }=\frac{T P}{T P+F P}
$$

$$
\text { Recall }=\frac{T P}{T P+F N}
$$

$$
F 1-\text { score }=\frac{2 \times \text { Precision } \times \text { Recall }}{\text { Precision }+\text { Recall }}
$$

**Macro-average方法**
- This method is the simplest. It directly adds up the evaluation indicators of different categories (Precision/ Recall/ F1-score) and averages them, giving all categories the same weight. This method can treat each category equally, but its value will be affected by the rare category.

**Weighted-average方法**
- This method gives different weights to different categories (the weight is determined according to the true distribution ratio of the category), and each category is multiplied by the weight and then added. This method takes into account the imbalance of categories, and its value is more susceptible to the influence of the majority class.

**Micro-average方法**
- This method adds up the TP, FP, and FN of each category first, and then calculates it according to the two-category formula.


```python
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

Y_pred.shape
Y_test.shape
# 计算精确度
# correct_prediction = np.equal(Y_pred, Y_test)
# print(np.mean(correct_prediction))   <==> accuracy_score()

print('Accuracy:', accuracy_score(list(Y_pred), list(Y_test)))
# print('Precision:', precision_score(list(Y_pred), list(Y_test), average='weighted'))
# print('Recall:', recall_score(list(Y_pred), list(Y_test), average='weighted'))
# print('F1_score:', f1_score(list(Y_pred), list(Y_test), average='weighted'))

print('------Weighted------')
print('Weighted precision', precision_score(list(Y_pred), list(Y_test), average='weighted'))
print('Weighted recall', recall_score(list(Y_pred), list(Y_test), average='weighted'))
print('Weighted f1-score', f1_score(list(Y_pred), list(Y_test), average='weighted'))
print('------Macro------')
print('Macro precision', precision_score(list(Y_pred), list(Y_test), average='macro'))
print('Macro recall', recall_score(list(Y_pred), list(Y_test), average='macro'))
print('Macro f1-score', f1_score(list(Y_pred), list(Y_test), average='macro'))
print('------Micro------')
print('Micro precision', precision_score(list(Y_pred), list(Y_test), average='micro'))
print('Micro recall', recall_score(list(Y_pred), list(Y_test), average='micro'))
print('Micro f1-score', f1_score(list(Y_pred), list(Y_test), average='micro'))
```

    Accuracy: 0.9934193874968362
    ------Weighted------
    Weighted precision 0.993660434452819
    Weighted recall 0.9934193874968362
    Weighted f1-score 0.9934932765602341
    ------Macro------
    Macro precision 0.9692940889562662
    Macro recall 0.9891487280946443
    Macro f1-score 0.9789097544826614
    ------Micro------
    Micro precision 0.9934193874968362
    Micro recall 0.9934193874968362
    Micro f1-score 0.9934193874968362
![](https://github.com/ZhuJD-China/ECG-Classification-Diagnosis/blob/master/ECG%E5%AE%9E%E9%AA%8C%E6%8A%A5%E5%91%8Av1.1/output_60_0.png)
