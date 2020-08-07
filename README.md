# TextClassification
Text classification using different neural networks.
中文文本分类，使用TensorFlow 2.x实现TextCNN，TextRNN，TextRCNN，BiLSTM Attention, HAN等类型的深度学习模型。
## 数据
数据来源于 [搜狗新闻数据](https://www.sogou.com/labs/resource/ca.php)  
数据只取新闻中的五个类别：汽车、娱乐、军事、体育、科技
将五个类别分词后保存到data文件目录，作为分类模型构建与验证数据。
数据集|数据量
--|--
总数据|87747
训练集|65810
测试集|21973
## 环境
python 3.7   
TensorFlow 2.0+

## 使用说明
进入到相关模型目录下
```
# 使用默认参数运行
python main.py
```

### 参数
进入到相关模型目录下
```
# 查看模型相关参数
python main.py -h
```
### 后记
本项目是作者之前使用或者学习过的神经网络模型使用TensorFlow 2.x实现各个标准神经网络分类模型，适合学习和快速开发调试，实际项目使用需要结合数据和业务场景调整，望周知。