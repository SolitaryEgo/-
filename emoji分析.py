import pandas as pa
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


file_path = 'emoji_usage_dataset.csv'
data = pd.read_csv(file_path)

print(data.head())

# 统计每个表情的使用次数
emoji_usage_counts = data['Emoji'].value_counts().reset_index()
emoji_usage_counts.columns = ['emoji','count']
print(emoji_usage_counts)

# 年龄和表情对数据进行分组，计算使用频率
age_emoji_counts = data.groupby(['User Age','Emoji']).size().unstack(fill_value=0)
age_emoji_counts = age_emoji_counts.reindex(age_emoji_counts.sum(axis=1).sort_values(ascending=False).index,axis=0)
print(age_emoji_counts)

# 情景与表情的关联,计算使用频率
context_emoji_counts = data.groupby(['Context','Emoji']).size().unstack(fill_value=0)
context_emoji_counts = context_emoji_counts.reindex(context_emoji_counts.sum(axis=1).sort_values(ascending=False).index,axis=0)
print(context_emoji_counts)


# 用户偏好聚类分析

# 将分类特征转化为数值型
label_encoders = {}
for column in ['Context','Platform','User Age','User Gender']:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# 数据预处理，特征标准化
scaler = StandardScaler()
features_scaled = scaler.fit_transform(data[['Context','Platform','User Age','User Gender']])

# 进行PCA降维
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_scaled)

# kmeans聚类,并添加到原始数据中
kmeans = KMeans(n_clusters=3,random_state=0)
data['Cluster'] = kmeans.fit_predict(features_pca)

print(data[['Cluster', 'Context', 'Platform', 'User Age', 'User Gender']].head())

#         Cluster  Context  Platform  User Age  User Gender
# 0        2        0         2        34            1
# 1        1        6         1        23            1
# 2        2        0         2        38            1
# 3        1        2         0        51            1
# 4        1        1         2        43            0


