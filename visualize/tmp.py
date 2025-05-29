import matplotlib.pyplot as plt
import numpy as np

# 数据
labels = ['Standard Matches', 'Semantic Matches']
percentages = [92.9, 7.1]

# 创建直方图
x = np.arange(len(labels))  # 标签位置
width = 0.5  # 条形宽度

fig, ax = plt.subplots()
rects = ax.bar(x, percentages, width, color=['#1f77b4', '#ff7f0e'])

# 添加标签、标题和自定义x轴刻度标签等
ax.set_ylabel('Percentage (%)')
ax.set_title('Match Composition')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# 在条形上方显示百分比
def autolabel(rects):
    """在每个条形上方附加一个文本标签，显示其高度."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}%'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects)

fig.tight_layout()

# 显示图形
plt.show()