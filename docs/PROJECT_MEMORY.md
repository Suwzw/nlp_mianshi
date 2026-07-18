# 项目长期记忆

## 项目概述
保研面试课题：基于大小模型协同的小样本低空宽带信号检测
- 选题：A2+B7+C2
- 面试：7.19周日19:20，南科大卓工夏令营
- 数据集：低空五类宽带信号（WiFi/BT/ZigBee/Lightbridge/XPD），YOLO格式

## 关键约定
- 数据集必须按录制场次划分，不能随机划分（避免数据泄露）
- 知识蒸馏采用分级策略：伪标签保底→响应级KD冲刺
- 训练环境：conda pytorch环境，batch=8, workers=2, amp=False
- ultralytics安装在pytorch conda环境

## 用户偏好
- 深度优先，讲透一个小问题，不贪大求全
- 需要Demo展示（Streamlit）
- 最终交付：代码+实验图表+3页PPT+3分钟讲稿
