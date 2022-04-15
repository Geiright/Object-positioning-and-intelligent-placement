# Intro
边缘终端依据感知信息，确定目标物体，选择摆放空间，满足目标物体与摆放空间的匹配需求，实现智能摆放
! (/assets/firstplace1.png)
# Installation
    pip install -e .  。
# Data
数据集形式不变，环境的安装方法与原form2fit相同，具体在[Deltademo](https://github.com/Geiright/Deltademo)/**docs**/setup.md参阅
数据集放在[Deltademo](https://github.com/Geiright/Deltademo)/[form2fit](https://github.com/Geiright/Deltademo/tree/master/form2fit)/[code](https://github.com/Geiright/Deltademo/tree/master/form2fit/code)/[ml](https://github.com/Geiright/Deltademo/tree/master/form2fit/code/ml)/**dataset**/ 

11.12 数据集下载地址：  
链接：https://pan.baidu.com/s/1bgmDwMca-VCnwcaeMu9TKA  
提取码：    omuf  
# Weights
权重在[Deltademo](https://github.com/Geiright/Deltademo)/[form2fit](https://github.com/Geiright/Deltademo/tree/master/form2fit)/[code](https://github.com/Geiright/Deltademo/tree/master/form2fit/code)/[ml](https://github.com/Geiright/Deltademo/tree/master/form2fit/code/ml)/**savedmodel**/，目前全是吸取网络的权重
# Infer
使用已有权重进行推理，可以在Deltademo文件夹下使用命令

    python3 form2fit/code/infer_suction.py

训练自己的吸取网络，可以在Deltademo文件夹下使用命令

    python3 form2fit/code/train_suction.py

@inproceedings{zakka2020form2fit,
  title={Form2Fit: Learning Shape Priors for Generalizable Assembly from Disassembly},
  author={Zakka, Kevin and Zeng, Andy and Lee, Johnny and Song, Shuran},
  booktitle={Proceedings of the IEEE International Conference on Robotics and Automation},
  year={2020}
}
