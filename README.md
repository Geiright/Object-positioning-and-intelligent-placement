# Intro
本项目介绍包含三个部分： 场景需求， 解决方案， 和技术要素。  
场景需求：  
边缘终端依据感知信息，确定目标物体，选择摆放空间，满足目标物体与摆放空间的匹配需求，实现智能摆放.  
    ![](/assets/firstplace1.png)
针对工业场景化妆品装箱问题，设计物体定位与智能摆放课题。在限定的A区域放置待装箱的物体，机械臂将其摆放至B区域的盒子中，整个过程由计算机视觉检测 + 匹配自动完成。  
    ![](/assets/A&B.png)
输入：使用Realsense相机输入RGB-D场景图像  
输出：吸取放置位置坐标，通过4轴机械臂将盒外积木放进盒中    
解决方案：  
1、模型：建立三个FCN网络，负责物块的吸取、放置和匹配  
2、数据集：预训练吸取网络，利用机械臂从盒中吸取物体，放置于随机位置，并倒转整个放置顺序，自动采集数据  
3、预处理：使用霍夫变换和分割，制作物体的mask，并进行背景消除  
4、标定：Eye to hand手眼标定，使用MLP regression，建立全连接网络，完成相机坐标和机械臂坐标的非线性变换  

# Installation
Pytorch版本选择1.7.1版本    

    pip install torch===1.7.1+cu110 torchvision===0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html    
    
    pip install -e .      
    
   
# Data
数据集形式, 参阅
[Object-positioning-and-intelligent-placement](https://github.com/Geiright/Deltademo)/**docs**/setup.md  
数据集放置位置
[Object-positioning-and-intelligent-placement](https://github.com/Geiright/Deltademo)/[form2fit](https://github.com/Geiright/Deltademo/tree/master/form2fit)/[code](https://github.com/Geiright/Deltademo/tree/master/form2fit/code)/[ml](https://github.com/Geiright/Deltademo/tree/master/form2fit/code/ml)/**dataset**/ 

11.12 数据集下载地址：  
链接：https://pan.baidu.com/s/1bgmDwMca-VCnwcaeMu9TKA  
提取码：    omuf  

# Weights
权重在[Deltademo](https://github.com/Geiright/Deltademo)/[form2fit](https://github.com/Geiright/Deltademo/tree/master/form2fit)/[code](https://github.com/Geiright/Deltademo/tree/master/form2fit/code)/[ml](https://github.com/Geiright/Deltademo/tree/master/form2fit/code/ml)/**savedmodel**/，目前全是吸取网络的权重    

# Infer
使用已有权重进行推理，可以在根目录文件夹下使用命令

    python3 form2fit/code/infer_suction.py

制作自己的数据集，并训练自己的吸取、放置、匹配网络。可以在根目录文件夹下使用命令

    python3 form2fit/code/train_suction.py
    python3 form2fit/code/train_correspondence.py
    python3 form2fit/code/train_placement.py

****
@inproceedings{zakka2020form2fit,
  title={Form2Fit: Learning Shape Priors for Generalizable Assembly from Disassembly},
  author={Zakka, Kevin and Zeng, Andy and Lee, Johnny and Song, Shuran},
  booktitle={Proceedings of the IEEE International Conference on Robotics and Automation},
  year={2020}
}
