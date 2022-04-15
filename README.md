# Intro
边缘终端依据感知信息，确定目标物体，选择摆放空间，满足目标物体与摆放空间的匹配需求，实现智能摆放.  
针对工业场景化妆品装箱问题，设计物体定位与智能摆放课题。在限定的A区域放置待装箱的物体，机械臂将其摆放至B区域的盒子中，整个过程由计算机视觉检测 + 匹配自动完成。  

    ![](/assets/firstplace1.png)
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

@inproceedings{zakka2020form2fit,
  title={Form2Fit: Learning Shape Priors for Generalizable Assembly from Disassembly},
  author={Zakka, Kevin and Zeng, Andy and Lee, Johnny and Song, Shuran},
  booktitle={Proceedings of the IEEE International Conference on Robotics and Automation},
  year={2020}
}
