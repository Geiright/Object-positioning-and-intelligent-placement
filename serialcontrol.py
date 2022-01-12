import time
from typing import Counter
import serial
import threading



def pump_off():         # 旧的输出低电平方法 输出低电平，也就是吸泵停止
    i = 0
    while(1):
        if ser.isOpen():
            ser.write('00000000'.encode("gbk"))
            time.sleep(0.001)
            i += 1
            if i == 50:
                break


ser = serial.Serial("com8", 9600, timeout=0.5)
        

if __name__ == '__main__':

                                                                                                      
    pump_off()
    time.sleep(3)

    