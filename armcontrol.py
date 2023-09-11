import time

from HitbotInterface import HitbotInterface

print("-------------init the arm----------------")
robot_id = 18
robot = HitbotInterface(robot_id)
robot.net_port_initial()
time.sleep(0.5)
print("initial successed")
ret = robot.is_connect()
while ret != 1:
    time.sleep(0.1)
    ret = robot.is_connect()
    if(ret):
        print("robot is connected successfully.")
    else:
        time.sleep(2)
ret = robot.initial(3, 180)
if ret == 1:
    print("robot initial successful")
    robot.unlock_position()
else:
    print("robot initial failed")
if robot.unlock_position():
    print("------unlock------")
time.sleep(0.5)

if robot.is_connect():
    print("robot online")
    box_pos = [-64.6159, 269.41, -39]            # 不遮挡相机拍摄的机械臂位置，每次结束放置后移动至该位置。
    a = robot.new_movej_xyz_lr(box_pos[0], box_pos[1], -34, 0, speed=120, roughly=0, lr=1)
    robot.wait_stop()
    print("robot statics is {}".format(a))
    if a == 1: print("the robot is ready for the collection.")
    time.sleep(0.5)
