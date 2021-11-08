import serial
import serial.tools.list_ports

from pymycobot.mycobot import MyCobot



def setup():
    print("")

    plist = list(serial.tools.list_ports.comports())
    idx = 1
    for port in plist:
        print("{} : {}".format(idx, port))
        idx += 1
    if len(plist) == 0:
        raise RuntimeError('Can not find the arm, please check your usb port.')
    _in =  1 # input("\nPlease input 1 - {} to choice:".format(idx - 1))
    
    port = str(plist[int(_in) - 1]).split(" - ")[0].strip()
    print(port)
    print("")

    baud = 115200
    #_baud = input("Please input baud(default:115200):")
    _baud = 115200
    try:
        baud = int(_baud)
    except Exception:
        pass
    print(baud)
    print("")

    DEBUG = False
    #f = input("Wether DEBUG mode[Y/n]:")
    f = "n"
    if f in ["y", "Y", "yes", "Yes"]:
        DEBUG = True
    # mc = MyCobot(port, debug=True)
    mc = MyCobot(port, baud, debug=DEBUG)
    return mc

