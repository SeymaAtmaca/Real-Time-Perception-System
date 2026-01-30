#!/usr/bin/env python3

from gz.transport13 import Node
from gz.msgs10.twist_pb2 import Twist
import sys
import termios
import tty

def get_key():
    """Klavye tuşunu oku"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        key = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return key

def main():
    node = Node()
    pub = node.advertise("/model/quadcopter/cmd_vel", Twist)
    
    print("Quadcopter Controller")
    print("-------------------")
    print("W/S: Up/Down")
    print("A/D: Left/Right")
    print("I/K: Forward/Backward")
    print("Q: Exit")
    print()
    
    speed = 2.0  # Hareket hızı
    
    while True:
        key = get_key()
        
        msg = Twist()
        
        if key == 'w':
            msg.linear.z = speed
            print("↑ Yukarı")
        elif key == 's':
            msg.linear.z = -speed
            print("↓ Aşağı")
        elif key == 'a':
            msg.linear.y = speed
            print("← Sol")
        elif key == 'd':
            msg.linear.y = -speed
            print("→ Sağ")
        elif key == 'i':
            msg.linear.x = speed
            print("↑ İleri")
        elif key == 'k':
            msg.linear.x = -speed
            print("↓ Geri")
        elif key == 'q':
            print("Çıkış...")
            break
        else:
            msg.linear.x = 0
            msg.linear.y = 0
            msg.linear.z = 0
        
        pub.publish(msg)

if __name__ == '__main__':
    main()