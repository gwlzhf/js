# coding: utf-8
import serial 
from math import *
import numpy as np


X,Y = 255,0
#l为平面上的欧式距离
l = 255
A = ((l-115)**2-471) / (2*97*sqrt(27**2+(l-115)**2))
fi = atan(27/(l-115))

#1号杆转动定位角度弧度值theta1_h，2号杆转动定位角度弧度值theta2_h
theta2_h = asin(A) - fi
theta1_h = asin((97*cos(theta2_h) - 27) / 103)

#弧度转角度
theta2 = theta2_h*180/pi
theta1 = theta1_h*180/pi
#0-5电机的输入脉冲信号
P1i = int(1000 + (theta1 - 10)*500 / 70)
P2i = int(1500 + (90 + theta1 - theta2 - 10)*500 / 70)
P3i = int(1500 + (90 - theta2 - 8)*500 / 70)
P4i = 2142
P5i = 1500


def Motor_move(num = 1, time = 4000, motor_id = 0, position = 1500):
	command = [85, 85, 8, 3, 1, 232, 0, 0, 208, 8]
	num_of_bytes = num*3+5
	command[2] = num_of_bytes
	command[4] = num
	time_L = time % 256
	time_H = floor(time/256)
	command[5] = time_L
	command[6] = time_H
	command[7] = motor_id
	pos_L = position % 256
	pos_H = floor(position/256)
	command[8] = pos_L
	command[9] = pos_H
	command=bytes(command)
	ser.write(command)
	ser.read()
	print(command)



 
ser=serial.Serial("/dev/ttyUSB0",9600, timeout=0.5) 
#Motor_move(position=1500,time = 2000,motor_id=6)
Motor_move(position=1500,time = 2000,motor_id=4)
Motor_move(position=1500,time = 2000,motor_id=3)
Motor_move(position=1500,time=4000,motor_id=2)
#Motor_move(position=1500,time=4000,motor_id=1)


#Motor_move(position=1900,time = 2000,motor_id=6)
Motor_move(position=P4i,time = 2000,motor_id=4)
Motor_move(position=P3i,time = 2000,motor_id=3)
Motor_move(position=P2i,time=4000,motor_id=2)
#Motor_move(position=P1i,time=4000,motor_id=1)
##Motor_move(position=1500,time=4000,motor_id=0)

ser.close()
