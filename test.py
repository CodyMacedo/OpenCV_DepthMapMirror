import serial
import time

arduino = serial.Serial('COM5', 115200, timeout=.1)
time.sleep(1)

for i in range(32):
    for j in range(32):
        send = str(i) + " " + str(j)
        arduino.write(send)
