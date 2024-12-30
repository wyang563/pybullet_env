import socket
import threading
import pybullet as p
import cv2
import numpy as np
from simulator.simulator_utils import *
import csv

'''
We don't actually need this logic now since we can just simulate pybullet iteratively by time step, but this will be useful for
when we move to MAVSDK
'''
class DroneServer:
    def __init__(self, drone_id, ip, port, other_addresses, loop):
        self.drone_id = drone_id
        self.ip = ip
        self.port = port
        self.server = self.start_server()
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((ip, port))  # Bind to specific IP and port
        self.server_socket.listen(5)
        self.threads = []
        self.other_addresses = other_addresses
        self.loop = loop

    ## Server Housekeeping Stuff

    def start_server(self):
        try:
            self.running = True
            print(f"Drone {self.drone_id} server started on {self.ip}:{self.port}")
            while self.running:
                try:
                    client_socket, _ = self.server_socket.accept()
                    thread = threading.Thread(target=self.handle_connection, args=(client_socket,), daemon=True)
                    self.threads.append(thread)
                    thread.start()
                except OSError:
                    break
        except Exception as e:
            print(f"server error: {e}")
        finally:
            self.server_socket.close()
    
    def handle_connection(self, client_socket):
        data = client_socket.recv(1024).decode()
        command, data = data.split('|')
        if command == "fly-to":
            displacement, speed = data.split(';')
            print(f"Drone {self.drone_id} flying to displacement: {displacement} at speed: {speed}")
            dx, dy, dz = displacement.split(',')
            # convert values
            dx, dy, dz = float(dx), float(dy), float(dz)
            speed = float(speed)
            # TODO: add logic

            print("Drone finished fly-to operation")
        
        elif command == "set-velocity":
            forward_speed, right_speed, down_speed, yaw_speed, K, yaw_K = data.split(';')
            print(f"Drone {self.drone_id} setting velocity to: {forward_speed}, {right_speed}, {down_speed}, {yaw_speed}")
            forward_speed, right_speed, down_speed, yaw_speed, K, yaw_K = float(forward_speed), float(right_speed), float(down_speed), float(yaw_speed), float(K), float(yaw_K)
            # TODO: add logic 

            print("Drone finished set-velocity operation")

    def send_data(self, encoded_message):
        for ip, port in self.other_addresses:
            try:
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.connect((ip, port))  # Connect to specific IP and port
                client_socket.send(encoded_message)
                client_socket.close()
                print(f"Drone {self.drone_id} sent data to {ip}:{port}")
            except ConnectionRefusedError:
                print(f"Drone {self.drone_id} could not connect to Drone on {ip}:{port}")

    def send_command(self, command, *args):
        # data passed in as a list of parameters
        if command not in ["fly-to", "set-velocity", "land"]:
            print(f"Invalid command: {command}")
            return
        data_str = ";".join(args)
        message = f"{command}|" + data_str
        self.send_data(message.encode())

    def stop_server(self, signum=0, frame=0):
        print("Shutting down server...")
        self.running = False
        self.server_socket.close()
        for thread in self.threads:
            if thread.is_alive():
                thread.join()