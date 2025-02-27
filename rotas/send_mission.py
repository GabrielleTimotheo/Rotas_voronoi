#!/usr/bin/env python3

import math
from pymavlink import mavutil

class Mission_item:
    def __init__(self, i, current, x, y, z):
        """
        Class to format mission items
        
        Args:
            i (int): Sequence
            current (int): Current
            x (int): x coordinate
            y (int): y coordinate
            z (int): z coordinate
        """

        self.seq = i 
        self.frame = mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT 
        self.command = mavutil.mavlink.MAV_CMD_NAV_WAYPOINT
        self.current = current
        self.auto = 1
        self.param1 = 0.0
        self.param2 = 2.00
        self.param3 = 2.00
        self.param4 = 0
        self.param5 = int(x)
        self.param6 = int(y)
        self.param7 = int(z)
        self.mission_type = 0

def arm(the_connection):
    """
    Arm the drone
    
    Args:
        the_connection (mavutil.mavlink_connection): Connection to the drone
    """

    print("Arming")
    the_connection.mav.command_long_send(the_connection.target_system, the_connection.target_component,
                                         mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
    ack(the_connection, "COMMAND_ACK")

def upload_mission(the_connection, mission_items, retries=3):
    """
    Upload mission to the drone
    
    Args:
        the_connection (mavutil.mavlink_connection): Connection to the drone
        mission_items (list): List of mission items
        retries (int): Number of retries
    """

    n = len(mission_items)

    for attempt in range(retries):
        print(f"Sending mission attempt {attempt + 1}")
    
        the_connection.mav.mission_count_send(the_connection.target_system, the_connection.target_component, n, 0)
    
        for i, waypoint in enumerate(mission_items):
            print(f"Creating waypoint {waypoint.seq}: ({waypoint.param5}, {waypoint.param6}, {waypoint.param7})")
            
            the_connection.mav.mission_item_int_send(the_connection.target_system, 
                                                    the_connection.target_component, 
                                                    waypoint.seq, 
                                                    waypoint.frame, 
                                                    waypoint.command, 
                                                    waypoint.current, 
                                                    waypoint.auto, 
                                                    waypoint.param1, 
                                                    waypoint.param2, 
                                                    waypoint.param3, 
                                                    waypoint.param4, 
                                                    waypoint.param5, 
                                                    waypoint.param6, 
                                                    waypoint.param7, 
                                                    waypoint.mission_type)
            
            if i < n - 1:
                ack(the_connection, "MISSION_REQUEST")

# def upload_mission(the_connection, mission_items, retries=3):
#     """
#     Upload mission to the drone
    
#     Args:
#         the_connection (mavutil.mavlink_connection): Connection to the drone
#         mission_items (list): List of mission items
#         retries (int): Number of retries
#     """

#     n = len(mission_items)

#     for attempt in range(retries):
#         print(f"Sending mission attempt {attempt + 1}")

#         # Send the total number of waypoints
#         the_connection.mav.mission_count_send(the_connection.target_system, the_connection.target_component, n, 0)
    
#         # Send all waypoints at once
#         for waypoint in mission_items:
#             print(f"Creating waypoint {waypoint.seq}: ({waypoint.param5}, {waypoint.param6}, {waypoint.param7})")
#             the_connection.mav.mission_item_send(the_connection.target_system, 
#                                                  the_connection.target_component, 
#                                                  waypoint.seq, 
#                                                  waypoint.frame, 
#                                                  waypoint.command, 
#                                                  waypoint.current, 
#                                                  waypoint.auto, 
#                                                  waypoint.param1, 
#                                                  waypoint.param2, 
#                                                  waypoint.param3, 
#                                                  waypoint.param4, 
#                                                  waypoint.param5, 
#                                                  waypoint.param6, 
#                                                  waypoint.param7, 
#                                                  waypoint.mission_type)

#         ack(the_connection, "MISSION_ACK")

def start_mission(the_connection, retries=2):
    """
    Start mission
    
    Args:
        the_connection (mavutil.mavlink_connection): Connection to the drone
        retries (int): Number of retries
    """
    for attempt in range(retries):
        print(f"Starting mission attempt {attempt + 1}")
        the_connection.mav.command_long_send(the_connection.target_system, the_connection.target_component, 
                                             mavutil.mavlink.MAV_CMD_MISSION_START, 0, 0, 0, 0, 0, 0, 0, 0)
        ack(the_connection, "COMMAND_ACK")

def ack(the_connection, keyword):
    """
    Acknowledge the message
    
    Args:
        the_connection (mavutil.mavlink_connection): Connection to the drone
        keyword (str): Keyword to be read
    """
    print("Before sending %s" % keyword)
    print("After sending %s" % keyword)
    print("Message read " + str(the_connection.recv_match(type=keyword, blocking=True)))

def generate_mission(mission_points):
    """
    Generate mission
    
    Args:
        mission_points (list): List of mission points
    """
    the_connection = mavutil.mavlink_connection('udp:127.0.0.1:14550')

    while the_connection.target_system == 0:
        print("Checking heartbeat")
        the_connection.wait_heartbeat()
        print("Heartbeat from system (system %u component %u)" % (the_connection.target_system, the_connection.target_component))
    arm(the_connection)
    mission_waypoints = []
    counter = 0

    for i in range(0, len(mission_points)): 
        point = mission_points[i]
        mission_waypoints.append(Mission_item(counter, 0, point[0] * 10 ** 7, point[1] * 10 ** 7, 0))
        counter += 1
    print(mission_points)
    
    upload_mission(the_connection, mission_waypoints)
    ack(the_connection, "MISSION_ACK")
    start_mission(the_connection)

    for mission_item in mission_waypoints:
        print("Message Read " + str(the_connection.recv_match(type="MISSION_ITEM_INT")))

