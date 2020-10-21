#!/usr/bin/env python3

from planner_msgs.srv import StateGeneralAdmin, ActGeneralAdmin
import rclpy

def check_state_simulation(args=None):
    res_int = 255

    node = rclpy.create_node('check_state_simulation')
    stateAdminCli = node.create_client(StateGeneralAdmin, 'state_general_admin')

    while not stateAdminCli.wait_for_service(timeout_sec=1.0):
        print('StateGeneralAdmin not available, waiting again...')
    req = StateGeneralAdmin.Request()
    future = stateAdminCli.call_async(req)
    rclpy.spin_until_future_complete(node, future)
    if future.result() is not None:
        node.get_logger().info('Result of check_state_simulation: %s' % future.result().resulting_status.__str__())
        res_int = int.from_bytes(future.result().resulting_status, "big")
    else:
        node.get_logger().error('Exception while calling service: %r' % future.exception())

    node.destroy_node()

    return res_int

def act_on_simulation(args="0"):
    res_int = 255

    command = int(args)
    node = rclpy.create_node('act_on_simulation')
    genAdminCli = node.create_client(ActGeneralAdmin, 'act_general_admin')

    while not genAdminCli.wait_for_service(timeout_sec=1.0):
        print('ActGeneralAdmin not available, waiting again...')
    req = ActGeneralAdmin.Request()
    req.admin = bytes([command])
    future = genAdminCli.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=3.0)
    if future.result() is not None:
        node.get_logger().info('Result of act_on_simulation: %s' % future.result().resulting_status.__str__())
        res_int = int.from_bytes(future.result().resulting_status, "big")
    else:
        node.get_logger().error('Exception while calling service: %r' % future.exception())
        res_int = 0

    node.destroy_node()
    return res_int

if __name__ == '__main__':
    rclpy.init()
    ret=check_state_simulation()
    print("ret state value="+ret.__str__()+" type"+str(type(ret)))
    ret=act_on_simulation("2")
    print("ret act value="+ret.__str__()+" type"+str(type(ret)))
    rclpy.shutdown()