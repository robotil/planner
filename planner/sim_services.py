#!/usr/bin/env python3

from geometry_msgs.msg import Point
from planner_msgs.srv import CheckLOS, AllPathEntityToTarget
import rclpy


def check_line_of_sight(one, two):
    """ check_line_of_sight
    Args:
        one: geometry_msgs/Point
        two: geometry_msgs/Point

    Returns:
        Boolean
           True if in line of sight
           False if not
    """

    # TODO consider look at direction

    res = False
    node = rclpy.create_node('check_line_of_sight')
    line_of_sight_cli = node.create_client(CheckLOS, 'check_line_of_sight')

    while not line_of_sight_cli.wait_for_service(timeout_sec=1.0):
        print('CheckLOS not available, waiting again...')

    req = CheckLOS.Request()
    req.one = one
    req.two = two
    future = line_of_sight_cli.call_async(req)
    rclpy.spin_until_future_complete(node, future)
    if future.result() is not None:
        node.get_logger().info('Result of check_line_of_sight_request: %s' % future.result().is_los.__str__())
        res = future.result().is_los
    else:
        node.get_logger().error('Exception while calling service: %r' % future.exception())

    node.destroy_node()

    return res


# get_all_possible_ways
# Args:
#    entityid=String
#    target=geometry_msgs/Point
# Return:
#   Dictionary of paths
#   All possible path between entity and target

def get_all_possible_ways(entityid, target):
    res = {}
    node = rclpy.create_node('get_all_possible_ways')
    get_all_possible_ways_cli = node.create_client(AllPathEntityToTarget, 'get_all_possible_ways')

    while not get_all_possible_ways_cli.wait_for_service(timeout_sec=1.0):
        print('AllPathEntityToTarget not available, waiting again...')

    req = AllPathEntityToTarget.Request()
    req.entityid = entityid
    req.target = target
    future = get_all_possible_ways_cli.call_async(req)
    rclpy.spin_until_future_complete(node, future)
    if future.result() is not None:
        node.get_logger().info('Result of check_line_of_sight_request: %s' % future.result().path.__str__())
        res = future.result().path
    else:
        node.get_logger().error('Exception while calling service: %r' % future.exception())

    node.destroy_node()

    return res


if __name__ == '__main__':
    rclpy.init()
    p1 = Point(x=0.2, y=0.2, z=0.2)
    p2 = Point(x=0.4, y=0.4, z=0.4)
    ret = check_line_of_sight(p1, p2)
    print("ret state value=" + ret.__str__() + " type" + str(type(ret)))
    ret = get_all_possible_ways("Suicide", p2)
    print("ret act value=" + ret.__str__() + " type" + str(type(ret)))
    rclpy.shutdown()
