#!/usr/bin/env python3
import sys
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

from std_msgs.msg import String
from diagnostic_msgs.msg import DiagnosticStatus
from actionlib_msgs.msg import GoalID, GoalStatus, GoalStatusArray
from geometry_msgs.msg import PointStamped, PolygonStamped, Twist, TwistStamped, PoseStamped
from planner_msgs.msg import SDiagnosticStatus, SGlobalPose, SHealth, SImu, EnnemyReport
from planner_msgs.srv import ActGeneralAdmin, StateGeneralAdmin, CheckLOS

class WorldCom(Node):

    def __init__(self):
        super().__init__('world_communication')
        self.world_state = {}
        self.entities = {}
        self.ennemies = {}
        self.act_req = 0
        self.stat_req = 0
        self.ch_los_req = 0
        self.entityPoseSub = self.create_subscription(SGlobalPose, '/entity/global_pose', self.global_pose_callback, 10)

        self.entityDescriptionSub = self.create_subscription(SDiagnosticStatus, '/entity/description', self.entity_description_callback, 10)
        self.ennemyDescriptionSub = self.create_subscription(EnnemyReport, '/ennemy/description', self.ennemy_description_callback, 10)
        self.entityImuSub = self.create_subscription(SImu, '/entity/imu', self.entity_imu_callback, 10)
        self.entityOverallHealthSub = self.create_subscription(SHealth, '/entity/overall_health', self.entity_overall_health_callback, 10)

        self.moveToPub = self.create_publisher(SGlobalPose, '/entity/move_to/goal', 10)
        self.attackPub = self.create_publisher(SGlobalPose, '/entity/attack/goal', 10)
        self.lookPub = self.create_publisher(SGlobalPose, '/entity/look/goal', 10)

        self.genAdminCli = self.create_client(ActGeneralAdmin, 'act_general_admin')
        self.stateAdminCli = self.create_client(StateGeneralAdmin, 'state_general_admin')

        timer_period = 10  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.client_futures = []
        self.i = 2
        self.future = None

    def act_gen_admin_request(self, command):
        if (command > 2) or (command < 0):
            self.get_logger().warn('wrong command:'+ascii(command))
            return

        if self.act_req == 0:
            while not self.genAdminCli.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('service ActGeneralAdmin not available, waiting ...')
        self.act_req = ActGeneralAdmin.Request()

        self.act_req.admin = bytes([command])
        self.future = self.genAdminCli.call_async(self.act_req)

    def state_gen_admin_request(self):
        self.stat_req = StateGeneralAdmin.Request()
        while not self.stateAdminCli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service StateGeneralAdmin not available, waiting ...')

        self.client_futures.append(self.stateAdminCli.call_async(self.stat_req))
        # future = self.stateAdminCli.call_async(self.stat_req)
        # # #self.get_logger().info('state_gen_admin_request sent, waiting ...')
        # # #rclpy.spin_until_future_complete(self, future)
        # while not future.done():
        #     continue
        # try:
        #     response = future.result()
        # except Exception as e:
        #     self.get_logger().info('Service call failed %r' % (e,))
        # else:
        #     self.get_logger().info('State_gen_admin_request: %d' % response.resulting_status)

    def timer_callback(self):
        self.get_logger().info('Timer callback: "%d"' % self.i)
        self.act_gen_admin_request(self.i)
        #self.goto.publish(msg)
        if self.i==3:
            self.i=0
        else:
            self.i += 1
        self.state_gen_admin_request()

    def global_pose_callback(self, msg):
        self.get_logger().info('Received: "%s"' % msg)
        self.entities[msg.id.__str__()] = msg.gpose.point
        self.world_state['GlobalPose'] = msg.gpose.point

    def entity_description_callback(self, msg):
        self.get_logger().info('Received: "%s"' % msg)

        self.entities[msg.id.__string__].name = msg.name
        self.entities[msg.id.__string__].message = msg.message

    def ennemy_description_callback(self, msg):
        self.get_logger().info('Received: "%s"' % msg)

    #    self.ennemies[msg.id.__string__].name = msg.name
    #    self.entities[msg.id.__string__].message = msg.message

    def entity_imu_callback(self, msg):
        self.get_logger().info('Received: "%s"' % msg)
        self.entities[msg.id.__string__].imu = msg.imu

    def entity_overall_health_callback(self, msg):
        self.get_logger().info('Received: "%s"' % msg)
        self.entities[msg.id.__string__].overallhealth = msg.values

    def our_spin(self):
        while rclpy.ok():
            rclpy.spin_once(self)
            incomplete_futures = []
            for f in self.client_futures:
                if f.done():
                    res = f.result()
                    print("received service result: {}".format(res))
                else:
                    incomplete_futures.append(f)
            self.client_futures = incomplete_futures

def main(args=None):
    rclpy.init(args=args)

    world_communication = WorldCom()

    # while rclpy.ok():
    #     rclpy.spin_once(world_communication)
    #     if world_communication.future.done():
    #         try:
    #             response = world_communication.future.result()
    #         except Exception as e:
    #             world_communication.get_logger().info('Service call failed %r' % (e,))
    #         else:
    #             world_communication.get_logger().info('Result of ActGeneralAdmin: %d' % (int.from_bytes(response.resulting_status, "big")))

    world_communication.our_spin()
    #rclpy.spin(world_communication)
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    world_communication.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()