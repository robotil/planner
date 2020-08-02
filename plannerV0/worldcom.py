#!/usr/bin/env python3
import sys
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

from std_msgs.msg import String
from diagnostic_msgs.msg import DiagnosticStatus, KeyValue
from actionlib_msgs.msg import GoalID, GoalStatus, GoalStatusArray
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PointStamped, PolygonStamped, Twist, TwistStamped, PoseStamped
from planner_msgs.msg import SDiagnosticStatus, SGlobalPose, SHealth, SImu, EnemyReport
from planner_msgs.srv import ActGeneralAdmin, StateGeneralAdmin, CheckLOS
import planner_msgs

class WorldCom(Node):

    class Enemy:
        def __init__(self, msg):
            self.cep = msg.cep
            self.gpose = msg.gpose
            self.priority=msg.priority
            self.tclass = msg.tclass
            self.is_alive = msg.is_alive
            self.id = msg.id

        def update(self, n_enn):
            self.cep = n_enn.cep
            self.gpose = n_enn.gpose
            self.priority = n_enn.priority
            self.tclass = n_enn.tclass
            self.is_alive = n_enn.is_alive

    class Entity:
        def __init__(self, msg):
            self.id = msg.id
            self.diagstatus = msg.diagstatus
            self.gpose= PointStamped()
            self.imu = Imu()
            self.health = KeyValue()


        def update_desc(self, n_ent):
            self.diagstatus = n_ent.diagstatus

        def update_gpose(self, n_pose):
            self.gpose = n_pose

        def update_imu(self, n_ent):
            self.imu = n_ent.gpose

        def update_health(self, n_ent):
            self.health = n_ent.health

    def __init__(self):
        super().__init__('world_communication')
        self.world_state = {}
        self.entities = []
        self.enemies = []
        self.act_req = 0
        self.stat_req = 0
        self.ch_los_req = 0
        self.entityPoseSub = self.create_subscription(SGlobalPose, '/entity/global_pose', self.global_pose_callback, 10)

        self.entityDescriptionSub = self.create_subscription(SDiagnosticStatus, '/entity/description', self.entity_description_callback, 10)
        self.enemyDescriptionSub = self.create_subscription(EnemyReport, '/enemy/description', self.enemy_description_callback, 10)
        self.entityImuSub = self.create_subscription(SImu, '/entity/imu', self.entity_imu_callback, 10)
        self.entityOverallHealthSub = self.create_subscription(SHealth, '/entity/overall_health', self.entity_overall_health_callback, 10)

        self.moveToPub = self.create_publisher(SGlobalPose, '/entity/move_to/goal', 10)
        self.attackPub = self.create_publisher(SGlobalPose, '/entity/attack/goal', 10)
        self.lookPub = self.create_publisher(SGlobalPose, '/entity/look/goal', 10)

        self.genAdminCli = self.create_client(ActGeneralAdmin, 'act_general_admin')
        self.stateAdminCli = self.create_client(StateGeneralAdmin, 'state_general_admin')
        self.lineOfSightCli = self.create_client(CheckLOS, 'check_line_of_sight')

        timer_period = 10  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.client_futures = []
        self.i = 2
        self.future = None

    def get_entity(self, id):
        res = False
        found = None
        for elem in self.entities:
            if elem.id == id:
                found = elem
                break
        return elem

    def get_enemy(self, id):
        res = False
        found = None
        for elem in self.enemies:
            if elem.id == id:
                found = elem
                break
        return elem

    def act_gen_admin_request(self, command):
        if (command > 2) or (command < 0):
            self.get_logger().warn('wrong command:'+ascii(command))
            return
        self.act_req = ActGeneralAdmin.Request()
        while not self.genAdminCli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service ActGeneralAdmin not available, waiting ...')
        self.act_req.admin = bytes([command])
        self.client_futures.append(self.genAdminCli.call_async(self.act_req))

        # if self.act_req == 0:
        #     while not self.genAdminCli.wait_for_service(timeout_sec=1.0):
        #         self.get_logger().info('service ActGeneralAdmin not available, waiting ...')
        # self.act_req = ActGeneralAdmin.Request()
        #
        # self.act_req.admin = bytes([command])
        # self.future = self.genAdminCli.call_async(self.act_req)

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

    def check_line_of_sight_request(self, entity, enemy):
        self.ch_los_req = CheckLOS.Request()
        while not self.lineOfSightCli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service CheckLOS not available, waiting ...')

        self.ch_los_req.one = entity.gpose
        self.ch_los_req.two = enemy.gpose
        self.client_futures.append(self.lineOfSightCli.call_async(self.ch_los_req))

    def timer_callback(self):
        self.get_logger().info('Timer callback: "%d"' % self.i)
        self.act_gen_admin_request(self.i)
        if self.i==3:
            self.i=0
        else:
            self.i += 1
        self.state_gen_admin_request()

    def global_pose_callback(self, msg):
        this_entity = self.get_entity(msg.id)
        if (this_entity==None):
            self.get_logger().info('This entity "%s" is not managed yet' % msg.id)
            return
        this_entity.update_pose(msg.gpose.point)
        self.world_state['GlobalPose'] = msg.gpose.point
        self.get_logger().debug('Received: "%s"' % msg)


    def entity_description_callback(self, msg):
        a = self.Entity(msg)
        res = False
        for elem in self.entities:
            if elem.id == a.id:
                elem.update_desc(a)
                res = True
                break
        if not res:
            self.entities.append(a)

        self.get_logger().debug('Received: "%s"' % msg)

    def enemy_description_callback(self, msg):
        a = self.Enemy(msg)
        res=False
        for elem in self.enemies:
            if elem.id == a.id:
                elem.update(a)
                res=True
                break
        if not res:
            self.enemies.append(a)
        self.get_logger().debug('Received: "%s"' % msg)



    def entity_imu_callback(self, msg):
        this_entity = self.get_entity(msg.id)
        if (this_entity == None):
            self.get_logger().info('This entity "%s" is not managed yet' % msg.id)
            return
        this_entity.update_imu(msg.imu)
        self.world_state['IMU'] = msg.imu
        self.get_logger().debug('Received: "%s"' % msg)


    def entity_overall_health_callback(self, msg):
        this_entity = self.get_entity(msg.id)
        if (this_entity == None):
            self.get_logger().info('This entity "%s" is not managed yet' % msg.id)
            return
        this_entity.update_health(msg.values)
        self.world_state['Health'] = msg.values
        self.get_logger().debug('Received: "%s"' % msg)

    def our_spin(self):
        while rclpy.ok():
            rclpy.spin_once(self)
            incomplete_futures = []
            for f in self.client_futures:
                if f.done():
                    res = f.result()
                    #print("received service result: {}".format(res))
                    res_int = int.from_bytes(res.resulting_status,"big")
                    #print("service result: {}"+ascii(res_int))
                    #print("Cool: " + format(res))
                    #print("Class: " + res.__class__.__str__())
                    if type(res).__name__=='ActGeneralAdmin_Response':
                        print("ActGeneralAdmin_Response: "+res.resulting_status.__str__())
                    if type(res).__name__=='StateGeneralAdmin_Response':
                        print("StateGeneralAdmin_Response: " + res.resulting_status.__str__())
                    if type(res).__name__=='CheckLOS_Response':
                        print("CheckLOS_Response: " + res.is_los.__str__())
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