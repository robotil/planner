import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from diagnostic_msgs.msg import DiagnosticStatus
from actionlib_msgs.msg import GoalID, GoalStatus, GoalStatusArray
from geometry_msgs.msg import PointStamped, PolygonStamped, Twist, TwistStamped, PoseStamped
from planner_msgs.msg import SDiagnosticStatus, SGlobalPose, SHealth, SImu, EnemyReport
from planner_msgs.srv import ActGeneralAdmin, StateGeneralAdmin, CheckLOS

class DummyServer(Node):
    def __init__(self):
        super().__init__('dummy_server')
        self.state = bytes([0])
        self.actAdminSrv = self.create_service(ActGeneralAdmin, 'act_general_admin', self.act_general_admin_callback)
        self.stateAdminSrv = self.create_service(StateGeneralAdmin, 'state_general_admin', self.state_general_admin_callback)
        self.checkLOSSrv = self.create_service(CheckLOS, 'check_line_of_sight', self.check_line_of_sight_callback)

    def act_general_admin_callback(self, request, response):
        response.resulting_status = request.admin
        self.get_logger().info('Incoming request\na: adm: %d' % (int.from_bytes(request.admin, "big")))
        self.state = request.admin
        return response

    def state_general_admin_callback(self, request, response):
        self.get_logger().info('Returning state\n state: %d' % (int.from_bytes(self.state, "big")))
        response.resulting_status = self.state
        return response

    def check_line_of_sight_callback(self, request, response):
        point1 = request.one
        point2 = request.two
        self.get_logger().info('Got request for point 1='+point1.__str__()+' and point 2='+point2.__str__())
        response.is_los = True
        return response

def main(args=None):
    rclpy.init(args=args)

    dummy_server = DummyServer()

    rclpy.spin(dummy_server)

    rclpy.shutdown()

if __name__ == '__main__':
    main()