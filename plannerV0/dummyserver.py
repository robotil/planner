import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from diagnostic_msgs.msg import DiagnosticStatus
from actionlib_msgs.msg import GoalID, GoalStatus, GoalStatusArray
from geometry_msgs.msg import PointStamped, PolygonStamped, Twist, TwistStamped, PoseStamped
from planner_msgs.msg import SDiagnosticStatus, SGlobalPose, SHealth, SImu, EnnemyReport
from planner_msgs.srv import ActGeneralAdmin, StateGeneralAdmin, CheckLOS

class DummyServer(Node):

    def __init__(self):
        super().__init__('dummy_server')
        self.srv = self.create_service(ActGeneralAdmin, 'act_general_admin', self.act_general_admin_callback)        # CHANGE

    def act_general_admin_callback(self, request, response):
        response.resulting_status = request.admin                                               # CHANGE
        self.get_logger().info('Incoming request\na: adm: %d' % (int.from_bytes(request.admin, "big")))
        return response

def main(args=None):
    rclpy.init(args=args)

    dummy_server = DummyServer()

    rclpy.spin(dummy_server)

    rclpy.shutdown()

if __name__ == '__main__':
    main()