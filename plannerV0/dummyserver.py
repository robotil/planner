import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from diagnostic_msgs.msg import DiagnosticStatus
from actionlib_msgs.msg import GoalID, GoalStatus, GoalStatusArray
from geometry_msgs.msg import PointStamped, PolygonStamped, Twist, TwistStamped, PoseStamped, Point
from planner_msgs.msg import SDiagnosticStatus, SGlobalPose, SHealth, SImu, EnemyReport, OPath
from planner_msgs.srv import ActGeneralAdmin, StateGeneralAdmin, CheckLOS, AllPathEntityToTarget

class DummyServer(Node):
    def __init__(self):
        super().__init__('dummy_server')
        self.state = bytes([0])
        self.actAdminSrv = self.create_service(ActGeneralAdmin, 'act_general_admin', self.act_general_admin_callback)
        self.stateAdminSrv = self.create_service(StateGeneralAdmin, 'state_general_admin', self.state_general_admin_callback)
        self.checkLOSSrv = self.create_service(CheckLOS, 'check_line_of_sight', self.check_line_of_sight_callback)
        self.getAllPathSrv = self.create_service(AllPathEntityToTarget, 'get_all_possible_ways', self.get_all_possible_ways_callback)

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

    def get_all_possible_ways_callback(self, request, response):
        entity = request.entityid
        target = request.target
        self.get_logger().info('Got request for entity ='+entity+' and target ='+target.__str__())
        p1 = Point()
        p1.x = p1.y = 0.1
        p1.z = 0.0
        p2 = Point(x=0.2, y=0.2, z=0.2)
        path1 = OPath()
        path1.point.append(p1)
        path1.point.append(p2)
        path1.name = "maxim"
        pathes = []
        pathes.append(path1)
        response.path = pathes
        return response

def main(args=None):
    rclpy.init(args=args)

    dummy_server = DummyServer()

    rclpy.spin(dummy_server)

    rclpy.shutdown()

if __name__ == '__main__':
    main()