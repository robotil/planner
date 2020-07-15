import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from diagnostic_msgs.msg import DiagnosticStatus
from actionlib_msgs.msg import GoalID, GoalStatus, GoalStatusArray
from geometry_msgs.msg import PointStamped, PolygonStamped, Twist, TwistStamped, PoseStamped

class WorldCom(Node):

    def __init__(self):
        super().__init__('world_communication')
        self.world_state = {}
        self.subscription = self.create_subscription(
            PoseStamped,
            '/entity/id1/pose',
            self.pose_callback,
            10)
        self.subscription  # prevent unused variable warning

        self.publisher_ = self.create_publisher(PoseStamped, '/entity/id1/move_to/goal', 10)
        timer_period = 10  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 2


    def timer_callback(self):
        msg = PoseStamped()
        msg.pose.position.x = 1/self.i
        msg.pose.position.y = 0.1/self.i
        msg.pose.position.z = 0.2/self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Sending: "%s"' % msg.pose)
        self.i += 1

    def pose_callback(self, msg):
        self.get_logger().info('Received: "%s"' % msg.pose)


def main(args=None):
    rclpy.init(args=args)

    world_communication = WorldCom()

    rclpy.spin(world_communication)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    world_communication.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()