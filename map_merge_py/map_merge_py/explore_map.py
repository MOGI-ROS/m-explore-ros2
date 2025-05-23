import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from rosgraph_msgs.msg import Clock

from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose

import numpy as np


class MultiRobotExplorer(Node):
    def __init__(self):
        super().__init__('multi_robot_explorer')

        # sim_time parameter
        self.declare_parameter('sim_time', True)
        self.sim_time = self.get_parameter('sim_time').get_parameter_value().bool_value

        self.latest_clock = None
        if self.sim_time:
            self.create_subscription(Clock, '/clock', self.clock_callback, 10)

        # TF buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Map subscriptions
        self.create_subscription(OccupancyGrid, '/map', self.global_map_callback, 10)
        self.create_subscription(OccupancyGrid, '/robot_1/map', self.robot1_map_callback, 10)
        self.create_subscription(OccupancyGrid, '/robot_2/map', self.robot2_map_callback, 10)

        # Goal publishers
        self.pub_1 = self.create_publisher(PoseStamped, '/robot_1/goal_pose', 10)
        self.pub_2 = self.create_publisher(PoseStamped, '/robot_2/goal_pose', 10)

        # Exploration timer
        self.timer = self.create_timer(2.0, self.explore)

        # Map holders
        self.global_map = None
        self.local_map_1 = None
        self.local_map_2 = None

        self.get_logger().info(f"Multi-robot explorer initialized (sim_time={self.sim_time})")

    def clock_callback(self, msg):
        self.latest_clock = msg.clock

    def global_map_callback(self, msg):
        self.global_map = msg

    def robot1_map_callback(self, msg):
        self.local_map_1 = msg

    def robot2_map_callback(self, msg):
        self.local_map_2 = msg

    def explore(self):
        if not self.global_map or not self.local_map_1 or not self.local_map_2:
            self.get_logger().warn("Waiting for all maps...")
            return

        global_frontiers = self.find_frontiers(self.global_map)
        self.get_logger().info(f"Global frontiers: {len(global_frontiers)}")

        if not global_frontiers:
            self.get_logger().info("No frontiers left in global map. Exploration complete.")
            self.timer.cancel()
            return

        local_frontiers_1 = self.find_frontiers(self.local_map_1)
        local_frontiers_2 = self.find_frontiers(self.local_map_2)

        self.get_logger().info(
            f"Robot 1 frontiers: {len(local_frontiers_1)} | Robot 2 frontiers: {len(local_frontiers_2)}"
        )

        if local_frontiers_1:
            self.send_goal(local_frontiers_1[0], self.local_map_1, 'robot_1/map', self.pub_1)
        if local_frontiers_2:
            self.send_goal(local_frontiers_2[0], self.local_map_2, 'robot_2/map', self.pub_2)

        if not local_frontiers_1 and not local_frontiers_2:
            self.get_logger().info("Frontiers exist globally, but none are visible to either robot.")

    def send_goal(self, cell, map_msg, source_frame, pub):
        y, x = cell
        resolution = map_msg.info.resolution
        origin = map_msg.info.origin.position

        # Create goal in local frame
        local_goal = PoseStamped()
        local_goal.header.frame_id = source_frame
        local_goal.header.stamp = self.latest_clock if self.sim_time and self.latest_clock else self.get_clock().now().to_msg()
        local_goal.pose.position.x = origin.x + (x + 0.5) * resolution
        local_goal.pose.position.y = origin.y + (y + 0.5) * resolution
        local_goal.pose.position.z = 0.0
        local_goal.pose.orientation.w = 1.0

        # Transform to world frame
        try:
            transform = self.tf_buffer.lookup_transform(
                'world',
                source_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5)
            )
            world_goal = PoseStamped()
            world_goal.pose = do_transform_pose(local_goal.pose, transform)
            world_goal.header.stamp = local_goal.header.stamp  # preserve correct stamp
            world_goal.header.frame_id = 'world'
            pub.publish(world_goal)

            self.get_logger().info(
                f"Sent goal to {pub.topic} at ({world_goal.pose.position.x:.2f}, {world_goal.pose.position.y:.2f}) in world frame"
            )
        except Exception as e:
            self.get_logger().warn(f"TF transform failed from {source_frame} to world: {e}")

    def find_frontiers(self, map_msg):
        height = map_msg.info.height
        width = map_msg.info.width
        data = np.array(map_msg.data, dtype=np.int8).reshape((height, width))
        frontiers = []

        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if data[y, x] != 0:
                    continue
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if data[ny, nx] == -1:
                            frontiers.append((y, x))
                            break
                    else:
                        continue
                    break

        return frontiers


def main(args=None):
    rclpy.init(args=args)
    node = MultiRobotExplorer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()