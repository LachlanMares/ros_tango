"""
Author:
    Stefan Podgorski, stefan.podgorski@adelaide.edu.au

License:
    GPL-3.0

Description:

"""

import networkx as nx
import torch
import numpy as np
import kornia as K
from typing import Iterator, Tuple
from scipy.interpolate import splrep, BSpline
from kornia import morphology as morph
import cv2


class CostMapGraphNX:

    def __init__(self, width: int, height: int, cost_map: np.ndarray):
        self.width = width
        self.height = height
        self.cost_map = cost_map - cost_map.min()
        self.graph = self.build_graph()

    def build_graph(self):
        graph = nx.Graph()
        xs, ys = np.meshgrid(range(self.width), range(self.height))
        xs = xs.reshape(-1)
        ys = ys.reshape(-1)
        for x, y in zip(xs, ys):
            for neighbours in self.neighbours((x, y)):
                x_neighbour, y_neighbour = neighbours
                cost = self.cost_map[y_neighbour, x_neighbour]
                graph.add_edges_from([(f'{x},{y}', f'{x_neighbour},{y_neighbour}')], weight=cost)
        return graph

    def in_bounds(self, coordinates: tuple) -> bool:
        (x, y) = coordinates
        return 0 <= x < self.width and 0 <= y < self.height

    def neighbours(self, coordinates: tuple) -> Iterator[tuple]:
        (x, y) = coordinates

        neighbours = [
            (x, y + 1),
            (x + 1, y),
            (x, y - 1),
            (x - 1, y),
            (x + 1, y + 1),
            (x + 1, y - 1),
            (x - 1, y - 1),
            (x - 1, y + 1),
        ]
        results = filter(self.in_bounds, neighbours)
        return results

    def get_path(self, start: tuple, goal: tuple) -> np.ndarray:
        shortest_path = [
            coord.split(',') for coord in nx.bidirectional_dijkstra(
                self.graph, f'{start[0]},{start[1]}', f'{goal[0]},{goal[1]}',
                weight='weight'
            )[1]]

        shortest_path = np.array(shortest_path, dtype=float)
        return shortest_path


class GoalControl:
    @torch.inference_mode
    def __init__(self, pid_steer, traversable_classes, default_velocity_control: float,
                 h_image: int, w_image: int, hfov_rads: float, cam_height=3.5, intrinsics: torch.Tensor = torch.eye(3),
                 grid_size: float = 0.1, device: str = 'cpu'):
        self.default_velocity_control = default_velocity_control
        self.traversable_classes = traversable_classes  # [25, 30, 37]
        self.device = device
        self.pid_steer = pid_steer
        # camera stuff
        self.intrinsics = intrinsics.to(self.device)
        self.intrinsics_inv = torch.linalg.inv(self.intrinsics)
        self.hfov_rads = hfov_rads
        self.blind_distance = ((cam_height / np.tan(hfov_rads)) / grid_size).astype(int)
        print(self.blind_distance)
        # perspective image
        self.h_image, self.w_image = h_image, w_image
        u = torch.arange(0, self.w_image, device=self.device)
        v = torch.arange(0, self.h_image, device=self.device)
        vs, us = torch.meshgrid(v, u)
        us = us.reshape(-1)
        vs = vs.reshape(-1)
        self.homogeneous_pts = torch.concatenate(
            (us[..., None],
             vs[..., None],
             torch.ones(size=(us.shape[0], 1), device=us.device)), dim=1
        ).T

        # bev occupancy grid (x, y, z): (+-5m, +-5m, 0-10m)
        self.grid_size = grid_size
        self.grid_min = torch.tensor([-5, -0.75, 0], device=self.device)
        self.grid_max = torch.tensor([5, 50, 10], device=self.device)
        self.cells = ((self.grid_max - self.grid_min) / self.grid_size).to(int)
        self.w_bev, self.h_bev = self.cells[2].item(), self.cells[0].item()

        self.grid_shift = torch.tensor([self.w_bev // 2, -self.blind_distance], dtype=int, device=self.device)  # -8
        self.start_bev = (self.w_bev // 2, 0)
        self.x_bev_range = torch.arange(
            self.grid_min[0].item(), self.grid_max[0].item(), self.grid_size
        ).round(decimals=3)
        self.z_bev_range = torch.arange(
            self.grid_min[2].item(), self.grid_max[2].item(), self.grid_size
        ).round(decimals=3)
        self.kernel_erode = torch.ones((3, 3), device=self.device)
        self.occupied = torch.zeros(
            self.h_bev, self.w_bev,
            device=self.device, dtype=torch.long
        )
        self.free = torch.zeros(
            self.h_bev, self.w_bev,
            device=self.device, dtype=torch.long
        )
        # end point buffer
        self.goal_distance_buffer = []
        self.goal_distance_buffer_limit = 3
        self.goal_distance_limit = 10  # 3.5 / self.grid_size
        self.buffer_counter = 0
        self.stop_flag = False

        # theta buffer
        self.theta_buffer = []
        self.theta_buffer_limit = 5
        self.theta_buffer_counter = 0

    @torch.inference_mode
    def compute_goal_point(self, depth: torch.Tensor, goal_point) -> torch.Tensor:
        pixel_goal = goal_point[0, :]

        homogeneous_pts = torch.ones(3, device=self.device)
        homogeneous_pts[0] = pixel_goal[1]
        homogeneous_pts[1] = pixel_goal[0]
        unprojected_point = self.unproject_points(
            depth[pixel_goal[0], pixel_goal[1]],
            intrinsics_inv=self.intrinsics_inv,
            homogeneous_pts=homogeneous_pts
        )
        point_goal_bev = torch.floor(unprojected_point / self.grid_size).to(int)[0::2] + self.grid_shift
        point_goal_bev[0] = point_goal_bev[0].clip(0, self.w_bev - 1)
        point_goal_bev[1] = point_goal_bev[1].clip(0, self.h_bev - 1)
        return point_goal_bev

    @staticmethod
    def unproject_points(depth: torch.Tensor, intrinsics_inv, homogeneous_pts) -> torch.Tensor:
        unprojected_points = (torch.matmul(intrinsics_inv.to(torch.double), homogeneous_pts.to(torch.double))).T
        unprojected_points *= depth
        return unprojected_points

    @torch.inference_mode
    def compute_relative_bev(self,
                             traversable: torch.Tensor,
                             depth: torch.Tensor) -> torch.Tensor:

        unprojected_points = self.unproject_points(depth, self.intrinsics_inv, self.homogeneous_pts)
        upper = unprojected_points < self.grid_max
        lower = unprojected_points > self.grid_min
        mask_in_range = torch.logical_and(lower, upper).all(1)
        unprojected_points = unprojected_points[mask_in_range]
        traversable = traversable[mask_in_range]

        voxels = torch.floor(unprojected_points / self.grid_size).to(int)
        xy_t_ij = torch.concatenate((voxels[:, 0][:, None], voxels[:, 2][:, None]), dim=1) + self.grid_shift

        self.occupied.zero_()
        self.free.zero_()
        self.occupied = (self.occupied.to(int).index_put_(
            (xy_t_ij[:, 1], xy_t_ij[:, 0]),
            torch.logical_not(traversable).to(int),
            accumulate=True) > 0).int()
        self.free = (self.free.to(int).index_put_(
            (xy_t_ij[:, 1], xy_t_ij[:, 0]), traversable.to(int),
            accumulate=True) > 0).int()
        occupancy = (self.free - self.occupied).clip(0, 1)
        return occupancy.float()

    @staticmethod
    def compute_point_tangents(points: np.ndarray) -> np.ndarray:
        point_next = np.roll(points, axis=0, shift=-1)
        point_diff = point_next - points
        xs, zs = point_diff[:, 0], point_diff[:, 1]
        thetas = np.arctan2(xs, zs)  # estimate tangents with points in front
        thetas[-1] = thetas[-2]  # estimate tangent from previous point
        thetas = np.roll(thetas, axis=0, shift=1)  # make sure we aim at the next point
        thetas[-1] = thetas[0]  # we dont know which way to face because we have no next point
        thetas[0] = 0  # initially facing forward
        return thetas[..., None]

    def get_point_poses_numpy(self, path_traversable_bev: np.ndarray) -> np.ndarray:
        skips = 2
        traversable_bev_xs = self.x_bev_range[path_traversable_bev[:, 0]]
        traversable_bev_zs = path_traversable_bev[:, 1] * self.grid_size
        if path_traversable_bev.shape[0] > skips:
            traversable_bev_xs = traversable_bev_xs[::skips]
            traversable_bev_zs = traversable_bev_zs[::skips]
        try:
            t = np.concatenate(
                (np.array([0]),
                 np.cumsum(np.diff(traversable_bev_xs, 1) ** 2 + np.diff(traversable_bev_zs, 1) ** 2))
            ) / traversable_bev_xs.shape[0]
            ti = np.linspace(0, t[-1], 20)
            tck_x = splrep(t, traversable_bev_xs, s=0)
            tck_z = splrep(t, traversable_bev_zs, s=0)
            traversable_bev_xs = BSpline(*tck_x)(ti)
            traversable_bev_zs = BSpline(*tck_z)(ti)
        except TypeError:
            pass  # sometimes things just don't go to plan so default to janky paths
        traversable_bev = np.concatenate((-1 * traversable_bev_xs[:, None], traversable_bev_zs[:, None]), axis=1)
        thetas = self.compute_point_tangents(traversable_bev)
        point_poses = np.concatenate((traversable_bev, thetas), axis=1)
        return point_poses  # (x, z, theta)

    def add_safety_margin(self, traversable: torch.Tensor) -> torch.Tensor:
        traversable_with_margin = morph.dilation(traversable[None, None, ...], 2 * self.kernel_erode)
        traversable_with_margin = morph.erosion(traversable_with_margin, 2 * self.kernel_erode)
        return traversable_with_margin

    @staticmethod
    def check_if_traversable(traversable_relative_bev: torch.Tensor) -> bool:
        is_traversable = ((500 - traversable_relative_bev) > 0).sum() > 10
        return is_traversable
        # return True

    def control(self,
                depth: torch.Tensor,
                traversable_perspective: torch.Tensor,
                goal_mask: torch.Tensor, goal_point,
                time_delta: float) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
        # default values to be overridden if all is good
        point_poses = np.array([[0., 0.], [0., 0.]])
        theta_control = 0.
        velocity_control = 0.
        h, w = traversable_perspective.shape
        traversable_perspective[h - 7:, :] = 255
        depth = depth.to(self.device)
        point_goal_bev = self.compute_goal_point(
            depth=depth,
            goal_point=goal_point
        )
        depth = depth.reshape(-1)[:, None].repeat(1, 3)

        traversable_relative_bev = self.compute_relative_bev(
            traversable=traversable_perspective.reshape(-1),
            depth=depth
        )

        # add a margin around the non-traversable objects
        traversable_relative_bev_safe = self.add_safety_margin(traversable_relative_bev)
        cost_map_relative_bev_safe = 1-K.contrib.distance_transform(
            1-traversable_relative_bev_safe, kernel_size=3
        )
        cost_map_relative_bev_safe = K.filters.box_blur(
            cost_map_relative_bev_safe, (5, 5),
        ).squeeze(0, 1)  # soften edges to help keep robot from hitting wall
        cost_scaler = 500
        # occupancy edges
        h, w = cost_map_relative_bev_safe.shape
        traversable_relative_bev_safe[:, :, :6, w // 2 - 4:w // 2 + 4] = 1
        edges = cv2.Canny(traversable_relative_bev_safe.cpu().squeeze(0, 1).numpy().astype(np.uint8), 0, 1)
        edges[:6, w // 2 - 4:w // 2 + 4] = 0
        edges = (cv2.blur(edges, ksize=(2, 2)) > 0)

        edges = edges.astype(np.uint8) * cost_scaler
        cost_map_relative_bev_safe /= cost_map_relative_bev_safe.max()  # scale 0-1
        cost_map_relative_bev_safe *= cost_scaler

        cost_map_relative_bev_safe = cost_map_relative_bev_safe.cpu().numpy() + edges
        # cost_map_relative_bev_safe = np.clip(cost_map_relative_bev_safe, 0, cost_scaler)
        # get the furthest Euclidean goal point
        if self.check_if_traversable(cost_map_relative_bev_safe):
            goal_x = point_goal_bev[0].item()
            # goal_y = torch.where(traversable_relative_bev_safe.squeeze(0, 1) == 1)[0].max().cpu().item()
            goal_y = point_goal_bev[1].item()
            goal_bev = (goal_x, goal_y)
            h, w = cost_map_relative_bev_safe.shape
            # find a path in the cost map
            cmg = CostMapGraphNX(
                width=w, #self.w_bev,
                height=h, #self.h_bev,
                cost_map=cost_map_relative_bev_safe
            )
            path_traversable_bev = cmg.get_path(self.start_bev, goal_bev)
            # stop condition
            goal_stop_dist = goal_y
            if len(self.goal_distance_buffer) < self.goal_distance_limit:
                self.goal_distance_buffer.append(goal_stop_dist)
            else:
                self.goal_distance_buffer[self.buffer_counter] = goal_stop_dist
            self.buffer_counter += 1
            self.buffer_counter = self.buffer_counter % self.goal_distance_buffer_limit
            goal_y_dist = np.median(np.array(self.goal_distance_buffer))
            print(self.goal_distance_limit, goal_y_dist, goal_y, self.stop_flag)
            if goal_y_dist <= self.goal_distance_limit:
                self.stop_flag = True
                velocity_control = 0
            else:
                if path_traversable_bev.shape[0] > 0:
                    point_poses = self.get_point_poses_numpy(path_traversable_bev)  # [1:]
                    # print(point_poses)
                    # find the theta control signal: thetaj current pose, thetai target pose
                    thetaj, thetai = 0, point_poses[3:5, 2].mean()
                    theta_control = self.pid_steer.control(
                        value_goal=thetai,
                        value_actual=-thetaj,
                        time_delta=time_delta
                    )
                    # print(self.stop_flag)
                    if not self.stop_flag:
                        velocity_control = self.default_velocity_control
                    else:
                        velocity_control = 0
            traversable_relative_bev_safe = traversable_relative_bev_safe.squeeze(0, 1).cpu().numpy()
        else:
            traversable_relative_bev_safe = torch.zeros_like(self.occupied).cpu().numpy()
            cost_map_relative_bev_safe = torch.zeros_like(self.occupied).cpu().numpy()
            theta_control = 0
            velocity_control = 0
        if len(self.theta_buffer) < self.theta_buffer_limit:
            self.theta_buffer.append(theta_control)
        else:
            self.theta_buffer[self.theta_buffer_counter] = theta_control
            self.theta_buffer_counter += 1
            self.theta_buffer_counter = self.theta_buffer_counter % self.theta_buffer_limit
        return velocity_control, -theta_control, traversable_relative_bev_safe, cost_map_relative_bev_safe, point_poses, goal_bev
