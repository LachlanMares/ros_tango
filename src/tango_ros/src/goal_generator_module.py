"""
Author:
    Stefan Podgorski, stefan.podgorski@adelaide.edu.au

License:
    GPL-3.0

Description:

"""

import os
import numpy as np
import cv2
import pickle

import matplotlib.pyplot as plt

from libs.segmentor import sam
from libs.localizer import loc_topo
from libs.planner_global import plan_topo
from libs.commons import utils, utils_data, utils_viz


class GoalGeneratorClass:
    def __init__(self, settings_dict: dict):
        self.width = settings_dict['width']
        self.H = H
        self.G = None
        self.nodeID_to_imgRegionIdx = None
        self.segmentor = sam.Seg_SAM(modelPath, device)
        self.localizer = None
        self.goalNodeIdx = 0
        self.planner_g = None

    def load_episode(self, mapPath):
        self.G = pickle.load(open(f"{mapPath}/nodes_graphObject_4.pickle", 'rb'))
        utils.change_edge_attr(self.G)

        self.nodeID_to_imgRegionIdx = np.array([self.G.nodes[node]['map'] for node in self.G.nodes()])

        self.localizer = loc_topo.Localize_Topological(f"{mapPath}/images", self.G, self.W, self.H)

        self.goalNodeIdx = utils_data.get_goalNodeIdx(mapPath, self.G, self.nodeID_to_imgRegionIdx)

        self.planner_g = plan_topo.Plan_Topological(self.G, self.goalNodeIdx)

    def get_goal_mask(self, qryImg):
        self.qryNodes = self.segmentor.segment(qryImg, False, self.W, self.H)
        self.qryMasks = utils.nodes2key(self.qryNodes, 'segmentation')
        self.qryCoords = utils.nodes2key(self.qryNodes, 'coords')

        self.matchPairs = self.localizer.localize(qryImg, self.qryNodes)

        self.pls, nodesClose2Goal = self.planner_g.get_pathLengths_matchedNodes(self.matchPairs[:, 1])

        self.goalMask = 100 * np.ones((self.H, self.W))  # default value for invalid goal segments
        for i in range(len(self.pls)):
            self.goalMask[self.qryMasks[self.matchPairs[i, 0]]] = self.pls[i]
        return self.goalMask

    def visualize_goal_mask(self, qryImg, display=False):
        colors, norm = utils_viz.value2color(self.pls, cmName='viridis')
        vizImg = utils_viz.drawMasksWithColors(qryImg, self.qryMasks[self.matchPairs[:, 0]], colors)
        if display:
            plt.imshow(vizImg)
            plt.colorbar()
            plt.show()
        return vizImg


if __name__ == "__main__":
    modelPath = f"{os.path.expanduser('~')}/workspace/s/sg_habitat/models/segment-anything/"
    goalie = Goal_Gen(modelPath, W=160, H=120)

    mapPath = f"{os.path.expanduser('~')}/fastdata/navigation/hm3d_iin_train/1S7LAXRdDqK_0000000_plant_42_/"

    goalie.load_episode(mapPath)

    qryImg = cv2.imread(f"{mapPath}/images/00000.png")[:, :, ::-1]

    goalMask = goalie.get_goal_mask(qryImg)
    _ = goalie.visualize_goal_mask(qryImg, display=True)