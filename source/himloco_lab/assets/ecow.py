# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Ecow robots.
"""

import os
import math
import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg


from isaaclab.utils import configclass

from himloco_lab.assets import ecow_actuators

ECOW_MODEL_DIR = "/home/panda/qiu/HimLoco-AMP-Lab/ecow/ecow"


@configclass
class EcowArticulationCfg(ArticulationCfg):
    """Configuration for Ecow articulations."""

    joint_sdk_names: list[str] = None

    soft_joint_pos_limit_factor = 0.9


@configclass
class EcowUsdFileCfg(sim_utils.UsdFileCfg):
    activate_contact_sensors: bool = True
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
    )


@configclass
class EcowUrdfFileCfg(sim_utils.UrdfFileCfg):
    fix_base: bool = False
    activate_contact_sensors: bool = True
    replace_cylinders_with_capsules = True
    joint_drive = sim_utils.UrdfConverterCfg.JointDriveCfg(
        gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4,
    )
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )

    def replace_asset(self, meshes_dir, urdf_path):
        """Replace the asset with a temporary copy to avoid modifying the original asset.

        When need to change the collisions, place the modified URDF file separately in this repository,
        This function will auto construct a complete `robot_description` file structure in the `/tmp` directory.
        Note: The mesh references inside the URDF should be in the same directory level as the URDF itself.
        """
        tmp_meshes_dir = "/tmp/IsaacLab/ecow_rl_lab/meshes"
        if os.path.exists(tmp_meshes_dir):
            os.remove(tmp_meshes_dir)
        os.makedirs("/tmp/IsaacLab/ecow_rl_lab", exist_ok=True)
        os.symlink(meshes_dir, tmp_meshes_dir)

        self.asset_path = "/tmp/IsaacLab/ecow_rl_lab/robot.urdf"
        if os.path.exists(self.asset_path):
            os.remove(self.asset_path)
        os.symlink(urdf_path, self.asset_path)


""" Configuration for the Ecow robots."""
""" Kp=Iw2  Kd=2*e*w*I-B   w=10*2*pi w2=3947.84162575  w=6*2*pi w2=1421.22298527"""
ECOW_CFG = EcowArticulationCfg(
    spawn=EcowUrdfFileCfg(
        asset_path=f"{ECOW_MODEL_DIR}/urdf/ecow.urdf",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        # joint_pos={
        #     ".*R_hip_joint": 0.2, #0.1
        #     ".*L_hip_joint": -0.2, #-0.1
        #     "FL_thigh_joint": -0.55,
        #     "FR_thigh_joint": -0.55,
        #     "RL_thigh_joint": 0.55,
        #     "RR_thigh_joint": 0.55,
        #     "FL_calf_joint": 1.22,
        #     "FR_calf_joint": 1.22,
        #     "RL_calf_joint": -1.22,
        #     "RR_calf_joint": -1.22,
        # },
        joint_pos={
            ".*R_hip_joint": 0.2, #0.25
            ".*L_hip_joint": -0.2, #-0.25
            "FL_thigh_joint": -0.7,
            "FR_thigh_joint": -0.7,
            "RL_thigh_joint": 0.7,
            "RR_thigh_joint": 0.7,
            "FL_calf_joint": 1.4,
            "FR_calf_joint": 1.4,
            "RL_calf_joint": -1.4,
            "RR_calf_joint": -1.4,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "hip": ecow_actuators.EcowActuatorCfg_HIP(
            joint_names_expr=[".*_hip_joint"],
            stiffness=1500.0,
            damping=50,
            friction=0.01,
        ),
        "thigh": ecow_actuators.EcowActuatorCfg_THIGH(
            joint_names_expr=[".*_thigh_joint"],
            stiffness=1500.0,
            damping=50,
            friction=0.01,
        ),
        "calf": ecow_actuators.EcowActuatorCfg_CALF(
            joint_names_expr=[".*_calf_joint"],
            stiffness=3000.0,
            damping=80,
            friction=0.01,
        ),
    },
    # actuators={
    #     "legs": DCMotorCfg(
    #         joint_names_expr=[".*_joint"],
    #         effort_limit=33.5,
    #         saturation_effort=33.5,
    #         velocity_limit=21.0,
    #         stiffness=20.0,
    #         damping=0.5,
    #         friction=0.0,
    #     ),
    # },
    # fmt: off
    joint_sdk_names=[
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"
    ],
    # fmt: on
)

