import os
import time
import mujoco as mj
import mujoco.viewer as mjv
import imageio
from scipy.spatial.transform import Rotation as R
from general_motion_retargeting import ROBOT_XML_DICT, ROBOT_BASE_DICT, VIEWER_CAM_DISTANCE_DICT
from loop_rate_limiters import RateLimiter
import numpy as np
from rich import print
import threading
import traceback


def draw_frame(
    pos,
    mat,
    v,
    size,
    joint_name=None,
    orientation_correction=R.from_euler("xyz", [0, 0, 0]),
    pos_offset=np.array([0, 0, 0]),
):
    """Draw coordinate frame with better error handling"""
    try:
        rgba_list = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
        for i in range(3):
            if v.user_scn.ngeom >= v.user_scn.maxgeom:
                print(f"Warning: Maximum number of geometries reached ({v.user_scn.maxgeom})")
                break
                
            geom = v.user_scn.geoms[v.user_scn.ngeom]
            mj.mjv_initGeom(
                geom,
                type=mj.mjtGeom.mjGEOM_ARROW,
                size=[0.01, 0.01, 0.01],
                pos=pos + pos_offset,
                mat=mat.flatten(),
                rgba=rgba_list[i],
            )
            if joint_name is not None:
                geom.label = joint_name
            fix = orientation_correction.as_matrix()
            print(f'orientation_matrix = {fix}')
            mj.mjv_connector(
                v.user_scn.geoms[v.user_scn.ngeom],
                type=mj.mjtGeom.mjGEOM_ARROW,
                width=0.005,
                from_=pos + pos_offset,
                to=pos + pos_offset + size * (mat @ fix)[:, i],
            )
            v.user_scn.ngeom += 1
    except Exception as e:
        print(f"Error in draw_frame: {e}")


class RobotMotionViewer:
    def __init__(self,
                robot_type,
                camera_follow=True,
                motion_fps=30,
                transparent_robot=0,
                # video recording
                record_video=False,
                video_path=None,
                video_width=640,
                video_height=480,
                # stability options (new parameters with defaults)
                max_step_failures=10,
                viewer_timeout=30.0,
                **kwargs):
        
        self.robot_type = robot_type
        self.motion_fps = motion_fps
        self.camera_follow = camera_follow
        self.record_video = record_video
        self.max_step_failures = max_step_failures
        self.step_failure_count = 0
        self.viewer_timeout = viewer_timeout
        self.is_closed = False
        self.last_successful_step = time.time()
        
        # Thread safety
        self._lock = threading.Lock()
        
        try:
            # Initialize MuJoCo model
            self.xml_path = ROBOT_XML_DICT[robot_type]
            self.model = mj.MjModel.from_xml_path(str(self.xml_path))
            self.data = mj.MjData(self.model)
            self.robot_base = ROBOT_BASE_DICT[robot_type]
            self.viewer_cam_distance = VIEWER_CAM_DISTANCE_DICT[robot_type]
            
            # Initial simulation step
            mj.mj_step(self.model, self.data)
            print(f"Model loaded successfully. DOF: {self.model.nq}")
            
        except Exception as e:
            print(f"Error loading robot model: {e}")
            raise
        
        try:
            # Initialize rate limiter
            self.rate_limiter = RateLimiter(frequency=self.motion_fps, warn=False)
            
            # Initialize viewer
            print("Initializing MuJoCo viewer...")
            self.viewer = mjv.launch_passive(
                model=self.model,
                data=self.data,
                show_left_ui=False,
                show_right_ui=False
            )
            
            if self.viewer is None:
                raise RuntimeError("Failed to create MuJoCo viewer")
                
            self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = transparent_robot
            print("Viewer initialized successfully")
            
        except Exception as e:
            print(f"Error initializing viewer: {e}")
            raise
        
        # Initialize video recording if requested
        if self.record_video:
            try:
                self._init_video_recording(video_path, video_width, video_height)
            except Exception as e:
                print(f"Error initializing video recording: {e}")
                self.record_video = False
    
    def _init_video_recording(self, video_path, video_width, video_height):
        """Initialize video recording with error handling"""
        assert video_path is not None, "Please provide video path for recording"
        self.video_path = video_path
        video_dir = os.path.dirname(self.video_path)
        
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
            
        self.mp4_writer = imageio.get_writer(
            self.video_path, 
            fps=self.motion_fps,
            macro_block_size=None  # Better compatibility
        )
        print(f"Recording video to {self.video_path}")
        
        # Initialize renderer for video recording
        self.renderer = mj.Renderer(
            self.model, 
            height=video_height, 
            width=video_width
        )
    
    def _validate_inputs(self, root_pos, root_rot, dof_pos):
        """Validate input data to prevent crashes"""
        try:
            # Check dimensions
            if len(root_pos) != 3:
                raise ValueError(f"root_pos must have 3 elements, got {len(root_pos)}")
            if len(root_rot) != 4:
                raise ValueError(f"root_rot must have 4 elements (quaternion), got {len(root_rot)}")
            
            expected_dof = self.model.nq - 7  # Total DOF minus root position and orientation
            if len(dof_pos) != expected_dof:
                raise ValueError(f"dof_pos must have {expected_dof} elements, got {len(dof_pos)}")
            
            # Check for NaN or infinite values
            if not np.all(np.isfinite(root_pos)):
                raise ValueError("root_pos contains NaN or infinite values")
            if not np.all(np.isfinite(root_rot)):
                raise ValueError("root_rot contains NaN or infinite values")
            if not np.all(np.isfinite(dof_pos)):
                raise ValueError("dof_pos contains NaN or infinite values")
            
            # Check quaternion normalization
            quat_norm = np.linalg.norm(root_rot)
            if abs(quat_norm - 1.0) > 0.1:
                print(f"Warning: Quaternion not normalized (norm={quat_norm:.3f}), normalizing...")
                root_rot = root_rot / quat_norm
            
            return root_pos, root_rot, dof_pos
            
        except Exception as e:
            print(f"Input validation failed: {e}")
            raise
    
    def step(self, 
            # robot data
            root_pos, root_rot, dof_pos, 
            # human data
            human_motion_data=None, 
            show_human_body_name=False,
            # scale for human point visualization
            human_point_scale=0.1,
            # human pos offset add for visualization    
            human_pos_offset=np.array([0.0, 0.0, 0]),
            # rate limit
            rate_limit=True, 
            follow_camera=True,
            ):
        """
        Improved step function with better error handling and stability
        """
        
        if self.is_closed:
            print("Viewer is closed, cannot step")
            return False
        
        with self._lock:
            try:
                # Check for viewer timeout
                current_time = time.time()
                if current_time - self.last_successful_step > self.viewer_timeout:
                    print(f"Viewer timeout after {self.viewer_timeout}s, reinitializing...")
                    self._reinitialize_viewer()
                
                # Validate inputs
                root_pos, root_rot, dof_pos = self._validate_inputs(root_pos, root_rot, dof_pos)
                
                # Check if viewer is still alive (simple check)
                if self.is_closed:
                    print("Viewer is closed, cannot continue")
                    return False
                
                # Update simulation state
                self.data.qpos[:3] = root_pos
                self.data.qpos[3:7] = root_rot  # quaternion (scalar first for MuJoCo)
                self.data.qpos[7:] = dof_pos
                

                mj.mj_forward(self.model, self.data)
                
                # Update camera
                if follow_camera and self.camera_follow:
                    try:
                        self.viewer.cam.lookat = self.data.xpos[self.model.body(self.robot_base).id]
                        self.viewer.cam.distance = self.viewer_cam_distance
                        self.viewer.cam.elevation = -10
                    except Exception as e:
                        print(f"Camera update error: {e}")
                
                # Handle human motion visualization
                if human_motion_data is not None:
                    try:
                        # Clean custom geometry (with bounds checking)
                        self.viewer.user_scn.ngeom = 0
                        
                        # Draw human motion frames
                        for human_body_name, (pos, rot) in human_motion_data.items():
                            if self.viewer.user_scn.ngeom >= self.viewer.user_scn.maxgeom - 3:
                                print("Warning: Reached maximum geometry limit for human visualization")
                                break
                                
                            draw_frame(
                                pos,
                                R.from_quat(rot, scalar_first=True).as_matrix(),
                                self.viewer,
                                human_point_scale,
                                pos_offset=human_pos_offset,
                                joint_name=human_body_name if show_human_body_name else None
                            )
                    except Exception as e:
                        print(f"Error in human motion visualization: {e}")
                
                # Sync viewer
                self.viewer.sync()
                
                # Handle video recording
                if self.record_video:
                    try:
                        self.renderer.update_scene(self.data, camera=self.viewer.cam)
                        img = self.renderer.render()
                        self.mp4_writer.append_data(img)
                    except Exception as e:
                        print(f"Video recording error: {e}")
                        self.record_video = False  # Disable recording on error
                
                # Rate limiting
                if rate_limit:
                    self.rate_limiter.sleep()
                
                # Reset failure count on success
                self.step_failure_count = 0
                self.last_successful_step = current_time
                return True
                
            except Exception as e:
                self.step_failure_count += 1
                print(f"Error in viewer step (failure {self.step_failure_count}/{self.max_step_failures}): {e}")
                traceback.print_exc()
                
                if self.step_failure_count >= self.max_step_failures:
                    print("Maximum step failures reached, closing viewer")
                    self.close()
                    return False
                
                return False
    
    def _reinitialize_viewer(self):
        """Attempt to reinitialize the viewer if it becomes unresponsive"""
        try:
            print("Attempting to reinitialize viewer...")
            if hasattr(self, 'viewer') and self.viewer is not None:
                self.viewer.close()
                time.sleep(0.5)
            
            self.viewer = mjv.launch_passive(
                model=self.model,
                data=self.data,
                show_left_ui=False,
                show_right_ui=False
            )
            
            if self.viewer is None:
                raise RuntimeError("Failed to reinitialize viewer")
                
            self.step_failure_count = 0
            self.last_successful_step = time.time()
            print("Viewer reinitialized successfully")
            
        except Exception as e:
            print(f"Failed to reinitialize viewer: {e}")
            self.is_closed = True
    
    def is_alive(self):
        """Check if the viewer is still alive and responsive"""
        if self.is_closed:
            return False
        try:
            # MuJoCo viewer doesn't have is_running() method in all versions
            # So we check if the viewer object exists and hasn't been closed
            return hasattr(self, 'viewer') and self.viewer is not None and not self.is_closed
        except:
            return False
    
    def close(self):
        """Safely close the viewer and clean up resources"""
        if self.is_closed:
            return
            
        with self._lock:
            try:
                print("Closing viewer...")
                self.is_closed = True
                
                if hasattr(self, 'viewer') and self.viewer is not None:
                    self.viewer.close()
                    
                if self.record_video and hasattr(self, 'mp4_writer'):
                    self.mp4_writer.close()
                    print(f"Video saved to {self.video_path}")
                
                # Small delay to ensure cleanup
                time.sleep(0.5)
                print("Viewer closed successfully")
                
            except Exception as e:
                print(f"Error during viewer cleanup: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        if not self.is_closed:
            self.close()