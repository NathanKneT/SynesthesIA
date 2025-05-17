import numpy as np
import mediapipe as mp
from enum import Enum, auto
from typing import Dict, Any, Tuple, List, Optional

mp_pose = mp.solutions.pose

class GestureType(Enum):
    """Types of gestures recognized by the system"""
    NEUTRAL = auto()       # Neutral position
    POWER_UP = auto()      # Arms raised
    CROSSED_ARMS = auto()  # Arms crossed
    T_POSE = auto()        # T position
    TAP_LEFT = auto()      # Tap left arm
    TAP_RIGHT = auto()     # Tap right arm
    POINT_UP = auto()      # Point upwards (for instrument selection)
    POINT_DOWN = auto()    # Point downwards (for instrument selection)
    POINT_LEFT = auto()    # Point left
    POINT_RIGHT = auto()   # Point right


class GestureRecognizer:
    def __init__(self):
        # Threshold values for gesture detection
        self.thresholds = {
            # Thresholds for existing gestures
            "WRIST_SHOULDER_RATIO": 0.2,        # Threshold for raised arms
            "CROSS_DISTANCE_THRESHOLD": 0.1,    # Threshold for crossed arms
            "T_POSE_HORIZONTAL_THRESHOLD": 0.15, # Threshold for arm horizontality in T-pose
            "T_POSE_EXTENSION_RATIO": 0.3,      # Minimum arm spread relative to body size for T-pose
            
            # New thresholds for additional gestures
            "TAP_DISTANCE_THRESHOLD": 0.1,      # Minimum distance to detect a "tap"
            "POINT_ANGLE_THRESHOLD": 0.3,       # Angle to detect pointing
            "WRIST_VELOCITY_THRESHOLD": 0.05    # Velocity threshold to detect fast movement
        }
        
        # Position history for velocity calculation
        self.previous_landmarks = None
        self.landmark_velocities = {}
        
        # For debugging
        self.debug_values = {}
        
        # Mapping of gestures to their readable description
        self.gesture_names = {
            GestureType.NEUTRAL: "neutral",
            GestureType.POWER_UP: "power_up",
            GestureType.CROSSED_ARMS: "crossed_arms",
            GestureType.T_POSE: "t_pose",
            GestureType.TAP_LEFT: "tap_left",
            GestureType.TAP_RIGHT: "tap_right",
            GestureType.POINT_UP: "point_up",
            GestureType.POINT_DOWN: "point_down",
            GestureType.POINT_LEFT: "point_left",
            GestureType.POINT_RIGHT: "point_right"
        }
        
        # Parameters to associate gestures with sound/visual actions
        self.gesture_parameters = {
            GestureType.POWER_UP: {"frequency": "high", "visual": "vortex_up"},
            GestureType.CROSSED_ARMS: {"frequency": "low", "visual": "vortex_down"},
            GestureType.T_POSE: {"frequency": "mid", "visual": "expand"},
            GestureType.TAP_LEFT: {"instrument": "snare", "visual": "impact_left"},
            GestureType.TAP_RIGHT: {"instrument": "kick", "visual": "impact_right"},
            GestureType.POINT_UP: {"instrument_select": "high", "visual": "select_up"},
            GestureType.POINT_DOWN: {"instrument_select": "low", "visual": "select_down"},
            GestureType.POINT_LEFT: {"instrument_select": "left", "visual": "select_left"},
            GestureType.POINT_RIGHT: {"instrument_select": "right", "visual": "select_right"}
        }
        
    def _calculate_velocities(self, current_landmarks):
        """Calculates keypoint velocities between two frames"""
        if self.previous_landmarks is None:
            self.previous_landmarks = current_landmarks
            return {}
        
        velocities = {}
        for landmark in mp_pose.PoseLandmark:
            name = landmark.name
            curr_pos = np.array([
                current_landmarks.landmark[landmark].x,
                current_landmarks.landmark[landmark].y,
                current_landmarks.landmark[landmark].z
            ])
            prev_pos = np.array([
                self.previous_landmarks.landmark[landmark].x,
                self.previous_landmarks.landmark[landmark].y,
                self.previous_landmarks.landmark[landmark].z
            ])
            velocities[name] = np.linalg.norm(curr_pos - prev_pos)
        
        # Update previous landmarks
        self.previous_landmarks = current_landmarks
        self.landmark_velocities = velocities
        return velocities
    
    def _extract_landmarks(self, pose_landmarks) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """Extracts keypoint coordinates and visibility"""
        landmarks = {}
        visibility = {}
        
        for landmark in mp_pose.PoseLandmark:
            landmarks[landmark.name] = np.array([
                pose_landmarks.landmark[landmark].x,
                pose_landmarks.landmark[landmark].y,
                pose_landmarks.landmark[landmark].z
            ])
            visibility[landmark.name] = pose_landmarks.landmark[landmark].visibility
            
        return landmarks, visibility
    
    def calibrate(self, calibration_frames):
        """
        Calibrates detection thresholds based on user data.
        
        Args:
            calibration_frames: List of MediaPipe pose frames for calibration
            
        Returns:
            bool: True if calibration succeeded, False otherwise
        """
        if not calibration_frames or len(calibration_frames) < 10:
            return False
            
        try:
            # Extract statistics from calibration frames
            body_heights = []
            shoulder_widths = []
            
            for frame in calibration_frames:
                # Extract landmarks for analysis
                landmarks, visibility = self._extract_landmarks(frame)
                
                # Calculate basic metrics
                body_height = np.linalg.norm(landmarks['NOSE'] - landmarks['LEFT_HIP'])
                shoulder_width = np.linalg.norm(landmarks['LEFT_SHOULDER'] - landmarks['RIGHT_SHOULDER'])
                
                body_heights.append(body_height)
                shoulder_widths.append(shoulder_width)
            
            # Calculate averages
            avg_body_height = np.mean(body_heights)
            avg_shoulder_width = np.mean(shoulder_widths)
            
            # Calculate average user proportions
            arm_proportion = avg_shoulder_width / avg_body_height
            
            # Adjust thresholds based on proportions
            self.thresholds["WRIST_SHOULDER_RATIO"] = max(0.2, min(0.4, 0.3 * arm_proportion))
            self.thresholds["CROSS_DISTANCE_THRESHOLD"] = max(0.08, min(0.2, 0.15 * arm_proportion))
            self.thresholds["T_POSE_HORIZONTAL_THRESHOLD"] = max(0.1, min(0.3, 0.2 * arm_proportion))
            self.thresholds["T_POSE_EXTENSION_RATIO"] = max(0.2, min(0.4, 0.25 * arm_proportion))
            self.thresholds["TAP_DISTANCE_THRESHOLD"] = max(0.1, min(0.25, 0.15 * arm_proportion))
            self.thresholds["POINT_ANGLE_THRESHOLD"] = max(0.2, min(0.35, 0.25 * arm_proportion))
            
            # Store these values for debugging
            self.debug_values["calibrated_body_height"] = float(avg_body_height)
            self.debug_values["calibrated_shoulder_width"] = float(avg_shoulder_width)
            self.debug_values["calibrated_arm_proportion"] = float(arm_proportion)
            
            return True
            
        except Exception as e:
            print(f"Calibration error: {str(e)}")
            return False
        
    def _detect_power_up(self, landmarks, visibility) -> bool:
        """Detects 'arms raised' gesture"""
        left_wrist_above = landmarks['LEFT_WRIST'][1] < landmarks['LEFT_SHOULDER'][1] - self.thresholds["WRIST_SHOULDER_RATIO"]
        right_wrist_above = landmarks['RIGHT_WRIST'][1] < landmarks['RIGHT_SHOULDER'][1] - self.thresholds["WRIST_SHOULDER_RATIO"]
        
        # Both arms must be raised and wrists must be visible
        left_wrist_visible = visibility['LEFT_WRIST'] > 0.5
        right_wrist_visible = visibility['RIGHT_WRIST'] > 0.5
        
        # To avoid false positives, verify wrists are in reasonable position
        left_wrist_reasonable_x = abs(landmarks['LEFT_WRIST'][0] - landmarks['LEFT_SHOULDER'][0]) < 0.3
        right_wrist_reasonable_x = abs(landmarks['RIGHT_WRIST'][0] - landmarks['RIGHT_SHOULDER'][0]) < 0.3
        
        arms_raised = (left_wrist_above and right_wrist_above and 
                      left_wrist_visible and right_wrist_visible and
                      left_wrist_reasonable_x and right_wrist_reasonable_x)
        
        # Store values for debugging
        self.debug_values['left_wrist_above'] = left_wrist_above
        self.debug_values['right_wrist_above'] = right_wrist_above
        
        return arms_raised
    
    def _detect_crossed_arms(self, landmarks, visibility) -> bool:
        """Detects 'arms crossed' gesture"""
        wrists_close = np.linalg.norm(landmarks['LEFT_WRIST'] - landmarks['RIGHT_WRIST']) < self.thresholds["CROSS_DISTANCE_THRESHOLD"]
        left_wrist_right = landmarks['LEFT_WRIST'][0] > landmarks['NOSE'][0]
        right_wrist_left = landmarks['RIGHT_WRIST'][0] < landmarks['NOSE'][0]
        
        # Wrists must be visible and at similar height (not one up and one down)
        wrists_similar_height = abs(landmarks['LEFT_WRIST'][1] - landmarks['RIGHT_WRIST'][1]) < 0.15
        
        # Elbows must be bent
        elbows_bent = (landmarks['LEFT_ELBOW'][0] < landmarks['LEFT_WRIST'][0] and 
                      landmarks['RIGHT_ELBOW'][0] > landmarks['RIGHT_WRIST'][0])
        
        arms_crossed = (wrists_close and 
                       visibility['LEFT_WRIST'] > 0.5 and 
                       visibility['RIGHT_WRIST'] > 0.5 and 
                       wrists_similar_height and 
                       ((left_wrist_right and right_wrist_left) or elbows_bent))
        
        self.debug_values['wrists_close'] = wrists_close
        self.debug_values['left_wrist_right'] = left_wrist_right
        self.debug_values['right_wrist_left'] = right_wrist_left
        
        return arms_crossed
    
    def _detect_t_pose(self, landmarks, visibility) -> bool:
        """Detects 'T-pose' gesture"""
        # Calculate some basic metrics
        body_height = np.linalg.norm(landmarks['NOSE'] - landmarks['LEFT_HIP'])
        
        # 1. Verify vertical body alignment
        upper_body_aligned = abs(landmarks['NOSE'][0] - landmarks['LEFT_HIP'][0]) < 0.1
        
        # 2. Verify arms are horizontal
        left_arm_horizontal = abs(landmarks['LEFT_WRIST'][1] - landmarks['LEFT_SHOULDER'][1]) < self.thresholds["T_POSE_HORIZONTAL_THRESHOLD"]
        right_arm_horizontal = abs(landmarks['RIGHT_WRIST'][1] - landmarks['RIGHT_SHOULDER'][1]) < self.thresholds["T_POSE_HORIZONTAL_THRESHOLD"]
        
        # 3. Measure arm spread relative to body size
        arms_span = np.linalg.norm(landmarks['LEFT_WRIST'] - landmarks['RIGHT_WRIST'])
        arms_extension_ratio = arms_span / body_height
        arms_sufficiently_extended = arms_extension_ratio > self.thresholds["T_POSE_EXTENSION_RATIO"]
        
        # 4. Verify arms are extended to each side
        left_arm_left = landmarks['LEFT_WRIST'][0] < landmarks['LEFT_SHOULDER'][0] - 0.1
        right_arm_right = landmarks['RIGHT_WRIST'][0] > landmarks['RIGHT_SHOULDER'][0] + 0.1
        
        # 5. Ensure keypoints are visible
        key_points_visible = (
            visibility['LEFT_WRIST'] > 0.5 and 
            visibility['RIGHT_WRIST'] > 0.5 and
            visibility['LEFT_SHOULDER'] > 0.5 and 
            visibility['RIGHT_SHOULDER'] > 0.5
        )
        
        t_pose = (
            upper_body_aligned and 
            left_arm_horizontal and 
            right_arm_horizontal and
            arms_sufficiently_extended and
            left_arm_left and 
            right_arm_right and
            key_points_visible
        )
        
        # Debug values for T-pose
        self.debug_values['upper_body_aligned'] = upper_body_aligned
        self.debug_values['left_arm_horizontal'] = left_arm_horizontal
        self.debug_values['right_arm_horizontal'] = right_arm_horizontal
        self.debug_values['arms_extension_ratio'] = float(arms_extension_ratio)
        
        return t_pose
    
    def _detect_tap(self, landmarks, visibility) -> Optional[GestureType]:
        """Detects if one arm is tapping the other"""
        # Check distance between right wrist and left elbow (for tapping left arm)
        right_to_left_dist = np.linalg.norm(landmarks['RIGHT_WRIST'] - landmarks['LEFT_ELBOW'])
        tap_left = (right_to_left_dist < self.thresholds["TAP_DISTANCE_THRESHOLD"] and 
                   visibility['RIGHT_WRIST'] > 0.5 and 
                   visibility['LEFT_ELBOW'] > 0.5 and
                   self.landmark_velocities.get('RIGHT_WRIST', 0) > self.thresholds["WRIST_VELOCITY_THRESHOLD"])
        
        # Check distance between left wrist and right elbow (for tapping right arm)
        left_to_right_dist = np.linalg.norm(landmarks['LEFT_WRIST'] - landmarks['RIGHT_ELBOW'])
        tap_right = (left_to_right_dist < self.thresholds["TAP_DISTANCE_THRESHOLD"] and 
                    visibility['LEFT_WRIST'] > 0.5 and 
                    visibility['RIGHT_ELBOW'] > 0.5 and
                    self.landmark_velocities.get('LEFT_WRIST', 0) > self.thresholds["WRIST_VELOCITY_THRESHOLD"])
        
        self.debug_values['right_to_left_dist'] = float(right_to_left_dist)
        self.debug_values['left_to_right_dist'] = float(left_to_right_dist)
        self.debug_values['right_wrist_velocity'] = float(self.landmark_velocities.get('RIGHT_WRIST', 0))
        self.debug_values['left_wrist_velocity'] = float(self.landmark_velocities.get('LEFT_WRIST', 0))
        
        if tap_left:
            return GestureType.TAP_LEFT
        elif tap_right:
            return GestureType.TAP_RIGHT
        return None
    
    def _detect_pointing(self, landmarks, visibility) -> Optional[GestureType]:
        """Detects pointing gestures for instrument selection"""
        # Check if right index finger is extended (pointing position)
        if visibility['RIGHT_INDEX'] < 0.5 or visibility['RIGHT_WRIST'] < 0.5:
            return None
            
        # Pointing direction (vector from wrist to index finger)
        pointing_vector = landmarks['RIGHT_INDEX'] - landmarks['RIGHT_WRIST']
        pointing_vector = pointing_vector / np.linalg.norm(pointing_vector)
        
        # Check pointing directions
        pointing_up = pointing_vector[1] < -self.thresholds["POINT_ANGLE_THRESHOLD"]
        pointing_down = pointing_vector[1] > self.thresholds["POINT_ANGLE_THRESHOLD"]
        pointing_left = pointing_vector[0] < -self.thresholds["POINT_ANGLE_THRESHOLD"]
        pointing_right = pointing_vector[0] > self.thresholds["POINT_ANGLE_THRESHOLD"]
        
        self.debug_values['pointing_vector_x'] = float(pointing_vector[0])
        self.debug_values['pointing_vector_y'] = float(pointing_vector[1])
        
        if pointing_up:
            return GestureType.POINT_UP
        elif pointing_down:
            return GestureType.POINT_DOWN
        elif pointing_left:
            return GestureType.POINT_LEFT
        elif pointing_right:
            return GestureType.POINT_RIGHT
        
        return None
        
    def detect_gestures(self, pose_landmarks) -> Dict[str, Any]:
        """
        Detects body gestures and returns the main gesture with its parameters
        Returns a dictionary with:
        - 'type': Gesture type (GestureType Enum)
        - 'name': Readable gesture name
        - 'parameters': Parameters associated with the gesture (for sound/visual generation)
        """
        # Extract landmarks and calculate velocities
        landmarks, visibility = self._extract_landmarks(pose_landmarks)
        velocities = self._calculate_velocities(pose_landmarks)
        
        # Reset debug values
        self.debug_values = {}
        
        # Calculate some basic metrics for debugging
        body_height = np.linalg.norm(landmarks['NOSE'] - landmarks['LEFT_HIP'])
        shoulder_width = np.linalg.norm(landmarks['LEFT_SHOULDER'] - landmarks['RIGHT_SHOULDER'])
        self.debug_values['body_height'] = float(body_height)
        self.debug_values['shoulder_width'] = float(shoulder_width)
        
        # Detect gestures in order of priority (from most specific to most general)
        # 1. Check "tap" gestures (newly added)
        tap_gesture = self._detect_tap(landmarks, visibility)
        if tap_gesture:
            return {
                'type': tap_gesture, 
                'name': self.gesture_names[tap_gesture],
                'parameters': self.gesture_parameters[tap_gesture]
            }
            
        # 2. Check pointing gestures (for instrument selection)
        pointing_gesture = self._detect_pointing(landmarks, visibility)
        if pointing_gesture:
            return {
                'type': pointing_gesture, 
                'name': self.gesture_names[pointing_gesture],
                'parameters': self.gesture_parameters[pointing_gesture]
            }
        
        # 3. Check T-pose
        if self._detect_t_pose(landmarks, visibility):
            return {
                'type': GestureType.T_POSE, 
                'name': self.gesture_names[GestureType.T_POSE],
                'parameters': self.gesture_parameters[GestureType.T_POSE]
            }
            
        # 4. Check "arms raised" gesture
        if self._detect_power_up(landmarks, visibility):
            return {
                'type': GestureType.POWER_UP, 
                'name': self.gesture_names[GestureType.POWER_UP],
                'parameters': self.gesture_parameters[GestureType.POWER_UP]
            }
            
        # 5. Check "arms crossed" gesture
        if self._detect_crossed_arms(landmarks, visibility):
            return {
                'type': GestureType.CROSSED_ARMS, 
                'name': self.gesture_names[GestureType.CROSSED_ARMS],
                'parameters': self.gesture_parameters[GestureType.CROSSED_ARMS]
            }
            
        # No specific gesture detected
        return {
            'type': GestureType.NEUTRAL, 
            'name': self.gesture_names[GestureType.NEUTRAL],
            'parameters': {}
        }
    
    def get_debug_info(self):
        """Returns debug information for visualization"""
        return self.debug_values