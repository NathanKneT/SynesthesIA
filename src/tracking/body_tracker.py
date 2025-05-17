"""
Body movement tracking module for Body-Sound Vision.
Improved version with more reliable gesture detection and direct audio integration.
"""
import numpy as np
import cv2
import mediapipe as mp
import time
from typing import Dict, Any, List, Optional, Union

# Import custom modules
from tracking.gesture_recognition import GestureRecognizer, GestureType
from communication.osc_manager import OscManager
from audio.audio_manager import AudioManager  # Verify this line is present

class BodyTracker:
    """
    Main class for body movement tracking and gesture detection.
    Handles video capture, gesture recognition, display, OSC communication
    and direct audio generation.
    """
    def __init__(self, camera_index=0):
        """
        Initialize the body tracker with the specified camera.
        
        Args:
            camera_index: Index of the camera to use (0 by default for webcam)
        """
        # Initialize MediaPipe Pose
        print("Initializing MediaPipe components...")
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize camera
        print(f"Opening camera {camera_index}...")
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(camera_index)
            # Check if camera is open
        if not self.cap.isOpened():
            print(f"WARNING: Failed to open camera {camera_index}")
            print("Trying alternative camera index...")
            
            # Try another camera index
            for alt_index in [0, 1, 2]:
                if alt_index != camera_index:
                    print(f"Trying camera index {alt_index}...")
                    self.cap = cv2.VideoCapture(alt_index)
                    if self.cap.isOpened():
                        print(f"Successfully opened camera {alt_index}")
                        self.camera_index = alt_index
                        break
            
            # If still no camera, warn
            if not self.cap.isOpened():
                print("ERROR: No camera available. The application may not work properly.")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Camera configuration for better performance
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # Target FPS
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Compression format

        # Set window size
        self.display_width = 1280 
        self.display_height = 720  
        
        # Initialize gesture manager
        self.gesture_recognizer = GestureRecognizer()
        
        # Initialize OSC manager for communication
        self.osc_manager = OscManager()
        
        # Initialize audio manager (NEW)
        self.audio_manager = AudioManager()
        self.audio_enabled = False
        
        # Event log
        self.event_log = []
        self.MAX_LOG_ENTRIES = 10
        
        # Interface state variables
        self.show_logs = False
        self.show_help = False
        self.show_debug = False
        self.show_hud = True
        self.show_osc_history = False
        self.recording = False
        self.recording_frames = []
        self.recording_start_time = None
        self.MAX_RECORDING_FRAMES = 300  # 10 seconds at 30 FPS
        self.hud_x_offset = 80  # HUD offset
        
        # Gesture state
        self.last_gesture = None
        self.gesture_cooldown = 0
        self.selected_instrument = "snare"  # Default instrument
        
        # Variables for FPS calculation
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.fps = 0
        self.update_fps_interval = 10  # Update FPS every 10 frames
        
        # Instrument zones for HUD (for selection)
        self.instrument_zones = {
            "snare": {"x": 50, "y": 50, "width": 100, "height": 50, "color": (0, 255, 0)},
            "kick": {"x": 50, "y": 120, "width": 100, "height": 50, "color": (0, 0, 255)},
            "hihat": {"x": 50, "y": 190, "width": 100, "height": 50, "color": (255, 0, 0)},
            "bass": {"x": 50, "y": 260, "width": 100, "height": 50, "color": (0, 255, 255)}
        }
        
        # Zone for visualizing sound intensity
        self.audio_level = 0.0
        self.audio_level_decay = 0.1
        
        # Display parameters
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.line_thickness = 1
        self.colors = {
            "text": (255, 255, 255),
            "highlight": (0, 255, 255),
            "success": (0, 255, 0),
            "warning": (0, 165, 255),  # Orange
            "error": (0, 0, 255),      # Red
            "bg_dark": (50, 50, 50),
            "bg_light": (80, 80, 80),
            "border": (200, 200, 200)
        }
        
        # For calibration (NEW)
        self.calibration_mode = False
        self.calibration_start_time = None
        self.calibration_frames = []
        self.CALIBRATION_DURATION = 5  # 5 seconds
    
    def add_to_log(self, message: str, message_type: str = "info"):
        """
        Add a message to the event log.
        
        Args:
            message: Message to add
            message_type: Message type ('info', 'warning', 'error', 'success')
        """
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        entry = {
            "timestamp": timestamp,
            "message": message,
            "type": message_type
        }
        self.event_log.append(entry)
        
        # Keep only the most recent entries
        if len(self.event_log) > self.MAX_LOG_ENTRIES:
            self.event_log.pop(0)
    
    def update_fps(self):
        """Update FPS calculation"""
        self.fps_frame_count += 1
        
        # Update FPS every N frames
        if self.fps_frame_count >= self.update_fps_interval:
            current_time = time.time()
            elapsed = current_time - self.fps_start_time
            self.fps = self.fps_frame_count / elapsed
            
            # Reset counter
            self.fps_start_time = current_time
            self.fps_frame_count = 0
            
            # Log FPS every ~5 seconds (150 frames at 30fps)
            if self.fps_frame_count % 150 == 0:
                self.add_to_log(f"Performance: {self.fps:.1f} FPS")
    
    def draw_hud(self, image):
        """
        Draw the user interface (HUD) with instrument selection.
        
        Args:
            image: OpenCV image to draw on
            
        Returns:
            OpenCV image with HUD
        """
        if not self.show_hud:
            return image
        
        # Draw main HUD area (shifted to the right)
        cv2.rectangle(image, (self.hud_x_offset + 30, 30), 
                    (self.hud_x_offset + 180, 330), self.colors["bg_dark"], -1)
        cv2.rectangle(image, (self.hud_x_offset + 30, 30), 
                    (self.hud_x_offset + 180, 330), self.colors["border"], 2)
        cv2.putText(image, "INSTRUMENTS", (self.hud_x_offset + 45, 25), 
                self.font, self.font_scale, self.colors["text"], self.line_thickness)
        
        # Draw instrument zones
        for name, zone in self.instrument_zones.items():
            # Background color (brighter if selected)
            adj_x = zone["x"] + self.hud_x_offset
            if name == self.selected_instrument:
                bg_color = tuple([min(255, int(c * 1.5)) for c in zone["color"]])
                border_thickness = 2
            else:
                bg_color = (30, 30, 30)
                border_thickness = 1
            
            # Draw instrument rectangle
            cv2.rectangle(image, 
                         (adj_x, zone["y"]), 
                         (adj_x + zone["width"], zone["y"] + zone["height"]), 
                         bg_color, -1)
            
            # Border
            cv2.rectangle(image, 
                         (adj_x, zone["y"]), 
                         (adj_x + zone["width"], zone["y"] + zone["height"]), 
                         zone["color"], border_thickness)
            
            # Instrument name
            cv2.putText(image, name.upper(), 
                       (adj_x + 10, zone["y"] + 30), 
                       self.font, self.font_scale, self.colors["text"], self.line_thickness)
        
        # MODIFIED: Audio level indicator position (lower and more to the right)
        audio_x = self.hud_x_offset + 140  # Moved more to the right
        audio_y = 450  # Moved lower
        
        # Display audio level indicator
        level_width = int(100 * self.audio_level)
        cv2.rectangle(image, (audio_x, audio_y), 
                    (audio_x + 130, audio_y + 20), self.colors["bg_light"], -1)
        cv2.rectangle(image, (audio_x, audio_y), 
                    (audio_x + level_width, audio_y + 20), 
                    (0, int(255 * self.audio_level), int(255 * (1.0 - self.audio_level))), -1)
        cv2.rectangle(image, (audio_x, audio_y), 
                    (audio_x + 130, audio_y + 20), self.colors["border"], 1)
        cv2.putText(image, "AUDIO LEVEL", (audio_x + 5, audio_y - 5), 
                self.font, self.font_scale, self.colors["text"], self.line_thickness)
        
        # Gradually decrease audio level (for animation)
        self.audio_level = max(0, self.audio_level - self.audio_level_decay)
        
        # MODIFIED: OSC and audio status positions (lower)
        status_x = self.hud_x_offset + 45
        status_y_osc = 400
        status_y_audio = 430
        
        # Display OSC status
        osc_status = "OSC: " + ("ON" if self.osc_manager.osc_enabled else "OFF")
        cv2.putText(image, osc_status, (status_x, status_y_osc), self.font, self.font_scale, 
                self.colors["success"] if self.osc_manager.osc_enabled else self.colors["warning"], 
                self.line_thickness)
        
        # Display Audio status
        audio_status = "AUDIO: " + ("ON" if self.audio_enabled else "OFF")
        cv2.putText(image, audio_status, (status_x, status_y_audio), self.font, self.font_scale, 
                self.colors["success"] if self.audio_enabled else self.colors["warning"], 
                self.line_thickness)
                   
        # Display recording status
        if self.recording:
            # Blink recording text
            blink = int(time.time() * 2) % 2
            if blink:
                cv2.putText(image, "RECORDING", (45, 430), self.font, self.font_scale, 
                           self.colors["error"], self.line_thickness)
                           
            # Display recording time
            elapsed = time.time() - self.recording_start_time
            cv2.putText(image, f"{elapsed:.1f}s", (150, 430), self.font, self.font_scale, 
                       self.colors["text"], self.line_thickness)
        
        # Display FPS in bottom right
        cv2.putText(image, f"{self.fps:.1f} FPS", (self.width - 100, self.height - 20), 
                   self.font, self.font_scale, self.colors["text"], self.line_thickness)
                   
        return image
    
    def draw_help(self, image):
        """
        Draw the help overlay on the image.
        
        Args:
            image: OpenCV image to draw on
            
        Returns:
            OpenCV image with help
        """
        if not self.show_help:
            return image
            
        # MODIFIED: Position more to the right to avoid overlap
        help_x_offset = self.width - 550  # Moved completely to the right
        
        # Create a semi-transparent overlay in the right part
        help_overlay = image.copy()
        cv2.rectangle(help_overlay, (help_x_offset, 0), (self.width, 280), (0, 0, 0), -1)
        image = cv2.addWeighted(help_overlay, 0.7, image, 0.3, 0)
        
        # Title (moved to the right)
        cv2.putText(image, "BODY GESTURE GUIDE:", (help_x_offset + 10, 30), self.font, 0.8, 
                self.colors["highlight"], 2)
        
        # Available gestures (moved to the right)
        gestures = [
            "Arms Raised: power_up gesture (high sound)",
            "Arms Crossed: crossed_arms gesture (low sound)",
            "T-Pose: t_pose gesture (expanding visual)",
            "Tap Left Arm: tap_left gesture (snare sound)",
            "Tap Right Arm: tap_right gesture (kick sound)",
            "Point Up/Down/Left/Right: Select instrument"
        ]
        
        for i, gesture in enumerate(gestures):
            cv2.putText(image, gesture, (help_x_offset + 20, 70 + i * 30), self.font, self.font_scale,
                    self.colors["highlight"], self.line_thickness)
        
        # Keyboard controls (moved to the right but lower)
        controls = [
            "H: Toggle help | L: Toggle logs | D: Toggle debug",
            "O: Toggle OSC | A: Toggle Audio | V: Toggle OSC history",
            "C: Calibrate tracking | R: Start/Stop recording | S: Save recording",
            "ESC: Exit"
        ]
        
        # MODIFIED: Lower position for controls (bottom right)
        controls_y_start = self.height - 140  # Moved to the bottom
        
        for i, control in enumerate(controls):
            cv2.putText(image, control, (help_x_offset + 20, controls_y_start + i * 25), 
                    self.font, self.font_scale, self.colors["text"], self.line_thickness)
        
        return image
    
    def draw_logs(self, image):
        """
        Draw the event log on the image.
        
        Args:
            image: OpenCV image to draw on
            
        Returns:
            OpenCV image with logs
        """
        if not self.show_logs:
            return image
            
        # Create log area
        log_y_start = self.height - 220
        cv2.rectangle(image, (0, log_y_start), (self.width, self.height), (0, 0, 0), -1)
        cv2.putText(image, "EVENT LOG:", (10, log_y_start + 25), self.font, 0.7,
                   self.colors["text"], 2)
        
        # Display log entries
        for i, entry in enumerate(reversed(self.event_log)):
            # Determine color based on message type
            if entry["type"] == "error":
                color = self.colors["error"]
            elif entry["type"] == "warning":
                color = self.colors["warning"]
            elif entry["type"] == "success":
                color = self.colors["success"]
            else:  # info
                color = self.colors["text"]
                
            message = f"[{entry['timestamp']}] {entry['message']}"
            cv2.putText(image, message, (20, log_y_start + 60 + (i * 30)),
                       self.font, self.font_scale, color, self.line_thickness)
        
        return image
    
    def draw_osc_history(self, image):
        """
        Draw the OSC message history on the image.
        
        Args:
            image: OpenCV image to draw on
            
        Returns:
            OpenCV image with OSC history
        """
        if not self.show_osc_history or not self.osc_manager.osc_enabled:
            return image
            
        # Get OSC message history
        osc_history = self.osc_manager.get_message_history()
        if not osc_history:
            return image
            
        # Create OSC history area
        history_width = 400
        history_height = min(30 * len(osc_history) + 40, 300)
        start_x = self.width - history_width - 10
        start_y = 10
        
        cv2.rectangle(image, (start_x, start_y), 
                     (start_x + history_width, start_y + history_height), 
                     (0, 0, 0), -1)
        cv2.rectangle(image, (start_x, start_y), 
                     (start_x + history_width, start_y + history_height), 
                     self.colors["border"], 1)
        
        # Title
        cv2.putText(image, "OSC MESSAGE HISTORY:", (start_x + 10, start_y + 25), 
                   self.font, 0.7, self.colors["highlight"], 2)
        
        # Display messages
        for i, message in enumerate(reversed(osc_history)):
            # Truncate messages that are too long
            if len(message) > 50:
                message = message[:47] + "..."
                
            cv2.putText(image, message, (start_x + 10, start_y + 55 + (i * 25)),
                       self.font, 0.5, self.colors["text"], 1)
            
            # Limit number of displayed messages
            if i >= 9:  # 10 messages maximum
                break
        
        return image
    
    def draw_debug_values(self, image, debug_values):
        """
        Draw debug values on the image.
        
        Args:
            image: OpenCV image to draw on
            debug_values: Dictionary of debug values
            
        Returns:
            OpenCV image with debug values
        """
        if not self.show_debug or not debug_values:
            return image
            
        debug_y_start = 250
        cv2.putText(image, "DEBUG VALUES:", (250, debug_y_start), 
                   self.font, 0.6, self.colors["highlight"], 1)
        
        # Display up to 10 debug values
        top_debug_values = list(debug_values.items())[:10]
        
        for i, (key, value) in enumerate(top_debug_values):
            # Color based on value for booleans
            if isinstance(value, bool):
                color = self.colors["success"] if value else self.colors["error"]
            else:
                color = self.colors["text"]
                
            text = f"{key}: {value}"
            if isinstance(value, float):
                text = f"{key}: {value:.3f}"
                
            cv2.putText(image, text, (250, debug_y_start + 30 + (i * 25)),
                       self.font, 0.5, color, 1)
        
        return image
    
    def start_calibration(self):
        """Start the calibration process"""
        if self.calibration_mode:
            return
            
        self.calibration_mode = True
        self.calibration_start_time = time.time()
        self.calibration_frames = []
        self.add_to_log("Calibration started, please stand in a neutral position", "info")

    def process_calibration(self, pose_landmarks):
        """Process calibration frames"""
        # Add landmarks to calibration collection
        self.calibration_frames.append(pose_landmarks)
        
        # Check if calibration time has elapsed
        elapsed = time.time() - self.calibration_start_time
        if elapsed >= self.CALIBRATION_DURATION:
            # Finish calibration
            success = self.gesture_recognizer.calibrate(self.calibration_frames)
            if success:
                self.add_to_log("Calibration successful", "success")
            else:
                self.add_to_log("Calibration failed", "error")
            
            # Reset calibration variables
            self.calibration_mode = False
            self.calibration_frames = []
            self.calibration_start_time = None
    
    def process_gesture(self, gesture_info):
        """
        Process a detected gesture and handle appropriate actions.
        
        Args:
            gesture_info: Dictionary containing gesture information
        """
        gesture_type = gesture_info['type']
        gesture_name = gesture_info['name']
        
        # Ignore if it's the same gesture as before and cooldown is active
        if (self.last_gesture and self.last_gesture['name'] == gesture_name 
                and self.gesture_cooldown > 0):
            self.gesture_cooldown -= 1
            return
            
        # Ignore neutral gesture to avoid spamming the log
        if gesture_type == GestureType.NEUTRAL:
            return
            
        # Reset cooldown
        self.gesture_cooldown = 15
        
        # Record new gesture
        self.last_gesture = gesture_info
        
        # Add gesture to log
        self.add_to_log(f"Detected: {gesture_name}", "success")
        
        # Process instrument selection gestures
        parameters = gesture_info.get('parameters', {})
        
        # Handle instrument selection via pointing
        if gesture_type in [GestureType.POINT_UP, GestureType.POINT_DOWN, 
                           GestureType.POINT_LEFT, GestureType.POINT_RIGHT]:
            direction = parameters.get('instrument_select')
            if direction == 'up':
                self.selected_instrument = "snare"
                self.add_to_log(f"Selected: {self.selected_instrument}")
            elif direction == 'down':
                self.selected_instrument = "kick"
                self.add_to_log(f"Selected: {self.selected_instrument}")
            elif direction == 'left':
                self.selected_instrument = "hihat"
                self.add_to_log(f"Selected: {self.selected_instrument}")
            elif direction == 'right':
                self.selected_instrument = "bass"
                self.add_to_log(f"Selected: {self.selected_instrument}")
        
        # Simulate audio level based on gesture
        if gesture_type in [GestureType.TAP_LEFT, GestureType.TAP_RIGHT]:
            self.audio_level = 1.0  # Strong percussion sound
        elif gesture_type == GestureType.POWER_UP:
            self.audio_level = 0.8  # Medium-strong sound
        elif gesture_type == GestureType.CROSSED_ARMS:
            self.audio_level = 0.6  # Medium sound
        elif gesture_type == GestureType.T_POSE:
            self.audio_level = 0.9  # Strong sound
        
        # Send gesture information via OSC
        if self.osc_manager.osc_enabled:
            # Add currently selected instrument to parameters
            if 'parameters' in gesture_info and gesture_info['parameters']:
                gesture_info['parameters']['active_instrument'] = self.selected_instrument
                
            # For tap gestures, add selected instrument
            if gesture_type in [GestureType.TAP_LEFT, GestureType.TAP_RIGHT]:
                gesture_info['parameters']['instrument'] = self.selected_instrument
                
            success = self.osc_manager.send_gesture(gesture_info)
            if not success:
                self.add_to_log("OSC send failed", "error")

    def toggle_audio(self):
        """Enable or disable direct audio output"""
        try:
            if self.audio_enabled:
                # Disable audio
                self.audio_manager.stop()
                self.audio_enabled = False
                self.add_to_log("Audio output disabled", "info")
            else:
                # Verify audio_manager is properly initialized
                if not hasattr(self, 'audio_manager') or self.audio_manager is None:
                    from audio.audio_manager import AudioManager
                    self.audio_manager = AudioManager()
                    
                # Enable audio
                success = self.audio_manager.start()
                self.audio_enabled = success
                if success:
                    self.add_to_log("Audio output enabled", "success")
                else:
                    self.add_to_log("Failed to start audio output", "error")
        except Exception as e:
            # Add try-except block to capture errors
            self.add_to_log(f"Audio error: {str(e)}", "error")
            import traceback
            traceback.print_exc()
            self.audio_enabled = False
    
    def start_recording(self):
        """Start session recording"""
        if self.recording:
            return
            
        self.recording = True
        self.recording_frames = []
        self.recording_start_time = time.time()
        self.add_to_log("Recording started", "success")
    
    def stop_recording(self):
        """Stop session recording"""
        if not self.recording:
            return
            
        self.recording = False
        elapsed = time.time() - self.recording_start_time
        self.add_to_log(f"Recording stopped ({len(self.recording_frames)} frames, {elapsed:.1f}s)", "info")
    
    def save_recording(self):
        """Save recording"""
        if not self.recording_frames:
            self.add_to_log("No recording to save", "warning")
            return
            
        # Generate filename with date and time
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        filename = f"body_sound_vision_{timestamp}.mp4"
        
        # Video writer configuration
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30.0
        first_frame = self.recording_frames[0]
        height, width = first_frame.shape[:2]
        
        # Create writer
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        
        # Write frames
        for frame in self.recording_frames:
            out.write(frame)
            
        # Release writer
        out.release()
        
        self.add_to_log(f"Recording saved as {filename}", "success")
        self.recording_frames = []  # Free memory

    
    def run(self):
        """
        Execute the main body tracker loop.
        Capture video, detect gestures and display results.
        """
        try:
            print("Initializing MediaPipe Pose...")
            with self.mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=0  # 0, 1 or 2 (2 more precise but slower)
            ) as pose:
                
                # Initialize OSC (enabled by default - MODIFIED)
                print("Enabling OSC...")
                self.osc_manager.enable_osc(True)
                
                # Display startup instructions
                print("Adding startup messages to log...")
                self.add_to_log("Body-Sound Vision started", "info")
                self.add_to_log("Press 'H' for help", "info")

                # Verify camera is still open
                print("Checking camera status...")
                if not self.cap.isOpened():
                    print("Camera not open! Trying to re-open...")
                    self.cap = cv2.VideoCapture(self.camera_index)
                    if not self.cap.isOpened():
                        print("ERROR: Could not open camera. Exit.")
                        return
                    
                print("Starting main capture loop...")
                frame_count = 0
                
                while self.cap.isOpened():
                    try:
                        success, image = self.cap.read()
                        if not success:
                            print("Failed to read from camera, retrying...")
                            time.sleep(0.1)
                            continue
                    except Exception as e:
                        print(f"Error during camera read: {str(e)}")
                        self.add_to_log(f"Camera error: {str(e)}", "error")
                        time.sleep(0.1)
                        continue
                    
                    frame_count += 1
                    print(f"Frame {frame_count} captured successfully") if frame_count < 5 else None

                    image = cv2.resize(image, (self.display_width, self.display_height))
                        
                    # Update FPS
                    self.update_fps()
                    
                    # Flip image horizontally for mirror effect
                    image = cv2.flip(image, 1)
                    
                    # Convert image for MediaPipe
                    image.flags.writeable = False
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Detect body keypoints
                    results = pose.process(image)
                    
                    # Reconvert image for OpenCV
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    
                    # Draw HUD
                    image = self.draw_hud(image)
                    
                    # Process pose results
                    if results.pose_landmarks:
                        # Draw body keypoints and connections
                        self.mp_drawing.draw_landmarks(
                            image,
                            results.pose_landmarks,
                            self.mp_pose.POSE_CONNECTIONS,
                            self.mp_drawing_styles.get_default_pose_landmarks_style()
                        )
                        
                        # If in calibration mode, process calibration data
                        if self.calibration_mode:
                            self.process_calibration(results.pose_landmarks)
                            
                        # Otherwise, detect and process gestures
                        else:
                            try:
                                # Detect gestures with new adjusted thresholds
                                gesture_info = self.gesture_recognizer.detect_gestures(results.pose_landmarks)
                                
                                # Process detected gesture
                                self.process_gesture(gesture_info)
                                
                                # Display current gesture
                                gesture_name = gesture_info['name']
                                cv2.putText(image, f"Gesture: {gesture_name}", 
                                        (20, self.height - 20), 
                                        self.font, 0.9, self.colors["success"], 2)
                                
                                # Draw debug values
                                debug_values = self.gesture_recognizer.get_debug_info()
                                image = self.draw_debug_values(image, debug_values)
                                
                            except Exception as e:
                                self.add_to_log(f"Gesture error: {str(e)}", "error")
                                import traceback
                                traceback.print_exc()
                    
                    # Add other interface elements
                    image = self.draw_osc_history(image)
                    image = self.draw_logs(image)
                    image = self.draw_help(image)
                    
                    # Record frame if needed
                    if self.recording:
                        self.recording_frames.append(image.copy())
                        
                        # Limit recording size
                        if len(self.recording_frames) >= self.MAX_RECORDING_FRAMES:
                            self.stop_recording()
                            self.add_to_log("Recording stopped (maximum length reached)", "warning")
                    
                    # Display image
                    cv2.imshow('Body-Sound Vision', image)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(5) & 0xFF
                    if key == 27:  # ESC
                        break
                    elif key == ord('h') or key == ord('H'):
                        self.show_help = not self.show_help
                    elif key == ord('l') or key == ord('L'):
                        self.show_logs = not self.show_logs
                    elif key == ord('d') or key == ord('D'):
                        self.show_debug = not self.show_debug
                    elif key == ord('o') or key == ord('O'):
                        enabled = self.osc_manager.enable_osc(not self.osc_manager.osc_enabled)
                        self.add_to_log(f"OSC {'enabled' if enabled else 'disabled'}", 
                                      "success" if enabled else "info")
                    elif key == ord('a') or key == ord('A'):  # NEW: Audio toggle
                        self.toggle_audio()
                    elif key == ord('v') or key == ord('V'):
                        self.show_osc_history = not self.show_osc_history
                    elif key == ord('c') or key == ord('C'):  # NEW: Start calibration
                        self.start_calibration()
                    elif key == ord('r') or key == ord('R'):
                        if self.recording:
                            self.stop_recording()
                        else:
                            self.start_recording()
                    elif key == ord('s') or key == ord('S'):
                        self.save_recording()
        
        except Exception as e:
            self.add_to_log(f"Critical error: {str(e)}", "error")
            import traceback
            traceback.print_exc()
            
        finally:
            # Clean up resources
            if self.audio_enabled:
                self.audio_manager.stop()
                
            # Release camera and close windows
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
            print("Body-Sound Vision terminated - Resources released")