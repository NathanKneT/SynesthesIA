import cv2
import mediapipe as mp
import time
from tracking.gesture_recognition import GestureRecognizer

class BodyTracker:
    def __init__(self):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize gesture recognizer
        self.gesture_recognizer = GestureRecognizer()
        
        # Event log history
        self.show_logs = False
        self.event_log = []
        self.MAX_LOG_ENTRIES = 5
        
        # State variables
        self.show_help = False
        self.last_gesture = "none"
        self.gesture_cooldown = 0
    
    def add_to_log(self, message):
        """Add a message to the scrolling event log"""
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        self.event_log.append(f"[{timestamp}] {message}")
        # Keep only the most recent entries
        if len(self.event_log) > self.MAX_LOG_ENTRIES:
            self.event_log.pop(0)


    def run(self):
        try:
            with self.mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as pose:
                
                while self.cap.isOpened():
                    success, image = self.cap.read()
                    if not success:
                        print("Ignoring empty camera frame.")
                        continue
                    
                    if image is None:
                        print("Image is None, skipping frame.")
                        continue
                        
                    image = cv2.flip(image, 1)
                    
                    if image.size == 0:
                        print("Empty image, skipping frame.")
                        continue
                    
                    image.flags.writeable = False
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = pose.process(image)
                    
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    
                    # Create a help overlay
                    if self.show_help:
                        help_overlay = image.copy()
                        cv2.rectangle(help_overlay, (0, 0), (self.width, 180), (0, 0, 0), -1)
                        image = cv2.addWeighted(help_overlay, 0.7, image, 0.3, 0)
                        
                        cv2.putText(image, "BODY GESTURE GUIDE:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.putText(image, "Arms Raised: power_up gesture", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                        cv2.putText(image, "Arms Crossed: crossed_arms gesture", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                        cv2.putText(image, "T-Pose: t_pose gesture", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                        cv2.putText(image, "Press 'H' to toggle help | 'L' to toggle logs | ESC to exit", (self.width-450, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                    
                    # Event log display (only if show_logs is True)
                    if self.show_logs:
                        log_y_start = self.height - 180
                        cv2.rectangle(image, (0, log_y_start), (self.width, self.height), (0, 0, 0), -1)
                        cv2.putText(image, "EVENT LOG:", (10, log_y_start + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        for i, log_entry in enumerate(self.event_log):
                            cv2.putText(image, log_entry, (20, log_y_start + 60 + (i * 25)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    
                    if results.pose_landmarks:
                        self.mp_drawing.draw_landmarks(
                            image,
                            results.pose_landmarks,
                            self.mp_pose.POSE_CONNECTIONS,
                            self.mp_drawing_styles.get_default_pose_landmarks_style())
                        
                        try:
                            current_gesture = self.gesture_recognizer.detect_gestures(results.pose_landmarks)
                            
                            # Toujours afficher le geste actuel, même si les logs sont désactivés
                            cv2.putText(image, f"Current gesture: {current_gesture}", (10, self.height - 210 if self.show_logs else self.height - 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            # N'afficher les valeurs de débogage que si les logs sont activés
                            if self.show_logs:
                                debug_values = self.gesture_recognizer.get_debug_info()
                                debug_y_start = 250
                                cv2.putText(image, "DEBUG VALUES:", (10, debug_y_start), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                                
                                top_debug_values = list(debug_values.items())[:10]
                                
                                for i, (key, value) in enumerate(top_debug_values):
                                    if isinstance(value, bool):
                                        color = (0, 255, 0) if value else (0, 0, 255)
                                    else:
                                        color = (255, 255, 255)
                                    
                                    cv2.putText(image, f"{key}: {value}", (20, debug_y_start + 30 + (i * 25)), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                            
                            # Toujours suivre les changements de geste pour les logs, même s'ils ne sont pas affichés
                            if current_gesture != self.last_gesture and self.gesture_cooldown <= 0:
                                self.add_to_log(f"Detected gesture: {current_gesture}")
                                self.last_gesture = current_gesture
                                self.gesture_cooldown = 15
                            
                            if self.gesture_cooldown > 0:
                                self.gesture_cooldown -= 1
                        
                        except Exception as e:
                            print(f"Error processing pose: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    # Check for key presses
                    key = cv2.waitKey(5) & 0xFF
                    if key == 27:  # ESC key to exit
                        break
                    elif key == ord('h') or key == ord('H'):  # 'H' key to toggle help
                        self.show_help = not self.show_help
                    elif key == ord('l') or key == ord('L'):  # 'L' key to toggle logs
                        self.show_logs = not self.show_logs
                    
                    # Display the resulting frame
                    cv2.imshow('Body Gesture Detection', image)
                    
        except Exception as e:
            print(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
            print("Resources released successfully")