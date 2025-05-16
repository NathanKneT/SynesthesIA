import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

class GestureRecognizer:
    def __init__(self):
        # Valeurs de seuil pour la détection des gestes
        self.WRIST_SHOULDER_RATIO = 0.2  # Seuil pour les bras levés
        self.CROSS_DISTANCE_THRESHOLD = 0.1  # Seuil pour les bras croisés
        self.T_POSE_HORIZONTAL_THRESHOLD = 0.15  # Seuil pour l'horizontalité des bras en T-pose
        self.T_POSE_EXTENSION_RATIO = 0.3  # Écartement minimal des bras par rapport à la taille du corps pour la T-pose
        
        # Pour le débogage
        self.debug_values = {}
        
    def detect_gestures(self, pose_landmarks):
        """Detect body gestures based on pose landmarks with improved thresholds"""
        # Extract key points (normalized coordinates)
        landmarks = {}
        visibility = {}
        
        for landmark in mp_pose.PoseLandmark:
            landmarks[landmark.name] = np.array([
                pose_landmarks.landmark[landmark].x,
                pose_landmarks.landmark[landmark].y,
                pose_landmarks.landmark[landmark].z
            ])
            visibility[landmark.name] = pose_landmarks.landmark[landmark].visibility
        
        # Clear debug values
        self.debug_values = {}
        
        # Calculer quelques métriques de base
        body_height = np.linalg.norm(landmarks['NOSE'] - landmarks['LEFT_HIP'])  # Approximation de la taille du haut du corps
        shoulder_width = np.linalg.norm(landmarks['LEFT_SHOULDER'] - landmarks['RIGHT_SHOULDER'])
        
        self.debug_values['body_height'] = float(body_height)
        self.debug_values['shoulder_width'] = float(shoulder_width)
        
        # ====== GESTURE 1: Arms raised (POWER UP) ======
        # Détecte si les deux poignets sont significativement au-dessus des épaules
        left_wrist_above = landmarks['LEFT_WRIST'][1] < landmarks['LEFT_SHOULDER'][1] - self.WRIST_SHOULDER_RATIO
        right_wrist_above = landmarks['RIGHT_WRIST'][1] < landmarks['RIGHT_SHOULDER'][1] - self.WRIST_SHOULDER_RATIO
        
        # Les deux bras doivent être levés et les poignets doivent être visibles
        left_wrist_visible = visibility['LEFT_WRIST'] > 0.5
        right_wrist_visible = visibility['RIGHT_WRIST'] > 0.5
        
        # Pour éviter les faux positifs, vérifier que les poignets sont dans une position raisonnable
        left_wrist_reasonable_x = abs(landmarks['LEFT_WRIST'][0] - landmarks['LEFT_SHOULDER'][0]) < 0.3
        right_wrist_reasonable_x = abs(landmarks['RIGHT_WRIST'][0] - landmarks['RIGHT_SHOULDER'][0]) < 0.3
        
        arms_raised = (left_wrist_above and right_wrist_above and 
                      left_wrist_visible and right_wrist_visible and
                      left_wrist_reasonable_x and right_wrist_reasonable_x)
        
        # Stocker les valeurs pour le débogage
        self.debug_values['left_wrist_above'] = left_wrist_above
        self.debug_values['right_wrist_above'] = right_wrist_above
        
        # ====== GESTURE 2: Arms crossed ======
        # Mesurer la distance entre les poignets et vérifier s'ils sont croisés
        wrists_close = np.linalg.norm(landmarks['LEFT_WRIST'] - landmarks['RIGHT_WRIST']) < self.CROSS_DISTANCE_THRESHOLD
        left_wrist_right = landmarks['LEFT_WRIST'][0] > landmarks['NOSE'][0]
        right_wrist_left = landmarks['RIGHT_WRIST'][0] < landmarks['NOSE'][0]
        
        # Les poignets doivent être visibles et à une hauteur similaire (pas un bras en haut et un en bas)
        wrists_similar_height = abs(landmarks['LEFT_WRIST'][1] - landmarks['RIGHT_WRIST'][1]) < 0.15
        
        # Les coudes doivent être pliés
        elbows_bent = (landmarks['LEFT_ELBOW'][0] < landmarks['LEFT_WRIST'][0] and 
                      landmarks['RIGHT_ELBOW'][0] > landmarks['RIGHT_WRIST'][0])
        
        arms_crossed = (wrists_close and left_wrist_visible and right_wrist_visible and 
                       wrists_similar_height and 
                       ((left_wrist_right and right_wrist_left) or elbows_bent))
        
        self.debug_values['wrists_close'] = wrists_close
        self.debug_values['left_wrist_right'] = left_wrist_right
        self.debug_values['right_wrist_left'] = right_wrist_left
        
        # ====== GESTURE 3: T-pose - AMÉLIORATION ======
        # 1. Vérifier l'alignement vertical du corps
        upper_body_aligned = abs(landmarks['NOSE'][0] - landmarks['LEFT_HIP'][0]) < 0.1
        
        # 2. Vérifier que les bras sont horizontaux
        left_arm_horizontal = abs(landmarks['LEFT_WRIST'][1] - landmarks['LEFT_SHOULDER'][1]) < self.T_POSE_HORIZONTAL_THRESHOLD
        right_arm_horizontal = abs(landmarks['RIGHT_WRIST'][1] - landmarks['RIGHT_SHOULDER'][1]) < self.T_POSE_HORIZONTAL_THRESHOLD
        
        # 3. Mesurer l'écartement des bras relatif à la taille du corps
        arms_span = np.linalg.norm(landmarks['LEFT_WRIST'] - landmarks['RIGHT_WRIST'])
        arms_extension_ratio = arms_span / body_height
        arms_sufficiently_extended = arms_extension_ratio > self.T_POSE_EXTENSION_RATIO
        
        # 4. Vérifier que les bras sont étendus de chaque côté
        left_arm_left = landmarks['LEFT_WRIST'][0] < landmarks['LEFT_SHOULDER'][0] - 0.1
        right_arm_right = landmarks['RIGHT_WRIST'][0] > landmarks['RIGHT_SHOULDER'][0] + 0.1
        
        # 5. S'assurer que les points clés sont visibles
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
        
        # Valeurs de débogage pour la T-pose
        self.debug_values['upper_body_aligned'] = upper_body_aligned
        self.debug_values['left_arm_horizontal'] = left_arm_horizontal
        self.debug_values['right_arm_horizontal'] = right_arm_horizontal
        self.debug_values['arms_span'] = float(arms_span)
        self.debug_values['arms_extension_ratio'] = float(arms_extension_ratio)
        self.debug_values['arms_sufficiently_extended'] = arms_sufficiently_extended
        self.debug_values['left_arm_left'] = left_arm_left
        self.debug_values['right_arm_right'] = right_arm_right
        
        # ====== Déterminer le geste ======
        # Priorité des gestes (du plus spécifique au plus général)
        if t_pose:
            return "t_pose"
        elif arms_raised:
            return "power_up"
        elif arms_crossed:
            return "crossed_arms"
        else:
            return "neutral"
    
    def get_debug_info(self):
        """Return debug information for visualization"""
        return self.debug_values