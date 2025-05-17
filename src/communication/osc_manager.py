"""
OSC (Open Sound Control) communication management module for Body-Sound Vision.
Allows sending OSC messages to sound and visual generation modules.
"""

import time
from typing import Dict, Any, List, Optional, Union
from pythonosc import udp_client
from pythonosc.osc_message_builder import OscMessageBuilder

class OscManager:
    """
    OSC communication manager for sending messages
    to sound and visual generation modules.
    """
    def __init__(self):
        # Default configuration
        self.osc_enabled = False
        self.osc_target_ip = "127.0.0.1"
        self.audio_port = 8000  # Port for audio module
        self.visual_port = 8001  # Port for visual module
        self.log_messages = True
        
        # OSC clients for audio and visual
        self.audio_client = None
        self.visual_client = None
        
        # Message history for debugging
        self.message_history = []
        self.MAX_HISTORY = 20
        
    def enable_osc(self, enabled=True):
        """Enable or disable OSC sending"""
        if enabled == self.osc_enabled:
            return  # No change
            
        self.osc_enabled = enabled
        
        if enabled:
            try:
                # Create OSC clients
                self.audio_client = udp_client.SimpleUDPClient(
                    self.osc_target_ip, self.audio_port)
                self.visual_client = udp_client.SimpleUDPClient(
                    self.osc_target_ip, self.visual_port)
                print(f"OSC enabled: Audio ({self.osc_target_ip}:{self.audio_port}), "
                      f"Visual ({self.osc_target_ip}:{self.visual_port})")
            except Exception as e:
                print(f"Error enabling OSC: {e}")
                self.osc_enabled = False
        else:
            # Close connections
            self.audio_client = None
            self.visual_client = None
            print("OSC disabled")
        
        return self.osc_enabled
        
    def set_audio_target(self, ip, port):
        """Set target address for audio OSC messages"""
        self.osc_target_ip = ip
        self.audio_port = port
        if self.osc_enabled:
            self.audio_client = udp_client.SimpleUDPClient(ip, port)
        print(f"Audio OSC target set to {ip}:{port}")
        
    def set_visual_target(self, ip, port):
        """Set target address for visual OSC messages"""
        self.osc_target_ip = ip
        self.visual_port = port
        if self.osc_enabled:
            self.visual_client = udp_client.SimpleUDPClient(ip, port)
        print(f"Visual OSC target set to {ip}:{port}")
        
    def _add_to_history(self, destination: str, address: str, args: List[Any]):
        """Add a message to history for debugging"""
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        args_str = ", ".join([str(arg) for arg in args]) if args else "no args"
        message = f"[{timestamp}] {destination}: {address} [{args_str}]"
        
        self.message_history.append(message)
        if len(self.message_history) > self.MAX_HISTORY:
            self.message_history.pop(0)
            
        if self.log_messages:
            print(message)
        
    def send_gesture(self, gesture_info: Dict[str, Any]):
        """
        Send gesture information via OSC to audio and visual modules
        
        Args:
            gesture_info: Dictionary containing gesture information
                - 'type': Gesture type (Enum)
                - 'name': Gesture name (str)
                - 'parameters': Parameters associated with the gesture (Dict)
        """
        if not self.osc_enabled or (self.audio_client is None and self.visual_client is None):
            return False
            
        # Extract gesture information
        gesture_name = gesture_info['name']
        parameters = gesture_info.get('parameters', {})
        
        # === Audio sending ===
        if self.audio_client:
            # Build OSC address based on gesture
            audio_address = f"/gesture/{gesture_name}/audio"
            
            # Filter relevant audio parameters
            audio_params = []
            
            # Add audio-specific parameters
            if 'frequency' in parameters:
                audio_params.append(parameters['frequency'])
            if 'instrument' in parameters:
                audio_params.append(parameters['instrument'])
            if 'intensity' in parameters:
                audio_params.append(float(parameters['intensity']))
                
            # Send message
            try:
                self.audio_client.send_message(audio_address, audio_params)
                self._add_to_history("AUDIO", audio_address, audio_params)
            except Exception as e:
                print(f"Error sending audio OSC message: {e}")
        
        # === Visual sending ===
        if self.visual_client:
            # Build OSC address based on gesture
            visual_address = f"/gesture/{gesture_name}/visual"
            
            # Filter relevant visual parameters
            visual_params = []
            
            # Add visual-specific parameters
            if 'visual' in parameters:
                visual_params.append(parameters['visual'])
            if 'intensity' in parameters:
                visual_params.append(float(parameters['intensity']))
            if 'color' in parameters:
                visual_params.extend(parameters['color'])  # RGB or HSV values
                
            # Send message
            try:
                self.visual_client.send_message(visual_address, visual_params)
                self._add_to_history("VISUAL", visual_address, visual_params)
            except Exception as e:
                print(f"Error sending visual OSC message: {e}")
                
        return True
    
    def send_custom_message(self, destination: str, address: str, *args):
        """
        Send a custom OSC message to a specific destination
        
        Args:
            destination: "audio" or "visual"
            address: OSC address (e.g. /control/volume)
            *args: Arguments to send with the message
        """
        if not self.osc_enabled:
            return False
            
        if destination.lower() == "audio" and self.audio_client:
            try:
                self.audio_client.send_message(address, args)
                self._add_to_history("AUDIO", address, args)
                return True
            except Exception as e:
                print(f"Error sending custom audio OSC message: {e}")
        
        elif destination.lower() == "visual" and self.visual_client:
            try:
                self.visual_client.send_message(address, args)
                self._add_to_history("VISUAL", address, args)
                return True
            except Exception as e:
                print(f"Error sending custom visual OSC message: {e}")
                
        return False
    
    def get_message_history(self) -> List[str]:
        """Return message history for debugging"""
        return self.message_history