import numpy as np
import sounddevice as sd
import threading
import time
from queue import Queue
from typing import Dict, Any, Optional

class AudioManager:
    """
    Audio manager for Body-Sound Vision.
    Generates sounds in response to detected gestures.
    """
    
    def __init__(self):
        # Audio parameters
        self.sample_rate = 44100
        self.volume = 0.5
        self.sound_queue = Queue()
        self.is_playing = False
        self.audio_thread = None
        
        # Sounds for each instrument
        self.instruments = {
            "snare": self._create_snare,
            "kick": self._create_kick,
            "hihat": self._create_hihat,
            "bass": self._create_bass
        }
        
        # Sounds for each gesture
        self.gesture_sounds = {
            "power_up": self._create_power_up,
            "crossed_arms": self._create_crossed_arms,
            "t_pose": self._create_t_pose,
            "tap_left": self._create_snare,  # Use snare for tap_left
            "tap_right": self._create_kick   # Use kick for tap_right
        }
    
    def _create_snare(self, duration=0.3, intensity=0.8):
        """Create a snare drum sound"""
        try:
            t = np.linspace(0, duration, int(self.sample_rate * duration), False)
            noise = np.random.normal(0, 1, int(self.sample_rate * duration))
            noise *= np.exp(-t * 15)  # Fast decay
            return noise * intensity * self.volume
        except Exception as e:
            print(f"Error creating snare sound: {str(e)}")
            # Return silent sound in case of error
            return np.zeros(int(self.sample_rate * duration))
    
    def _create_kick(self, duration=0.4, intensity=0.8):
        """Create a kick drum sound"""
        try:
            t = np.linspace(0, duration, int(self.sample_rate * duration), False)
            freq = 150 * np.exp(-t * 10)  # Decreasing frequency
            sound = np.sin(2 * np.pi * freq * t)
            sound *= np.exp(-t * 8)  # Decay
            return sound * intensity * self.volume
        except Exception as e:
            print(f"Error creating kick sound: {str(e)}")
            return np.zeros(int(self.sample_rate * duration))
    
    def _create_hihat(self, duration=0.2, intensity=0.6):
        """Create a hi-hat sound"""
        try:
            t = np.linspace(0, duration, int(self.sample_rate * duration), False)
            noise = np.random.normal(0, 1, int(self.sample_rate * duration))
            noise *= np.exp(-t * 25)  # Very fast decay
            filtered = np.cumsum(noise) * 0.1  # Simple high-pass filter
            filtered -= np.mean(filtered)
            return filtered * intensity * self.volume
        except Exception as e:
            print(f"Error creating hihat sound: {str(e)}")
            return np.zeros(int(self.sample_rate * duration))
    
    def _create_bass(self, duration=0.5, intensity=0.9):
        """Create a bass sound"""
        try:
            t = np.linspace(0, duration, int(self.sample_rate * duration), False)
            freq = 80
            sound = np.sin(2 * np.pi * freq * t)
            sound *= np.exp(-t * 5)  # Slow decay
            return sound * intensity * self.volume
        except Exception as e:
            print(f"Error creating bass sound: {str(e)}")
            return np.zeros(int(self.sample_rate * duration))
    
    def _create_power_up(self, duration=1.0, intensity=0.7):
        """Create an ascending sound for power_up gesture"""
        try:
            t = np.linspace(0, duration, int(self.sample_rate * duration), False)
            freq = 100 + 300 * t  # Increasing frequency
            sound = np.sin(2 * np.pi * freq * t)
            sound *= np.exp(-t * 2) * 0.8 + 0.2  # Slower decay
            return sound * intensity * self.volume
        except Exception as e:
            print(f"Error creating power_up sound: {str(e)}")
            return np.zeros(int(self.sample_rate * duration))
    
    def _create_crossed_arms(self, duration=1.0, intensity=0.7):
        """Create a descending sound for crossed_arms gesture"""
        try:
            t = np.linspace(0, duration, int(self.sample_rate * duration), False)
            freq = 400 - 300 * t  # Decreasing frequency
            sound = np.sin(2 * np.pi * freq * t)
            sound *= np.exp(-t * 2) * 0.8 + 0.2  # Slower decay
            return sound * intensity * self.volume
        except Exception as e:
            print(f"Error creating crossed_arms sound: {str(e)}")
            return np.zeros(int(self.sample_rate * duration))
    
    def _create_t_pose(self, duration=1.0, intensity=0.7):
        """Create a special sound for t_pose gesture"""
        try:
            t = np.linspace(0, duration, int(self.sample_rate * duration), False)
            freq1 = 200 + 50 * np.sin(2 * np.pi * 2 * t)  # Vibrato
            freq2 = 300 + 50 * np.sin(2 * np.pi * 3 * t)  # Vibrato
            sound = 0.5 * np.sin(2 * np.pi * freq1 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t)
            sound *= np.exp(-t * 2) * 0.8 + 0.2  # Slower decay
            return sound * intensity * self.volume
        except Exception as e:
            print(f"Error creating t_pose sound: {str(e)}")
            return np.zeros(int(self.sample_rate * duration))
    
    def _audio_callback(self, outdata, frames, time, status):
        """Callback for audio output"""
        try:
            if self.sound_queue.empty():
                outdata.fill(0)  # Silence when queue is empty
            else:
                sound = self.sound_queue.get()
                if len(sound) < frames:
                    outdata[:len(sound), 0] = sound
                    outdata[len(sound):, 0] = 0
                else:
                    outdata[:, 0] = sound[:frames]
        except Exception as e:
            print(f"Audio callback error: {str(e)}")
            outdata.fill(0)  # In case of error, produce silence
    
    def _audio_thread_func(self):
        """Function for audio thread"""
        try:
            with sd.OutputStream(channels=1, callback=self._audio_callback, samplerate=self.sample_rate):
                while self.is_playing:
                    time.sleep(0.1)  # Sleep to reduce CPU usage
        except Exception as e:
            print(f"Audio thread error: {str(e)}")
            self.is_playing = False
    
    def start(self):
        """Start audio generator"""
        try:
            if not self.is_playing:
                self.is_playing = True
                self.audio_thread = threading.Thread(target=self._audio_thread_func)
                self.audio_thread.daemon = True
                self.audio_thread.start()
                print("Audio generator started")
                return True
            return False
        except Exception as e:
            print(f"Error starting audio: {str(e)}")
            self.is_playing = False
            return False
    
    def stop(self):
        """Stop audio generator"""
        try:
            if self.is_playing:
                self.is_playing = False
                if self.audio_thread:
                    self.audio_thread.join(timeout=1.0)
                    self.audio_thread = None
                print("Audio generator stopped")
                return True
            return False
        except Exception as e:
            print(f"Error stopping audio: {str(e)}")
            self.is_playing = False
            return False
    
    def play_instrument(self, instrument, intensity=0.8):
        """Play a specific instrument"""
        try:
            if instrument in self.instruments:
                sound = self.instruments[instrument](intensity=intensity)
                self.sound_queue.put(sound)
                return True
            return False
        except Exception as e:
            print(f"Error playing instrument {instrument}: {str(e)}")
            return False
    
    def play_gesture_sound(self, gesture_name, intensity=0.7):
        """Play a sound associated with a gesture"""
        try:
            if gesture_name in self.gesture_sounds:
                sound = self.gesture_sounds[gesture_name](intensity=intensity)
                self.sound_queue.put(sound)
                return True
            return False
        except Exception as e:
            print(f"Error playing gesture sound {gesture_name}: {str(e)}")
            return False
    
    def process_gesture(self, gesture_info: Dict[str, Any]):
        """
        Process a gesture and play the appropriate sound
        
        Args:
            gesture_info: Dictionary containing gesture information
        """
        try:
            gesture_name = gesture_info.get('name')
            parameters = gesture_info.get('parameters', {})
            
            # Ignore neutral gesture
            if gesture_name == "neutral":
                return False
                
            # Get intensity (or use default value)
            intensity = float(parameters.get('intensity', 0.7))
            
            # For tap gestures, use selected instrument
            if gesture_name in ["tap_left", "tap_right"]:
                instrument = parameters.get('instrument', parameters.get('active_instrument', 'snare'))
                return self.play_instrument(instrument, intensity)
            else:
                # For other gestures, use gesture-associated sound
                return self.play_gesture_sound(gesture_name, intensity)
        except Exception as e:
            print(f"Error processing gesture: {str(e)}")
            return False