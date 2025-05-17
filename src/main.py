import traceback
import logging
logging.basicConfig(level=logging.DEBUG)

def main():
    print("Starting Body-Sound Vision - Interactive Body-Music Interface")
    print("=" * 60)
    print("Controls:")
    print("  H - Toggle help")
    print("  L - Toggle logs")
    print("  D - Toggle debug information")
    print("  O - Toggle OSC transmission")
    print("  A - Toggle Audio (direct sound)")
    print("  C - Calibrate tracking")
    print("  R - Start/Stop recording")
    print("  S - Save recording")
    print("  ESC - Exit")
    print("=" * 60)
    
    try:
        print("Importing BodyTracker...")
        from tracking.body_tracker import BodyTracker

        print("Initializing BodyTracker...")
        tracker = BodyTracker()
        print("BodyTracker initialized successfully")
        
        print("Starting main loop...")
        tracker.run()
        
    except ImportError as e:
        print(f"ERROR: Failed to import required modules: {e}")
        print("This may be due to missing dependencies. Please check requirements.txt")
        traceback.print_exc()
        
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        traceback.print_exc()
        
    print("Application terminated.")

if __name__ == "__main__":
    main()