"""
AI Guard Agent - Milestone 1 + BONUS: SpeechBrain Speaker Verification
========================================================================
Uses SpeechBrain's pre-trained speaker recognition model.

Requirements:
pip install SpeechRecognition pyaudio opencv-python numpy speechbrain torch torchaudio
"""

import speech_recognition as sr
import cv2
import time
from datetime import datetime
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# For speaker verification
try:
    from speechbrain.pretrained import EncoderClassifier
    import torch
    SPEAKER_VERIFY_AVAILABLE = True
except ImportError:
    SPEAKER_VERIFY_AVAILABLE = False
    print("⚠️  SpeechBrain not available.")
    print("   Install: pip install speechbrain torch torchaudio")


class SpeakerVerifier:
    """Speaker verification using SpeechBrain"""
    
    def __init__(self, model_path='speaker_model.pkl'):
        self.model_path = model_path
        self.classifier = None
        self.enrolled_embedding = None
        self.enrolled = False
        self.sample_rate = 16000
        
        if SPEAKER_VERIFY_AVAILABLE:
            print("Loading SpeechBrain model (first time may take a moment)...")
            from speechbrain.utils.fetching import LocalStrategy
            self.classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="tmp_model",
                local_strategy=LocalStrategy.COPY
            )
            print("✓ Model loaded")
        
    def extract_embedding(self, audio_data):
        """Extract voice embedding using SpeechBrain"""
        try:
            # Convert to torch tensor
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            audio_tensor = torch.tensor(audio_data).unsqueeze(0)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.classifier.encode_batch(audio_tensor)
                embedding = embedding.squeeze().cpu().numpy()
            
            return embedding
            
        except Exception as e:
            print(f"❌ Embedding error: {e}")
            return None
    
    def enroll_speaker(self, audio_samples):
        """Enroll speaker by averaging embeddings"""
        if len(audio_samples) < 3:
            print("⚠️  Need at least 3 samples")
            return False
        
        print(f"Extracting embeddings from {len(audio_samples)} samples...")
        
        embeddings = []
        for i, audio in enumerate(audio_samples):
            print(f"  Processing {i+1}/{len(audio_samples)}...")
            embedding = self.extract_embedding(audio)
            if embedding is not None:
                embeddings.append(embedding)
        
        if len(embeddings) < 3:
            print("❌ Not enough valid samples")
            return False
        
        # Average embeddings
        self.enrolled_embedding = np.mean(embeddings, axis=0)
        self.enrolled = True
        self.save_model()
        
        print("✓ Enrolled!")
        return True
    
    def verify_speaker(self, audio_data, threshold=0.7):
        """Verify speaker using SpeechBrain's scoring function"""
        if not self.enrolled or self.enrolled_embedding is None:
            return True, 0.0
        
        test_embedding = self.extract_embedding(audio_data)
        if test_embedding is None:
            return False, 0.0
        
        # Convert to torch tensors
        enrolled_tensor = torch.tensor(self.enrolled_embedding).unsqueeze(0)
        test_tensor = torch.tensor(test_embedding).unsqueeze(0)
        
        # Use SpeechBrain's similarity scoring (outputs between -1 and 1)
        with torch.no_grad():
            score = torch.nn.functional.cosine_similarity(
                enrolled_tensor, test_tensor
            ).item()
        
        is_owner = score > threshold
        
        return is_owner, float(score)
    
    def save_model(self):
        """Save enrolled embedding"""
        if self.enrolled_embedding is not None:
            try:
                if os.path.exists(self.model_path):
                    os.remove(self.model_path)
                
                with open(self.model_path, 'wb') as f:
                    pickle.dump({
                        'embedding': self.enrolled_embedding,
                        'enrolled': self.enrolled
                    }, f)
                print(f"✓ Saved to {self.model_path}")
            except Exception as e:
                print(f"❌ Save error: {e}")
    
    def load_model(self):
        """Load enrolled embedding"""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                    
                if 'embedding' in data:
                    self.enrolled_embedding = data['embedding']
                    self.enrolled = data['enrolled']
                    print(f"✓ Loaded from {self.model_path}")
                    return True
                else:
                    os.remove(self.model_path)
                    return False
                        
            except Exception as e:
                print(f"❌ Load error: {e}")
                try:
                    os.remove(self.model_path)
                except:
                    pass
                return False
        return False
    
    def is_enrolled(self):
        return self.enrolled


class GuardAgent:
    """AI Guard Agent with voice activation and speaker verification"""
    
    def __init__(self, use_speaker_verification=False):
        self.guard_mode = False
        self.running = False
        self.webcam = None
        self.recognizer = None
        self.microphone = None
        self.sample_rate = 16000
        
        self.use_speaker_verification = use_speaker_verification and SPEAKER_VERIFY_AVAILABLE
        self.speaker_verifier = None
        
        if self.use_speaker_verification:
            self.speaker_verifier = SpeakerVerifier()
            self.speaker_verifier.load_model()
        
        self.activation_phrases = ["guard my room", "start guard", "activate guard"]
        self.deactivation_phrases = ["stop guard", "deactivate guard", "end guard"]
    
    def initialize_audio(self):
        if self.recognizer is None:
            try:
                self.recognizer = sr.Recognizer()
                self.microphone = sr.Microphone()
                
                print("Calibrating microphone...")
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=2)
                print("✓ Audio initialized\n")
                return True
            except Exception as e:
                print(f"❌ Audio init failed: {e}")
                return False
        return True
    
    def record_audio_sample(self, duration=3):
        if not self.initialize_audio():
            return None
        
        try:
            with self.microphone as source:
                print(f"🎤 Recording for {duration}s... Speak naturally!")
                audio = self.recognizer.record(source, duration=duration)
            
            audio_data = np.frombuffer(audio.get_wav_data(), dtype=np.int16)
            audio_data = audio_data.astype(np.float32) / 32768.0
            
            return audio_data
            
        except Exception as e:
            print(f"❌ Recording error: {e}")
            return None
    
    def enroll_owner_voice(self, num_samples=5):
        if not self.use_speaker_verification:
            print("⚠️  Speaker verification not enabled")
            return False
        
        print("\n" + "="*60)
        print("OWNER VOICE ENROLLMENT")
        print("="*60)
        
        if os.path.exists(self.speaker_verifier.model_path):
            print("⚠️  Existing model found")
            overwrite = input("   Overwrite? (y/n): ").strip().lower()
            if overwrite != 'y':
                print("   Cancelled")
                return False
        
        print(f"\nRecord {num_samples} samples (3s each)")
        print("Tip: Speak naturally, vary phrases")
        print("="*60 + "\n")
        
        audio_samples = []
        
        for i in range(1, num_samples + 1):
            print(f"\nSample {i}/{num_samples}")
            input("Press Enter to record...")
            
            audio = self.record_audio_sample(duration=3)
            if audio is not None:
                print("✓ Recorded")
                audio_samples.append(audio)
            else:
                print("❌ Failed")
            
            time.sleep(0.5)
        
        if len(audio_samples) >= 3:
            success = self.speaker_verifier.enroll_speaker(audio_samples)
            if success:
                print("\n✅ Enrollment complete!\n")
                return True
        else:
            print("\n❌ Failed - not enough samples")
        
        return False
    
    def listen_for_command(self, timeout=5):
        """Listen for speech command (for activation only)"""
        if not self.initialize_audio():
            return None
            
        try:
            with self.microphone as source:
                print("🎤 Listening...")
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=5)
                
            print("⚙️  Processing...")
            text = self.recognizer.recognize_google(audio).lower()
            print(f"✓ Heard: '{text}'")
            return text
            
        except sr.WaitTimeoutError:
            print("⏱️  Timeout")
        except sr.UnknownValueError:
            print("❌ Could not understand")
        except sr.RequestError as e:
            print(f"❌ Recognition error: {e}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        return None
    
    def listen_to_intruder(self, timeout=5, phrase_time_limit=10):
        """
        Listen to what intruder is saying (for Milestone 3 integration).
        This allows the dialogue system to generate contextual responses.
        
        Args:
            timeout: Max seconds to wait for speech
            phrase_time_limit: Max seconds for the phrase
            
        Returns:
            str: What the intruder said, or None if nothing heard
        """
        if not self.initialize_audio():
            return None
        
        try:
            with self.microphone as source:
                print("🎤 Listening to intruder...")
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            
            print("⚙️  Processing intruder speech...")
            text = self.recognizer.recognize_google(audio).lower()
            print(f"👤 Intruder said: '{text}'")
            return text
            
        except sr.WaitTimeoutError:
            print("⏱️  Intruder didn't speak")
            return None
        except sr.UnknownValueError:
            print("❌ Could not understand intruder")
            return None
        except sr.RequestError as e:
            print(f"❌ Recognition error: {e}")
            return None
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def listen_for_deactivation_with_verification(self):
        """Special method for deactivation with speaker verification - records 3 seconds"""
        if not self.initialize_audio():
            return None, None
        
        try:
            with self.microphone as source:
                print("🎤 Say command + keep talking for 3 seconds total...")
                # Use same method as enrollment - fixed 3 second recording
                audio = self.recognizer.record(source, duration=3)
            
            print("⚙️  Processing...")
            text = self.recognizer.recognize_google(audio).lower()
            print(f"✓ Heard: '{text}'")
            
            # Get audio data for verification
            audio_data = np.frombuffer(audio.get_wav_data(), dtype=np.int16)
            audio_data = audio_data.astype(np.float32) / 32768.0
            
            return text, audio_data
            
        except sr.UnknownValueError:
            print("❌ Could not understand")
        except sr.RequestError as e:
            print(f"❌ Recognition error: {e}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        return None, None
    
    def check_activation_command(self, text):
        """Check for activation command with fuzzy matching"""
        if text is None:
            return False
        
        # Exact match
        if any(phrase in text for phrase in self.activation_phrases):
            return True
        
        # Fuzzy match for common mishears
        fuzzy_words = [
            ("guard", "room"), ("god", "room"), ("card", "room"), ("yard", "room"), ("gart", "room"),
            ("start", "guard"), ("start", "god"), ("start", "card"), ("start", "yard"),
            ("activate", "guard"), ("activate", "god"), ("activate", "card"), ("activate", "yard")
        ]
        
        # Check if both words appear in the text
        for word1, word2 in fuzzy_words:
            if word1 in text and word2 in text:
                return True
        
        return False
    
    def check_deactivation_command(self, text):
        """Check for deactivation command with fuzzy matching"""
        if text is None:
            return False
        
        # Exact match
        if any(phrase in text for phrase in self.deactivation_phrases):
            return True
        
        # Fuzzy match for common mishears - check if both words appear
        fuzzy_words = [
            ("stop", "guard"), ("stop", "god"), ("stop", "card"), ("stop", "yard"), ("stop", "gart"),
            ("end", "guard"), ("end", "god"), ("end", "card"), ("end", "yard"), ("end", "gart"),
            ("deactivate", "guard"), ("deactivate", "god"), ("deactivate", "card"), 
            ("deactivate", "yard"), ("deactivate", "gart"),
            ("stop", "guardian"), ("end", "guardian"), ("deactivate", "guardian")
        ]
        
        # Check if both words appear in the text (in any order, with words between)
        for word1, word2 in fuzzy_words:
            if word1 in text and word2 in text:
                return True
        
        return False
    
    def activate_guard_mode(self):
        self.guard_mode = True
        print("\n" + "="*50)
        print("🛡️  GUARD MODE ACTIVATED 🛡️")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*50 + "\n")
        self.init_webcam()
    
    def deactivate_guard_mode(self):
        self.guard_mode = False
        print("\n" + "="*50)
        print("⏹️  GUARD MODE DEACTIVATED")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*50 + "\n")
        self.release_webcam()
    
    def init_webcam(self):
        try:
            if self.webcam is None or not self.webcam.isOpened():
                self.webcam = cv2.VideoCapture(0)
                self.webcam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                if self.webcam.isOpened():
                    print("✓ Webcam ready (use option 8 to view)")
                else:
                    print("✗ Webcam failed")
        except Exception as e:
            print(f"✗ Webcam error: {e}")
    
    def release_webcam(self):
        if self.webcam is not None:
            try:
                cv2.destroyAllWindows()
                for _ in range(10):
                    cv2.waitKey(1)
                
                if self.webcam.isOpened():
                    self.webcam.release()
                
                self.webcam = None
                print("✓ Webcam released")
                
            except:
                try:
                    if self.webcam is not None:
                        self.webcam.release()
                    self.webcam = None
                except:
                    pass
    
    def show_webcam_feed(self, duration=10):
        if not self.guard_mode:
            print("⚠️  Guard mode not active")
            return
        
        if self.webcam is None or not self.webcam.isOpened():
            print("⚠️  Webcam unavailable")
            return
        
        print(f"\n📹 Showing feed for {duration}s (press 'q' to close)\n")
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                ret, frame = self.webcam.read()
                
                if ret:
                    cv2.putText(frame, "GUARD MODE: ACTIVE", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    if self.use_speaker_verification and self.speaker_verifier.is_enrolled():
                        cv2.putText(frame, "Speaker Verification: ON", (10, 90),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    cv2.putText(frame, timestamp, (10, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    cv2.imshow('AI Guard - Webcam Feed', frame)
                    
                    if cv2.waitKey(30) & 0xFF == ord('q'):
                        break
                else:
                    break
            
            cv2.destroyAllWindows()
            for _ in range(10):
                cv2.waitKey(1)
            
            print("✓ Feed closed\n")
            
        except Exception as e:
            print(f"Error: {e}")
            cv2.destroyAllWindows()
    
    def show_menu(self):
        print("\n" + "="*60)
        print("AI GUARD AGENT - MENU")
        print("="*60)
        if self.guard_mode:
            print("Status: 🛡️  GUARD MODE ACTIVE")
        else:
            print("Status: ⏸️  Guard Mode Inactive")
        
        if self.use_speaker_verification:
            if self.speaker_verifier.is_enrolled():
                print("Speaker Verification: ✅ Enrolled")
            else:
                print("Speaker Verification: ⚠️  Not Enrolled")
        
        print("\nOptions:")
        print("1. Listen for voice activation")
        print("2. Listen for voice deactivation")
        print("3. Manually activate guard")
        print("4. Manually deactivate guard")
        
        if self.use_speaker_verification:
            print("5. Enroll owner voice (BONUS)")
            print("6. Test speaker verification")
        
        print("7. Check status")
        
        if self.guard_mode:
            print("8. Show webcam feed (10s)")
        
        print("9. Exit")
        print("="*60)
    
    def listen_for_activation(self):
        print("\n" + "="*50)
        print("LISTENING FOR ACTIVATION")
        print("="*50)
        print("\nSay one of these:")
        for cmd in self.activation_phrases:
            print(f"  - '{cmd}'")
        print("\nPress Ctrl+C to cancel")
        print("="*50 + "\n")
        
        try:
            for attempt in range(1, 4):
                print(f"Attempt {attempt}/3")
                command = self.listen_for_command(timeout=5)
                
                if self.check_activation_command(command):
                    self.activate_guard_mode()
                    time.sleep(1)
                    return
                elif command:
                    print("Not an activation command. Try again...\n")
            
            print("\nMax attempts reached.\n")
            time.sleep(1)
        
        except KeyboardInterrupt:
            print("\n\n✓ Cancelled.\n")
            time.sleep(1)
    
    def listen_for_deactivation(self):
        print("\n" + "="*50)
        print("LISTENING FOR DEACTIVATION")
        print("="*50)
        print("\nSay one of these:")
        for cmd in self.deactivation_phrases:
            print(f"  - '{cmd}'")
        
        if self.use_speaker_verification and self.speaker_verifier.is_enrolled():
            print("\n⚠️  Speaker verification is ACTIVE")
            print("   Only enrolled owner can deactivate!")
            print("   📢 IMPORTANT: Say command + keep talking for 3 seconds total")
        
        print("\nPress Ctrl+C to cancel")
        print("="*50 + "\n")
        
        try:
            for attempt in range(1, 4):
                print(f"Attempt {attempt}/3")
                
                if self.use_speaker_verification and self.speaker_verifier.is_enrolled():
                    # USE THE SPECIAL 3-SECOND METHOD
                    command, audio_data = self.listen_for_deactivation_with_verification()
                else:
                    command = self.listen_for_command(timeout=5)
                    audio_data = None
                
                if self.check_deactivation_command(command):
                    # Verify speaker if enabled
                    if self.use_speaker_verification and self.speaker_verifier.is_enrolled():
                        if audio_data is not None:
                            print("🔍 Verifying speaker identity...")
                            is_owner, similarity = self.speaker_verifier.verify_speaker(audio_data)
                            
                            print(f"   Similarity score: {similarity:.3f} (threshold: 0.70)")
                            
                            if is_owner:
                                print("✅ Speaker verified as OWNER")
                                self.deactivate_guard_mode()
                                time.sleep(1)
                                return
                            else:
                                print("❌ Speaker NOT recognized as owner!")
                                print("   Deactivation DENIED for security.\n")
                                continue
                    else:
                        # No verification needed
                        self.deactivate_guard_mode()
                        time.sleep(1)
                        return
                elif command:
                    print("Not a deactivation command. Try again...\n")
            
            print("\nMax attempts reached.\n")
            time.sleep(1)
        
        except KeyboardInterrupt:
            print("\n\n✓ Cancelled.\n")
            time.sleep(1)
    
    def test_speaker_verification(self):
        if not self.use_speaker_verification:
            print("\n⚠️  Speaker verification not enabled\n")
            return
        
        if not self.speaker_verifier.is_enrolled():
            print("\n⚠️  Please enroll first (option 5)\n")
            return
        
        print("\n" + "="*50)
        print("SPEAKER VERIFICATION TEST")
        print("="*50)
        print("\nSpeak for 3 seconds...")
        print("="*50 + "\n")
        
        input("Press Enter to start...")
        
        try:
            audio_data = self.record_audio_sample(duration=3)
            
            if audio_data is not None:
                print("\n🔍 Verifying...")
                is_owner, similarity = self.speaker_verifier.verify_speaker(audio_data)
                
                print(f"\nSimilarity score: {similarity:.3f} (threshold: 0.70)")
                
                if is_owner:
                    print("✅ Result: OWNER verified\n")
                else:
                    print("❌ Result: NOT the owner\n")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("   Try re-enrolling (option 5)\n")
        
        input("Press Enter to continue...")
    
    def cleanup(self):
        print("\nCleaning up resources...")
        self.running = False
        self.release_webcam()
        print("✓ Cleanup complete!")
    
    def run(self):
        self.running = True
        
        print("\n" + "="*60)
        print("AI GUARD AGENT - MILESTONE 1 + BONUS")
        print("="*60)
        print("Welcome! Use the menu to control the guard system.\n")
        
        try:
            while self.running:
                self.show_menu()
                choice = input("\nEnter your choice: ").strip()
                
                if choice == "1":
                    if not self.guard_mode:
                        self.listen_for_activation()
                    else:
                        print("\n⚠️  Guard mode already active!\n")
                        time.sleep(1)
                
                elif choice == "2":
                    if self.guard_mode:
                        self.listen_for_deactivation()
                    else:
                        print("\n⚠️  Guard mode not active!\n")
                        time.sleep(1)
                
                elif choice == "3":
                    if not self.guard_mode:
                        self.activate_guard_mode()
                    else:
                        print("\n⚠️  Already active!\n")
                    time.sleep(1)
                
                elif choice == "4":
                    if self.guard_mode:
                        self.deactivate_guard_mode()
                    else:
                        print("\n⚠️  Already inactive!\n")
                    time.sleep(1)
                
                elif choice == "5" and self.use_speaker_verification:
                    self.enroll_owner_voice()
                
                elif choice == "6" and self.use_speaker_verification:
                    self.test_speaker_verification()
                
                elif choice == "7":
                    if self.guard_mode:
                        print("\n✓ Status: Guard mode is ACTIVE 🛡️\n")
                    else:
                        print("\n✓ Status: Guard mode is INACTIVE ⏸️\n")
                    
                    if self.use_speaker_verification and self.speaker_verifier.is_enrolled():
                        print("✓ Speaker verification: ENROLLED\n")
                    time.sleep(1)
                
                elif choice == "8" and self.guard_mode:
                    self.show_webcam_feed(duration=10)
                
                elif choice == "9":
                    print("\nExiting...")
                    self.running = False
                
                else:
                    print("\n❌ Invalid choice.\n")
                    time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n\nShutting down...")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        finally:
            self.cleanup()
            print("Goodbye!")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("AI GUARD AGENT - MILESTONE 1 + BONUS")
    print("="*60)
    print("\nChoose mode:")
    print("1. Basic mode (Milestone 1 only)")
    print("2. Enhanced mode with Speaker Verification (BONUS)")
    print()
    
    mode = input("Enter choice (1-2): ").strip()
    
    if mode == "2":
        if not SPEAKER_VERIFY_AVAILABLE:
            print("\n❌ SpeechBrain not available!")
            print("   Install: pip install speechbrain torch torchaudio\n")
            time.sleep(2)
            agent = GuardAgent(use_speaker_verification=False)
        else:
            print("\n✓ Speaker Verification enabled!\n")
            agent = GuardAgent(use_speaker_verification=True)
            
            if not agent.speaker_verifier.is_enrolled():
                print("⚠️  Owner voice not enrolled yet.")
                enroll = input("   Enroll now? (y/n): ").strip().lower()
                if enroll == 'y':
                    agent.enroll_owner_voice()
    else:
        print("\n✓ Basic mode\n")
        agent = GuardAgent(use_speaker_verification=False)
    
    time.sleep(1)
    agent.run()