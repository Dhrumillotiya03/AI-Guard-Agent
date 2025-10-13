#!/usr/bin/env python3
"""
AI Guard Agent - Complete Integration
======================================
Combines Milestones 1, 2, and 3:
- Voice activation and speaker verification (Milestone 1)
- Face recognition and trusted user enrollment (Milestone 2)
- Escalating dialogue with intruders (Milestone 3)

Requirements:
pip install face_recognition opencv-python numpy pillow speechrecognition pyaudio
pip install gtts pygame google-generativeai speechbrain torch torchaudio

Usage:
1. Enroll trusted users (faces)
2. Optional: Enroll owner voice for speaker verification
3. Activate guard mode via voice command
4. System monitors for intruders and responds accordingly
5. Deactivate via voice command (with speaker verification if enabled)
"""

import speech_recognition as sr
import face_recognition
import cv2
import numpy as np
import pickle
import os
import time
from datetime import datetime
from gtts import gTTS
import pygame
import tempfile
import warnings
import threading
warnings.filterwarnings('ignore')

# Optional: Speaker verification
try:
    from speechbrain.pretrained import EncoderClassifier
    import torch
    SPEAKER_VERIFY_AVAILABLE = True
except ImportError:
    SPEAKER_VERIFY_AVAILABLE = False

# Optional: LLM for dialogue
try:
    import google.generativeai as genai
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

GEMINI_API_KEY = "AIzaSyCUJRy1UMe5ADZumO6RdFS3u7cFZkmNqII"  # Replace with your key
FACE_DATABASE_FILE = "trusted_users.pkl"
SPEAKER_MODEL_FILE = "speaker_model.pkl"

# ============================================================================
# COMPONENT 1: TEXT-TO-SPEECH ENGINE
# ============================================================================

class TTSEngine:
    """Text-to-Speech using gTTS and pygame"""
    
    def __init__(self):
        pygame.mixer.init()
        self.temp_dir = tempfile.gettempdir()
        self.audio_counter = 0
    
    def clean_text_for_speech(self, text):
        """Remove markdown and special characters"""
        import re
        text = text.replace('*', '').replace('_', '')
        text = re.sub(r'\.{2,}', '.', text)
        text = ' '.join(text.split())
        return text.strip()
    
    def speak(self, text, lang='en'):
        """Convert text to speech and play it"""
        def _speak_thread():
            try:
                clean_text = self.clean_text_for_speech(text)
                print(f"🔊 Speaking: '{clean_text}'")
                
                self.audio_counter += 1
                audio_file = os.path.join(self.temp_dir, f'guard_speech_{self.audio_counter}.mp3')
                
                tts = gTTS(text=clean_text, lang=lang, slow=False)
                tts.save(audio_file)
                
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()
                
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                
                pygame.mixer.music.unload()
                time.sleep(0.1)
                
                try:
                    if os.path.exists(audio_file):
                        os.remove(audio_file)
                except:
                    pass
            except Exception as e:
                print(f"❌ TTS Error: {e}")
        
        thread = threading.Thread(target=_speak_thread, daemon=True)
        thread.start()
    
    def cleanup(self):
        """Cleanup pygame mixer"""
        try:
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
            pygame.mixer.quit()
        except:
            pass

# ============================================================================
# COMPONENT 2: SPEAKER VERIFICATION (BONUS)
# ============================================================================

class SpeakerVerifier:
    """Speaker verification using SpeechBrain"""
    
    def __init__(self, model_path=SPEAKER_MODEL_FILE):
        self.model_path = model_path
        self.classifier = None
        self.enrolled_embedding = None
        self.enrolled = False
        self.sample_rate = 16000
        
        if SPEAKER_VERIFY_AVAILABLE:
            print("Loading SpeechBrain model...")
            from speechbrain.utils.fetching import LocalStrategy
            self.classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="tmp_model",
                local_strategy=LocalStrategy.COPY
            )
            print("✓ Model loaded")
        
    def extract_embedding(self, audio_data):
        """Extract voice embedding"""
        try:
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            audio_tensor = torch.tensor(audio_data).unsqueeze(0)
            
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
        
        self.enrolled_embedding = np.mean(embeddings, axis=0)
        self.enrolled = True
        self.save_model()
        
        print("✓ Enrolled!")
        return True
    
    def verify_speaker(self, audio_data, threshold=0.75):
        """Verify speaker"""
        if not self.enrolled or self.enrolled_embedding is None:
            return True, 0.0
        
        test_embedding = self.extract_embedding(audio_data)
        if test_embedding is None:
            return False, 0.0
        
        enrolled_tensor = torch.tensor(self.enrolled_embedding).unsqueeze(0)
        test_tensor = torch.tensor(test_embedding).unsqueeze(0)
        
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
            except Exception as e:
                print(f"❌ Load error: {e}")
        return False
    
    def is_enrolled(self):
        return self.enrolled

# ============================================================================
# COMPONENT 3: FACE RECOGNITION SYSTEM
# ============================================================================

class FaceRecognitionSystem:
    """Face recognition for trusted user verification"""
    
    def __init__(self, database_file=FACE_DATABASE_FILE):
        self.database_file = database_file
        self.trusted_users = self.load_database()
        self.tolerance = 0.55
        self.face_detection_model = "hog"
        self.num_jitters = 1
        
    def load_database(self):
        """Load trusted users database"""
        if os.path.exists(self.database_file):
            try:
                with open(self.database_file, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        return {}
    
    def save_database(self):
        """Save trusted users database"""
        with open(self.database_file, 'wb') as f:
            pickle.dump(self.trusted_users, f)
        print(f"✅ Database saved ({len(self.trusted_users)} users)")
    
    def enroll_user(self, name, photo_paths):
        """Enroll a trusted user from 2 reference photos"""
        if isinstance(photo_paths, str):
            photo_paths = [photo_paths]
        
        if len(photo_paths) != 2:
            print(f"Error: Exactly 2 photos required, got {len(photo_paths)}")
            return False
        
        all_encodings = []
        valid_photos = []
        
        for photo_path in photo_paths:
            if not os.path.exists(photo_path):
                print(f"Error: Photo file '{photo_path}' not found")
                return False
            
            try:
                image = face_recognition.load_image_file(photo_path)
                face_locations = face_recognition.face_locations(image, model=self.face_detection_model)
                
                if len(face_locations) == 0:
                    print(f"Error: No faces detected in '{photo_path}'")
                    return False
                
                if len(face_locations) > 1:
                    face_locations = [max(face_locations, key=lambda loc: (loc[2]-loc[0])*(loc[1]-loc[3]))]
                
                face_encodings = face_recognition.face_encodings(image, face_locations, num_jitters=10)
                
                if len(face_encodings) == 0:
                    print(f"Error: Could not generate face encoding for '{photo_path}'")
                    return False
                
                all_encodings.append(face_encodings[0])
                valid_photos.append(photo_path)
                print(f"  ✓ Processed '{photo_path}'")
                
            except Exception as e:
                print(f"Error processing '{photo_path}': {str(e)}")
                return False
        
        self.trusted_users[name] = {
            'encodings': all_encodings,
            'photo_paths': valid_photos,
            'enrollment_date': datetime.now().isoformat(),
            'num_photos': 2
        }
        
        print(f"✅ Successfully enrolled '{name}' with 2 photos")
        return True
    
    def verify_face_from_frame(self, frame):
        """Verify a face from a video frame"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model=self.face_detection_model)
            
            if len(face_locations) == 0:
                return False, "No face detected", 0.0, None
            
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, num_jitters=1)
            
            if len(face_encodings) == 0:
                return False, "Could not encode face", 0.0, None
            
            for unknown_encoding in face_encodings:
                best_match_name = None
                best_distance = float('inf')
                
                for name, user_data in self.trusted_users.items():
                    all_user_encodings = user_data['encodings']
                    distances = face_recognition.face_distance(all_user_encodings, unknown_encoding)
                    min_distance = np.min(distances)
                    
                    if min_distance < self.tolerance and min_distance < best_distance:
                        best_distance = min_distance
                        best_match_name = name
                
                if best_match_name:
                    confidence = max(0, min(100, (1 - best_distance / self.tolerance) * 100))
                    return True, best_match_name, confidence, face_locations[0]
            
            return False, "Unknown user", 0.0, face_locations[0]
            
        except Exception as e:
            print(f"Error verifying frame: {str(e)}")
            return False, "Error", 0.0, None
    
    def list_trusted_users(self):
        """Display enrolled trusted users"""
        if not self.trusted_users:
            print("No trusted users enrolled.")
            return
        
        print(f"\n📋 Trusted Users ({len(self.trusted_users)} enrolled):")
        print("-" * 50)
        
        for name, data in self.trusted_users.items():
            print(f"• {name}")
            print(f"  Photos: {data['num_photos']}")
            for i, path in enumerate(data['photo_paths'], 1):
                print(f"    {i}. {path}")
            print(f"  Enrolled: {data['enrollment_date'][:10]}")
            print()

# ============================================================================
# COMPONENT 4: DIALOGUE MANAGER WITH ESCALATION
# ============================================================================

class DialogueManager:
    """Manages escalating conversation with intruders"""
    
    def __init__(self, api_key=None, use_tts=True, language='en'):
        self.escalation_level = 0
        self.interaction_count = 0
        self.use_tts = use_tts
        self.language = language
        self.tts = TTSEngine() if use_tts else None
        self.llm_available = LLM_AVAILABLE and api_key
        self.max_interactions = 5
        
        if self.llm_available:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
            print("✓ LLM initialized (Gemini)")
        else:
            self.model = None
            print("⚠️  Using fallback responses (no LLM)")
        
        self.escalation_thresholds = {
            1: 0,
            2: 2,
            3: 3,
            4: 4
        }
        self.max_interactions = 5
    
    def get_escalation_level(self):
        """Determine current escalation level"""
        for level in sorted(self.escalation_thresholds.keys(), reverse=True):
            if self.interaction_count >= self.escalation_thresholds[level]:
                return level
        return 1
    
    def generate_response_with_llm(self, level):
        """Generate response using LLM"""
        
        # English prompts
        prompts_en = {
            1: "You're an AI guard who watches too much TV. Someone unknown entered. Be friendly but curious like 'Hey, do I know you from somewhere?' Make it casual and slightly confused. Under 15 words.",
            
            2: "You're an AI guard channeling The Office awkwardness. They're still here. Use Michael Scott energy - try to be assertive but it comes out funny. Like 'Okay buddy, time to skedaddle.' Under 15 words.",
            
            3: "You're an AI guard doing your best Brooklyn 99 Jake Peralta impression. Final warning but make it snappy. Mix humor with 'I'm actually calling the cops now.' Sound cool under pressure. Under 15 words.",
            
            4: "You're an AI guard, Joey from Friends energy - simple and direct. 'Cops. Called. You. Screwed.' Short words, dramatic pauses. Under 10 words, punch each word."
        }
        
        # Hindi prompts (TMKOC style)
        prompts_hi = {
            1: "Aap ek friendly AI guard ho jo TMKOC dekhta hai. Koi anjaana aadmi aaya. Jethalal ki tarah curious but polite pucho 'Aap kaun hain bhai?' Natural aur friendly raho. 15 words se kam.",
            
            2: "Aap AI guard ho, ab thoda serious. Taarak Mehta style mein - respectful but firm. Bolo 'Arre bhai please jaiye yahan se' with slight frustration. Natural Hindi, 15 words se kam.",
            
            3: "Aap AI guard ho doing final warning. Sodhi ya Popatlal jaisa assertive bano. Police bulane wali baat karo but keep it desi. 'Abhi police bulaunga!' Natural tone. 15 words se kam.",
            
            4: "Aap AI guard ho. Simple aur seedhi baat - 'Police aayi. Camera chalu hai.' Bhide ki tarah strict. Chhote words. 10 words se kam."
        }
        
        # Gujarati prompts (witty and simple)
        prompts_gu = {
            1: "Tame ek friendly AI guard cho. Koi anjaan manush aavyu. Simple ane witty pucho 'Tamne hu olkhto? Tame kone cho?' Natural Gujarati, 15 words thi nani.",
            
            2: "Tame AI guard cho, have chintit. Thodu hasya par serious panu jovay. Kaho 'Bhai, have jaav please. Aa nakamu na karo.' Simple shabdo. 15 words thi nani.",
            
            3: "Tame AI guard cho doing chhelli warning. Witty par firm raho. Police ni vat karo - 'Have police ne phone karis. Seriously.' Natural tone. 15 words thi nani.",
            
            4: "Tame AI guard cho. Sidhi vat - 'Police aave che. Camera chalu che.' Simple ane powerful. 10 words thi nani."
        }
        
        # Select prompts based on language
        if self.language == 'hi':
            prompts = prompts_hi
        elif self.language == 'gu':
            prompts = prompts_gu
        else:
            prompts = prompts_en
        
        try:
            response = self.model.generate_content(prompts[level])
            return response.text.strip()
        except Exception as e:
            print(f"⚠️ LLM error: {e}, using fallback")
            return self.get_fallback_response(level)
    
    def get_fallback_response(self, level):
        """Fallback responses - natural speech"""
        
        # English responses
        responses_en = {
            1: [
                "Hey! New face alert. Got a name?",
                "Hi there, stranger danger. Who are you?",
                "Um, do we have an appointment I forgot?",
            ],
            2: [
                "Okay, this is getting weird. Time to go.",
                "Not to be rude, but... leave. Please?",
                "So, you're still here. That's... not ideal.",
            ],
            3: [
                "Alright, fun's over. Cops get called in 3... 2...",
                "Last chance, my friend. Police on speed dial.",
                "You really want me to call 911? Because I will.",
            ],
            4: [
                "Cops called. Smile for the camera.",
                "Police incoming. Hope you like handcuffs.",
                "911 dialed. You played yourself.",
            ]
        }
        
        # Hindi responses (TMKOC style)
        responses_hi = {
            1: [
                "Arre bhai, aap kaun ho? Naam batao.",
                "Namaskar! Pehli baar mil rahe hain. Aap kaun?",
                "Excuse me, aapko yahan kis ne bheja?",
            ],
            2: [
                "Arre yaar, ab jaao please. Request hai.",
                "Bhai sahab, yahan se chaliye. Serious ho gaya.",
                "Arre please, ab to chale jaao.",
            ],
            3: [
                "Bas! Abhi police ko phone lagta hun. Final warning!",
                "Chhodo mazaak, police bulani padegi ab.",
                "Ek second. Police ka number mil gaya.",
            ],
            4: [
                "Police aayi. Camera chalu hai.",
                "Police aane wali hai. Recording chal rahi.",
                "Galti kar di tumne. Police informed.",
            ]
        }
        
        # Gujarati responses (simple and witty)
        responses_gu = {
            1: [
                "Arre, tame kone cho? Naam kaho.",
                "Hello! Paheli vaar joi raheya. Tame kon?",
                "Maaf karo, tame aahe shu kaam che?",
            ],
            2: [
                "Bhai, have jaav please. Request che.",
                "Serious thay gayu. Have tarat jaav.",
                "Arre yaar, have chale jaav ne.",
            ],
            3: [
                "Bas! Have police ne phone karish. Last chance!",
                "Mazaak nathi. Police aavshe have.",
                "Ek second. Police nu number laine.",
            ],
            4: [
                "Police aave che. Camera chalu che.",
                "Police coming. Recording thay che.",
                "Bhul kari. Police ne khabar che.",
            ]
        }
        
        # Select responses based on language
        if self.language == 'hi':
            responses = responses_hi
        elif self.language == 'gu':
            responses = responses_gu
        else:
            responses = responses_en
        
        import random
        return random.choice(responses[level])
    
    def handle_intruder(self):
        """Handle intruder with escalating dialogue"""
        self.interaction_count += 1
        # ADD THIS CHECK:
        if self.interaction_count > self.max_interactions:
            print("\n" + "="*60)
            print("🚨 MAXIMUM ESCALATION REACHED - SYSTEM LOCKED")
            print(f"Interaction limit ({self.max_interactions}) exceeded")
            print("="*60)
            return 4, "Maximum security protocol activated. Authorities notified."
        
        level = self.get_escalation_level()
        
        print("\n" + "="*60)
        print(f"🚨 INTRUDER DETECTED - ESCALATION LEVEL {level}")
        print(f"Interaction #{self.interaction_count}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        if self.llm_available:
            response = self.generate_response_with_llm(level)
        else:
            response = self.get_fallback_response(level)
        
        print(f"\n💬 Guard: {response}\n")
        
        if self.use_tts and self.tts:
            self.tts.speak(response, lang=self.language)
        
        if level == 3:
            print("⚠️  WARNING: Approaching maximum escalation")
        elif level == 4:
            print("🚨 ALERT: Maximum escalation reached - authorities notified")
        
        return level, response
    
    def handle_trusted_user(self, name="Owner"):
        """Handle recognized trusted user"""
        
        # Language-specific greetings
        greetings_en = [
            f"Welcome back, {name}!",
            f"Hello {name}, all is well here.",
            f"Good to see you, {name}! Everything's secure.",
        ]
        
        greetings_hi = [
            f"Swagat hai {name}! Sab theek hai.",
            f"Namaste {name}, sab surakshit hai.",
            f"Aao {name}! Yahan sab accha hai.",
        ]
        
        greetings_gu = [
            f"Aavo {name}! Badhuj saru che.",
            f"Namaste {name}, aahe badhuj surakshit che.",
            f"Khush aavya {name}! Sab saras che.",
        ]
        
        # Select greetings based on language
        if self.language == 'hi':
            greetings = greetings_hi
        elif self.language == 'gu':
            greetings = greetings_gu
        else:
            greetings = greetings_en
        
        import random
        response = random.choice(greetings)
        
        print("\n" + "="*60)
        print(f"✅ TRUSTED USER RECOGNIZED: {name}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        print(f"\n💬 Guard: {response}\n")
        
        if self.use_tts and self.tts:
            self.tts.speak(response, lang=self.language)
        
        self.interaction_count = 0
        return response
    
    def reset_escalation(self):
        """Reset escalation level"""
        self.interaction_count = 0
        self.escalation_level = 0
        print("✓ Escalation reset")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.tts:
            self.tts.cleanup()

# ============================================================================
# COMPONENT 5: INTEGRATED AI GUARD AGENT
# ============================================================================

class IntegratedGuardAgent:
    """Complete AI Guard Agent with all features integrated"""
    
    def __init__(self, use_speaker_verification=False, use_llm=True, language='en'):
        # Core components
        self.face_system = FaceRecognitionSystem()
        self.dialogue_manager = DialogueManager(
            api_key=GEMINI_API_KEY if use_llm else None,
            use_tts=True,
            language=language
        )
        
        # Voice components
        self.recognizer = sr.Recognizer()
        self.microphone = None
        
        # Speaker verification (optional)
        self.use_speaker_verification = use_speaker_verification and SPEAKER_VERIFY_AVAILABLE
        self.speaker_verifier = None
        if self.use_speaker_verification:
            self.speaker_verifier = SpeakerVerifier()
            self.speaker_verifier.load_model()
        
        # State
        self.guard_mode = False
        self.webcam = None
        self.running = False
        
        # Voice commands
        self.activation_phrases = ["guard my room", "start guard", "activate guard"]
        self.deactivation_phrases = ["stop guard", "deactivate guard", "end guard"]
        
        # Monitoring parameters
        self.last_face_check = 0
        self.face_check_interval = 5  # Check every 5 seconds
        self.last_intruder_response = 0
        self.intruder_response_interval = 10  # Respond every 10 seconds
        self.last_trusted_user = None # Track last trusted user
    
    def initialize_audio(self):
        """Initialize audio system"""
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
            self.microphone = None  # Ensure it stays None if failed
            return False
    
    def record_audio_sample(self, duration=3):
        """Record audio sample for speaker verification"""
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
    
    def listen_for_command(self, timeout=5):
        """Listen for speech command"""
        if self.microphone is None:
            if not self.initialize_audio():
                return None, None
        
        try:
            with self.microphone as source:
                print("🎤 Listening...")
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=5)
            
            print("⚙️  Processing...")
            text = self.recognizer.recognize_google(audio).lower()
            print(f"✓ Heard: '{text}'")
            
            # Get audio data for verification if needed
            audio_data = np.frombuffer(audio.get_wav_data(), dtype=np.int16)
            audio_data = audio_data.astype(np.float32) / 32768.0
            
            return text, audio_data
        except sr.WaitTimeoutError:
            print("⏱️  Timeout")
        except sr.UnknownValueError:
            print("❌ Could not understand")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        return None, None
    
    def check_activation_command(self, text):
        """Check for activation command"""
        if text is None:
            return False
        return any(phrase in text for phrase in self.activation_phrases)
    
    def check_deactivation_command(self, text):
        """Check for deactivation command"""
        if text is None:
            return False
        return any(phrase in text for phrase in self.deactivation_phrases)
    
    def init_webcam(self):
        """Initialize webcam"""
        try:
            if self.webcam is not None:
                self.webcam.release()  # Release first
                time.sleep(0.5)
            
            self.webcam = cv2.VideoCapture(0)
            
            if not self.webcam.isOpened():
                print("✗ Webcam failed - trying different index...")
                self.webcam = cv2.VideoCapture(1)  # Try index 1
            
            if self.webcam.isOpened():
                # Test read
                ret, frame = self.webcam.read()
                if ret:
                    print("✓ Webcam ready")
                    return True
                else:
                    print("✗ Webcam opened but can't read frames")
                    return False
            else:
                print("✗ Can't open webcam - check permissions/other apps")
                return False
        except Exception as e:
            print(f"✗ Webcam error: {e}")
            return False
    
    def release_webcam(self):
        """Release webcam"""
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
                pass
    
    def activate_guard_mode(self):
        """Activate guard mode"""
        self.guard_mode = True
        self.last_trusted_user = None
        print("\n" + "="*50)
        print("🛡️  GUARD MODE ACTIVATED 🛡️")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*50 + "\n")
        self.init_webcam()
        self.dialogue_manager.reset_escalation()
    
    def deactivate_guard_mode(self):
        """Deactivate guard mode"""
        self.guard_mode = False
        print("\n" + "="*50)
        print("⏹️  GUARD MODE DEACTIVATED")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*50 + "\n")
        self.release_webcam()
    
    def capture_intruder_photo(self, frame, intruder_count):
        """Capture and save intruder photo"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"intruder_{timestamp}_{intruder_count}.jpg"
            
            # Create intruders folder if doesn't exist
            if not os.path.exists("intruders"):
                os.makedirs("intruders")
            
            filepath = os.path.join("intruders", filename)
            cv2.imwrite(filepath, frame)
            print(f"📸 Intruder photo saved: {filepath}")
            return filepath
        except Exception as e:
            print(f"❌ Photo capture error: {e}")
            return None
        
    def monitor_room(self):
        """Main monitoring loop"""
        print("\n🔍 Monitoring active... Press 'q' in video window to stop manual monitoring")

        cv2.namedWindow('AI Guard - Room Monitoring', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('AI Guard - Room Monitoring', 640, 480)
        cv2.waitKey(1)  # Force window creation
        
        while self.guard_mode:
            ret, frame = self.webcam.read()
            
            if not ret:
                print("❌ Failed to read frame")
                break
            
            current_time = time.time()
            
            # Check for faces periodically
            if current_time - self.last_face_check >= self.face_check_interval:
                self.last_face_check = current_time
                
                is_trusted, name, confidence, face_location = self.face_system.verify_face_from_frame(frame)
                
                if name != "No face detected":
                    if is_trusted:
                        # Trusted user detected
                        if not hasattr(self, 'last_trusted_user') or self.last_trusted_user != name:
                            self.dialogue_manager.handle_trusted_user(name)
                            self.last_trusted_user = name
                            self.dialogue_manager.reset_escalation()  # Reset intruder count
                        
                        cv2.rectangle(frame, (face_location[3], face_location[0]), 
                                    (face_location[1], face_location[2]), (0, 255, 0), 2)
                        cv2.putText(frame, f"TRUSTED: {name}", (face_location[3], face_location[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        # Intruder detected
                        if current_time - self.last_intruder_response >= self.intruder_response_interval:
                            self.last_intruder_response = current_time
                            self.last_trusted_user = None
                            photo_path = self.capture_intruder_photo(frame, self.dialogue_manager.interaction_count + 1)
                            self.dialogue_manager.handle_intruder()
                        
                        cv2.rectangle(frame, (face_location[3], face_location[0]), 
                                    (face_location[1], face_location[2]), (0, 0, 255), 2)
                        cv2.putText(frame, "INTRUDER!", (face_location[3], face_location[0] - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Display frame
            cv2.putText(frame, "GUARD MODE: ACTIVE", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cv2.putText(frame, timestamp, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('AI Guard - Room Monitoring', frame)
            if cv2.getWindowProperty('AI Guard - Room Monitoring', cv2.WND_PROP_VISIBLE) < 1:
                break
            
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
    
    def listen_for_activation(self):
        """Listen for activation command"""
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
                command, _ = self.listen_for_command(timeout=5)
                
                if self.check_activation_command(command):
                    self.activate_guard_mode()
                    time.sleep(1)
                    return True
                elif command:
                    print("Not an activation command. Try again...\n")
            
            print("\nMax attempts reached.\n")
        except KeyboardInterrupt:
            print("\n\n✓ Cancelled.\n")
        
        return False
    
    def listen_for_deactivation(self):
        """Listen for deactivation command"""
        print("\n" + "="*50)
        print("LISTENING FOR DEACTIVATION")
        print("="*50)
        print("\nSay one of these:")
        for cmd in self.deactivation_phrases:
            print(f"  - '{cmd}'")
        
        if self.use_speaker_verification and self.speaker_verifier.is_enrolled():
            print("\n⚠️  Speaker verification is ACTIVE")
            print("   Only enrolled owner can deactivate!")
        
        print("\nPress Ctrl+C to cancel")
        print("="*50 + "\n")
        
        try:
            for attempt in range(1, 4):
                print(f"Attempt {attempt}/3")
                
                command, audio_data = self.listen_for_command(timeout=5)
                
                if self.check_deactivation_command(command):
                    # Verify speaker if enabled
                    if self.use_speaker_verification and self.speaker_verifier.is_enrolled():
                        if audio_data is not None:
                            print("🔐 Verifying speaker identity...")
                            is_owner, similarity = self.speaker_verifier.verify_speaker(audio_data)
                            
                            print(f"   Similarity score: {similarity:.3f} (threshold: 0.75)")
                            
                            if is_owner:
                                print("✅ Speaker verified as OWNER")
                                self.deactivate_guard_mode()
                                time.sleep(1)
                                return True
                            else:
                                print("❌ Speaker NOT recognized as owner!")
                                print("   Deactivation DENIED for security.\n")
                                continue
                    else:
                        # No verification needed
                        self.deactivate_guard_mode()
                        time.sleep(1)
                        return True
                elif command:
                    print("Not a deactivation command. Try again...\n")
            
            print("\nMax attempts reached.\n")
        except KeyboardInterrupt:
            print("\n\n✓ Cancelled.\n")
        
        return False
    
    def enroll_owner_voice(self, num_samples=5):
        """Enroll owner voice for speaker verification"""
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
            print("\nâŒ Failed - not enough samples")
        
        return False
    
    def show_menu(self):
        """Display main menu"""
        print("\n" + "="*60)
        print("AI GUARD AGENT - INTEGRATED SYSTEM")
        print("="*60)
        
        if self.guard_mode:
            print("Status: 🛡️  GUARD MODE ACTIVE")
        else:
            print("Status: ⏸️  Guard Mode Inactive")
        
        print(f"Trusted Users: {len(self.face_system.trusted_users)}")
        
        if self.use_speaker_verification:
            if self.speaker_verifier.is_enrolled():
                print("Speaker Verification: ✅ Enrolled")
            else:
                print("Speaker Verification: ⚠️  Not Enrolled")
        
        print("\n📋 SETUP OPTIONS:")
        print("1. Enroll trusted user (face)")
        print("2. List trusted users")
        print("3. Remove trusted user")
        
        if self.use_speaker_verification:
            print("4. Enroll owner voice (BONUS)")
        
        print("\n🛡️  GUARD MODE:")
        print("5. Listen for voice activation")
        print("6. Listen for voice deactivation")
        print("7. Manually activate guard")
        print("8. Manually deactivate guard")
        
        if self.guard_mode:
            print("9. Start monitoring (with video)")
        
        print("\n10. Exit")
        print("="*60)
    
    def cleanup(self):
        """Cleanup all resources"""
        print("\nCleaning up resources...")
        self.release_webcam()
        self.dialogue_manager.cleanup()
        print("✓ Cleanup complete!")
    
    def run(self):
        """Main run loop"""
        self.running = True
        
        print("\n" + "="*60)
        print("AI GUARD AGENT - COMPLETE INTEGRATION")
        print("="*60)
        print("Milestones 1, 2, and 3 integrated!")
        print("Welcome to the complete AI Guard system.\n")
        
        try:
            while self.running:
                self.show_menu()
                choice = input("\nEnter your choice: ").strip()
                
                if choice == "1":
                    # Enroll trusted user (face)
                    name = input("Enter name: ").strip()
                    if name:
                        print(f"\nEnrolling '{name}' (2 photos required)")
                        print("Tip: Use photos with different lighting/angles")
                        
                        photo_paths = []
                        for i in range(2):
                            photo_path = input(f"  Photo {i+1} path: ").strip()
                            photo_paths.append(photo_path)
                        
                        if self.face_system.enroll_user(name, photo_paths):
                            save = input("Save now? (y/n): ").lower().startswith('y')
                            if save:
                                self.face_system.save_database()
                
                elif choice == "2":
                    # List trusted users
                    self.face_system.list_trusted_users()
                
                elif choice == "3":
                    # Remove trusted user
                    self.face_system.list_trusted_users()
                    name = input("Enter name to remove: ").strip()
                    if name and name in self.face_system.trusted_users:
                        del self.face_system.trusted_users[name]
                        print(f"✅ Removed '{name}'")
                        save = input("Save now? (y/n): ").lower().startswith('y')
                        if save:
                            self.face_system.save_database()
                    else:
                        print("❌ User not found")
                
                elif choice == "4" and self.use_speaker_verification:
                    # Enroll owner voice
                    self.enroll_owner_voice()
                
                elif choice == "5":
                    # Listen for voice activation
                    if not self.guard_mode:
                        self.listen_for_activation()
                    else:
                        print("\n⚠️  Guard mode already active!\n")
                        time.sleep(1)
                
                elif choice == "6":
                    # Listen for voice deactivation
                    if self.guard_mode:
                        self.listen_for_deactivation()
                    else:
                        print("\n⚠️  Guard mode not active!\n")
                        time.sleep(1)
                
                elif choice == "7":
                    # Manually activate guard
                    if not self.guard_mode:
                        self.activate_guard_mode()
                        start = input("Start monitoring now? (y/n): ").strip().lower()
                        if start == 'y':
                            self.monitor_room()
                    else:
                        print("\n⚠️  Already active!\n")
                    time.sleep(1)
                
                elif choice == "8":
                    # Manually deactivate guard
                    if self.guard_mode:
                        self.deactivate_guard_mode()
                    else:
                        print("\n⚠️  Already inactive!\n")
                    time.sleep(1)
                
                elif choice == "9" and self.guard_mode:
                    # Start monitoring
                    self.monitor_room()
                
                elif choice == "10":
                    # Exit
                    print("\nExiting...")
                    self.running = False
                
                else:
                    print("\n❌ Invalid choice.\n")
                    time.sleep(1)
        
        except KeyboardInterrupt:
            print("\n\nShutting down...")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
            print("Goodbye!")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("AI GUARD AGENT - COMPLETE INTEGRATION")
    print("="*60)

    # ADD LANGUAGE SELECTION:
    print("\n🗣️ Select language:")
    print("1. English (en)")
    print("2. Hindi (hi)")
    print("3. Gujarati (gu)")
    
    lang_choice = input("Enter choice (1-3, default=1): ").strip()
    lang_map = {'1': 'en', '2': 'hi', '3': 'gu'}
    selected_lang = lang_map.get(lang_choice, 'en')
    print(f"✓ Language: {selected_lang}\n")
    
    print("\n🎯 Choose mode:")
    print("1. Basic mode (Voice + Face + Dialogue)")
    print("2. Enhanced mode (+ Speaker Verification BONUS)")
    print()
    
    mode = input("Enter choice (1-2): ").strip()
    
    use_speaker_verify = False
    
    if mode == "2":
        if not SPEAKER_VERIFY_AVAILABLE:
            print("\n❌ SpeechBrain not available!")
            print("   Install: pip install speechbrain torch torchaudio\n")
            time.sleep(2)
        else:
            print("\n✓ Speaker Verification enabled!\n")
            use_speaker_verify = True
    else:
        print("\n✓ Basic mode\n")
    
    # Check if LLM is available
    if not LLM_AVAILABLE:
        print("⚠️  Google Generative AI not available - using fallback responses")
        print("   Install: pip install google-generativeai\n")
        time.sleep(2)
    
    # Create and run agent
    time.sleep(1)
    agent = IntegratedGuardAgent(
        use_speaker_verification=use_speaker_verify,
        use_llm=LLM_AVAILABLE,
        language=selected_lang
    )
    agent.run()
