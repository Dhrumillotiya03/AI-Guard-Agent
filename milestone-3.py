"""
Milestone 3: Escalation Dialogue System + TTS
==============================================
Integrates LLM dialogue with escalation + Text-to-Speech
Bonus: Creative personality, multi-language support

Requirements:
pip install gtts google-generativeai pygame
"""

import time
from datetime import datetime
from gtts import gTTS
import pygame
import os
import tempfile

# LLM Setup - Using Google Gemini (free)
try:
    import google.generativeai as genai
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("⚠️  Install: pip install google-generativeai")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Google Gemini API Key (free tier)
GEMINI_API_KEY = "AIzaSyCUJRy1UMe5ADZumO6RdFS3u7cFZkmNqII"

# ============================================================================
# PART 1: TEXT-TO-SPEECH ENGINE
# ============================================================================

class TTSEngine:
    """Text-to-Speech using gTTS and pygame"""
    
    def __init__(self):
        pygame.mixer.init()
        self.temp_dir = tempfile.gettempdir()
        self.audio_counter = 0  # To create unique filenames
    
    def clean_text_for_speech(self, text):
        """Remove markdown and special characters that shouldn't be spoken"""
        import re
        
        # Remove asterisks (bold/italic markdown)
        text = text.replace('*', '')
        
        # Remove underscores (markdown)
        text = text.replace('_', '')
        
        # Replace multiple dots with period
        text = re.sub(r'\.{2,}', '.', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def speak(self, text, lang='en'):
        """Convert text to speech and play it"""
        try:
            # Clean text before speaking
            clean_text = self.clean_text_for_speech(text)
            
            print(f"🔊 Speaking: '{clean_text}'")
            
            # Generate unique filename to avoid conflicts
            self.audio_counter += 1
            audio_file = os.path.join(self.temp_dir, f'guard_speech_{self.audio_counter}.mp3')
            
            # Generate speech
            tts = gTTS(text=clean_text, lang=lang, slow=False)
            tts.save(audio_file)
            
            # Play audio
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            
            # Wait for audio to finish
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            # Unload before deletion
            pygame.mixer.music.unload()
            time.sleep(0.1)  # Give system time to release file
            
            # Cleanup
            try:
                if os.path.exists(audio_file):
                    os.remove(audio_file)
            except:
                pass  # Ignore if file is still locked
                
        except Exception as e:
            print(f"❌ TTS Error: {e}")
    
    def cleanup(self):
        """Cleanup pygame mixer and temp files"""
        try:
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
            pygame.mixer.quit()
            
            # Clean up any leftover temp files
            for i in range(self.audio_counter + 1):
                temp_file = os.path.join(self.temp_dir, f'guard_speech_{i}.mp3')
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except:
                    pass
        except:
            pass


# ============================================================================
# PART 2: ESCALATION DIALOGUE MANAGER
# ============================================================================

class DialogueManager:
    """Manages escalating conversation with intruders"""
    
    def __init__(self, api_key=None, use_tts=True):
        self.escalation_level = 0
        self.interaction_count = 0
        self.use_tts = use_tts
        self.tts = TTSEngine() if use_tts else None
        self.llm_available = LLM_AVAILABLE and api_key
        
        # Language support
        self.languages = {
            'en': 'English',
            'hi': 'Hindi',
            'gu': 'Gujarati'
        }
        
        if self.llm_available:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
            print("✓ LLM initialized (Gemini 2.0 Flash)")
        else:
            self.model = None
            print("⚠️  Using fallback responses (no LLM)")
        
        # Escalation thresholds
        self.escalation_thresholds = {
            1: 0,   # Level 1: Immediate
            2: 2,   # Level 2: After 2 interactions
            3: 4,   # Level 3: After 4 interactions
            4: 6    # Level 4: After 6 interactions
        }
    
    def get_escalation_level(self):
        """Determine current escalation level"""
        for level in sorted(self.escalation_thresholds.keys(), reverse=True):
            if self.interaction_count >= self.escalation_thresholds[level]:
                return level
        return 1
    
    def generate_response_with_llm(self, level, intruder_detected=True):
        """Generate response using LLM with sitcom personality"""
        prompts = {
            1: "You are an AI security guard with personality inspired by sitcoms like Friends, The Office, Brooklyn 99, Modern Family, or Taarak Mehta Ka Ooltah Chashmah. Be witty and funny. An unknown person entered. Politely ask who they are. Add a subtle sitcom reference if natural. Keep under 25 words.",
            2: "You are an AI security guard with sitcom humor (think Michael Scott or Jake Peralta or Jethalal). The intruder hasn't left. Be firm but funny. Tell them to leave. A sitcom reference is welcome if it fits. Keep under 25 words.",
            3: "You are an AI security guard channeling your inner Dwight Schrute or Captain Holt. FINAL WARNING with sitcom flair. Mention authorities. Keep under 25 words.",
            4: "You are an AI security guard. Very serious now but with subtle sitcom reference if fitting (think serious Holt or serious Jethalal moment). State police are contacted. Keep under 25 words."
        }
        
        try:
            response = self.model.generate_content(prompts[level])
            return response.text.strip()
        except Exception as e:
            print(f"⚠️  LLM error: {e}, using fallback")
            return self.get_fallback_response(level)
    
    def get_fallback_response(self, level):
        """Fallback responses without LLM - with sitcom humor"""
        responses = {
            1: [
                "Hello! I don't recognize you. Who are you and what brings you here?",
                "Hi! Could you identify yourself? This isn't Central Perk, you need permission to be here.",
                "Umm, we've never met. I'm Michael Scott... I mean, the AI guard. Who are you?",
                "Nine-Nine! Wait, wrong catchphrase. Who are you and why are you in this room?",
                "Arey bhai, kaun ho tum? This isn't Gokuldham Society, identify yourself please!"
            ],
            2: [
                "You are not authorized to be here. Please leave immediately.",
                "I'm going to need you to leave. That's what she said... seriously, leave now.",
                "Title of your sex tape: Unauthorized Entry. Now please exit immediately.",
                "This is a restricted area. I'm putting you on the naughty list. Leave now.",
                "Tapu Sena ke saath confusion ho gaya kya? You need to leave, this instant!"
            ],
            3: [
                "FINAL WARNING. Leave now or authorities will be contacted immediately.",
                "Identity theft is not a joke! Neither is trespassing. Police in 10 seconds if you don't leave.",
                "Cool cool cool cool cool... NOT cool. Leave NOW or I'm calling the cops.",
                "That's BONE! B-O-N-E. And you're gonna be in trouble. Leave or police will arrive.",
                "Bapu ji ko bula loon? No seriously, police aa jayegi. Leave RIGHT NOW!"
            ],
            4: [
                "Police have been contacted. You are being recorded. Stay where you are.",
                "How you doin'? Not well, because the police are already on their way.",
                "Parkour! Into a police van if you don't explain yourself. Authorities notified.",
                "I am a golden god! Of security. Police have been called. Stay put.",
                "Gokuldham mein itna bada crime! Police ko phone kar diya. You're recorded."
            ]
        }
        
        import random
        return random.choice(responses[level])
    
    def generate_contextual_response(self, level, intruder_speech):
        """
        Generate response based on what intruder said (for conversational mode).
        
        Args:
            level: Current escalation level
            intruder_speech: What the intruder just said
            
        Returns:
            Contextual response string
        """
        if not self.llm_available:
            # Fallback if no LLM
            return f"I heard you, but you still need to leave. This is level {level} warning."
        
        base_context = {
            1: "You're an AI guard (level 1 - polite). The intruder just spoke.",
            2: "You're an AI guard (level 2 - firm). The intruder is making excuses.",
            3: "You're an AI guard (level 3 - stern). The intruder still won't leave.",
            4: "You're an AI guard (level 4 - serious). Police are coming."
        }
        
        prompt = base_context[level]
        prompt += f" The intruder said: '{intruder_speech}'. "
        prompt += "Respond directly to what they said while maintaining security stance. "
        prompt += "Use sitcom humor (Friends, Office, Brooklyn 99, TMKOC). Keep under 25 words."
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"⚠️  LLM error: {e}")
            return f"I understand, but you need to leave. Level {level} security protocol."
    
    def handle_intruder(self, face_info=None):
        """
        Handle intruder with escalating dialogue.
        face_info: dict with 'face_detected', 'is_trusted', 'name'
        """
        self.interaction_count += 1
        level = self.get_escalation_level()
        
        print("\n" + "="*60)
        print(f"🚨 INTRUDER DETECTED - ESCALATION LEVEL {level}")
        print(f"Interaction #{self.interaction_count}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        # Generate response
        if self.llm_available:
            response = self.generate_response_with_llm(level)
        else:
            response = self.get_fallback_response(level)
        
        print(f"\n💬 Guard: {response}\n")
        
        # Speak response
        if self.use_tts and self.tts:
            self.tts.speak(response)
        
        # Log escalation actions
        if level == 3:
            print("⚠️  WARNING: Approaching maximum escalation")
        elif level == 4:
            print("🚨 ALERT: Maximum escalation reached - authorities notified")
        
        return level, response
    
    def handle_trusted_user(self, name="Owner"):
        """Handle recognized trusted user with sitcom personality"""
        greetings = [
            f"Welcome back, {name}!",
            f"Hello {name}, all is well here.",
            f"How you doin', {name}? Everything's secure!",
            f"Oh. My. God. {name}! Welcome back!",
            f"That's what she said! I mean... welcome back, {name}!",
            f"Cool cool cool cool cool, {name} is back!",
            f"Jai Shree Krishna, {name}! Sab theek hai, welcome!"
        ]
        
        import random
        response = random.choice(greetings)
        
        print("\n" + "="*60)
        print(f"✅ TRUSTED USER RECOGNIZED: {name}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        print(f"\n💬 Guard: {response}\n")
        
        if self.use_tts and self.tts:
            self.tts.speak(response)
        
        # Reset escalation
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
# PART 3: DEMO / TEST SYSTEM
# ============================================================================

def demo_escalation_system(api_key=None):
    """Demo the escalation dialogue system"""
    
    print("\n" + "="*60)
    print("MILESTONE 3: ESCALATION DIALOGUE SYSTEM DEMO")
    print("="*60)
    print("\nThis demo simulates intruder detection with escalating responses.")
    print("In the full system, this integrates with face recognition from Milestone 2.\n")
    
    # Choose language
    print("Choose language:")
    print("1. English (with Harry Potter humor)")
    print("2. Hindi (हिंदी)")
    print("3. Gujarati (ગુજરાતી)")
    
    lang_choice = input("\nLanguage (1-3, default=1): ").strip() or "1"
    
    lang_map = {'1': 'en', '2': 'hi', '3': 'gu'}
    selected_lang = lang_map.get(lang_choice, 'en')
    
    # Initialize dialogue manager
    if selected_lang == 'en':
        dm = DialogueManager(api_key=api_key, use_tts=True)
    else:
        dm = MultiLanguageDialogue(api_key=api_key, use_tts=True, language=selected_lang)
    
    print(f"\n✓ Language set to: {dm.languages.get(selected_lang, 'English')}")
    
    print("\n" + "="*60)
    print("SCENARIO 1: Unknown Intruder (Escalation)")
    print("="*60)
    
    # Simulate multiple detections
    for i in range(5):
        input(f"\nPress Enter to trigger detection #{i+1}...")
        
        # Mock face info (intruder)
        face_info = {
            'face_detected': True,
            'is_trusted': False,
            'name': 'Unknown'
        }
        
        # Use multilingual response if not English
        if selected_lang != 'en':
            dm.interaction_count += 1
            level = dm.get_escalation_level()
            response = dm.get_multilingual_response(level)
            
            print("\n" + "="*60)
            print(f"🚨 INTRUDER DETECTED - ESCALATION LEVEL {level}")
            print(f"Interaction #{dm.interaction_count}")
            print("="*60)
            print(f"\n💬 Guard: {response}\n")
            
            dm.speak_multilingual(response)
        else:
            level, response = dm.handle_intruder(face_info)
        
        time.sleep(1)
        
        if level == 4:
            print("\n⚠️  Maximum escalation reached!")
            break
    
    print("\n" + "="*60)
    print("SCENARIO 2: Trusted User Returns")
    print("="*60)
    
    input("\nPress Enter to simulate trusted user detection...")
    
    # Mock face info (trusted)
    dm.handle_trusted_user(name="Owner")
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    
    # Cleanup
    dm.cleanup()


def test_tts():
    """Test TTS system"""
    print("\n" + "="*60)
    print("TTS SYSTEM TEST")
    print("="*60)
    
    tts = TTSEngine()
    
    test_phrases = [
        "Hello! I don't recognize you. Who are you?",
        "This is a restricted area. Please leave immediately.",
        "Final warning. Police are being contacted.",
        "Welcome back, Owner!"
    ]
    
    for i, phrase in enumerate(test_phrases, 1):
        print(f"\nTest {i}/4:")
        tts.speak(phrase)
        time.sleep(0.5)
    
    tts.cleanup()
    print("\n✓ TTS test complete")


def test_llm_responses(api_key):
    """Test LLM response generation"""
    print("\n" + "="*60)
    print("LLM RESPONSE GENERATION TEST")
    print("="*60)
    
    dm = DialogueManager(api_key=api_key, use_tts=False)
    
    if not dm.llm_available:
        print("❌ LLM not available. Showing fallback responses only.")
    
    for level in [1, 2, 3, 4]:
        print(f"\n--- Level {level} Response ---")
        if dm.llm_available:
            response = dm.generate_response_with_llm(level)
        else:
            response = dm.get_fallback_response(level)
        print(f"Guard: {response}")
        time.sleep(0.5)
    
    dm.cleanup()


# ============================================================================
# BONUS: MULTI-LANGUAGE SUPPORT
# ============================================================================

class MultiLanguageDialogue(DialogueManager):
    """Enhanced dialogue with multi-language support: English, Hindi, Gujarati"""
    
    def __init__(self, api_key=None, use_tts=True, language='en'):
        super().__init__(api_key, use_tts)
        self.language = language
        
        self.languages = {
            'en': 'English',
            'hi': 'Hindi',
            'gu': 'Gujarati'
        }
    
    def speak_multilingual(self, text):
        """Speak in selected language"""
        if self.use_tts and self.tts:
            self.tts.speak(text, lang=self.language)
    
    def get_multilingual_response(self, level):
        """Get response in selected language"""
        if self.language == 'en':
            return self.get_fallback_response(level)
        
        responses = {
            'hi': {
                1: "नमस्ते! मैं आपको नहीं पहचानता। आप कौन हैं?",
                2: "आप अधिकृत नहीं हैं। कृपया तुरंत निकल जाएं।",
                3: "अंतिम चेतावनी। अभी जाओ या पुलिस को बुलाया जाएगा।",
                4: "पुलिस को सूचित कर दिया गया है। आप रिकॉर्ड हो रहे हैं।"
            },
            'gu': {
                1: "નમસ્તે! હું તમને ઓળખતો નથી. તમે કોણ છો?",
                2: "તમને અધિકૃત નથી. કૃપા કરીને તાત્કાલિક બહાર નીકળો.",
                3: "અંતિમ ચેતવણી. હવે જાઓ નહીંતર પોલીસને બોલાવવામાં આવશે.",
                4: "પોલીસને જાણ કરી દેવામાં આવી છે. તમારું રેકોર્ડિંગ થઈ રહ્યું છે."
            }
        }
        
        if self.language in responses:
            return responses[self.language].get(level, self.get_fallback_response(level))
        
        return self.get_fallback_response(level)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("MILESTONE 3: DIALOGUE + TTS SYSTEM")
    print("="*60)
    print("\nChoose test mode:")
    print("1. Full escalation demo (with TTS)")
    print("2. Test TTS only")
    print("3. Test LLM responses")
    print("4. Skip demo")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        demo_escalation_system(api_key=GEMINI_API_KEY)
    
    elif choice == "2":
        test_tts()
    
    elif choice == "3":
        test_llm_responses(GEMINI_API_KEY)
    
    elif choice == "4":
        print("\nSkipping demo")
    
    else:
        print("\n❌ Invalid choice")
    
    print("\n✓ Milestone 3 code ready for integration!")