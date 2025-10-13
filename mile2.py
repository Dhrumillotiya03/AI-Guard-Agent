#!/usr/bin/env python3
"""
Face Recognition and Trusted User Enrollment System

This script demonstrates:
1. Face detection and recognition using face_recognition library
2. Trusted user enrollment from reference photos
3. Real-time verification against enrolled users
4. Embedding comparison for identity verification

Requirements:
pip install face_recognition opencv-python numpy pillow

Usage:
1. Run the script
2. Choose option 1 to enroll trusted users (provide photo paths)
3. Choose option 2 to verify faces (from webcam or image file)
4. Choose option 3 to list enrolled users
"""

import face_recognition
import cv2
import numpy as np
import pickle
import os
from PIL import Image
import json
from datetime import datetime

class TrustedUserSystem:
    def __init__(self, database_file="trusted_users.pkl"):
        self.database_file = database_file
        self.trusted_users = self.load_database()
        self.tolerance = 0.55  # Balanced for multiple photos per user
        self.face_detection_model = "hog"  # Use HOG for faster detection
        
    def load_database(self):
        """Load trusted users database from file"""
        if os.path.exists(self.database_file):
            try:
                with open(self.database_file, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        return {}
    
    def save_database(self):
        """Save trusted users database to file"""
        with open(self.database_file, 'wb') as f:
            pickle.dump(self.trusted_users, f)
        print(f"✅ Database saved ({len(self.trusted_users)} users)")
    
    def enroll_user(self, name, photo_paths):
        """
        Enroll a trusted user from exactly 2 reference photos
        
        Args:
            name (str): Name of the trusted user
            photo_paths (list): List of exactly 2 paths to reference photos
        
        Returns:
            bool: True if enrollment successful, False otherwise
        """
        # Convert single path to list
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
                # Load the image
                image = face_recognition.load_image_file(photo_path)
                
                # Find face locations using detection model
                face_locations = face_recognition.face_locations(image, model=self.face_detection_model)
                
                if len(face_locations) == 0:
                    print(f"Error: No faces detected in '{photo_path}'")
                    return False
                
                if len(face_locations) > 1:
                    # Select the largest face (most prominent)
                    face_locations = [max(face_locations, key=lambda loc: (loc[2]-loc[0])*(loc[1]-loc[3]))]
                
                # Get face encoding with higher number of jitters for better accuracy
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
        
        # Store the user data with 2 encodings
        self.trusted_users[name] = {
            'encodings': all_encodings,  # List of 2 encodings
            'photo_paths': valid_photos,
            'enrollment_date': datetime.now().isoformat(),
            'num_photos': 2
        }
        
        print(f"✅ Successfully enrolled '{name}' with 2 photos")
        
        return True
    
    def verify_face_from_image(self, image_path):
        """
        Verify a face from an image file
        
        Args:
            image_path (str): Path to the image to verify
        
        Returns:
            tuple: (is_trusted, user_name, confidence)
        """
        if not os.path.exists(image_path):
            print(f"Error: Image file '{image_path}' not found")
            return False, None, 0.0
        
        try:
            # Load the image
            image = face_recognition.load_image_file(image_path)
            is_trusted, user_name, confidence = self._verify_face_from_array(image)
            
            # Output USER or INTRUDER
            if is_trusted:
                print(f"\n🟢 STATUS: USER")
                print(f"   Name: {user_name}")
                print(f"   Confidence: {confidence:.2f}%")
            else:
                print(f"\n🔴 STATUS: INTRUDER")
                print(f"   Reason: {user_name}")
            
            return is_trusted, user_name, confidence
            
        except Exception as e:
            print(f"Error verifying image: {str(e)}")
            return False, None, 0.0
    
    def _verify_face_from_array(self, image_array):
        """
        Internal method to verify face from numpy array
        Uses minimum distance across all enrolled photos for better accuracy
        
        Args:
            image_array: RGB image array
            
        Returns:
            tuple: (is_trusted, user_name, confidence)
        """
        # Find faces in the image using detection model
        face_locations = face_recognition.face_locations(image_array, model=self.face_detection_model)
        
        if len(face_locations) == 0:
            return False, "No face detected", 0.0
        
        # Get face encodings with jitters for better accuracy
        face_encodings = face_recognition.face_encodings(image_array, face_locations, num_jitters=5)
        
        if len(face_encodings) == 0:
            return False, "Could not encode face", 0.0
        
        # Compare with trusted users (using all their photos)
        for unknown_encoding in face_encodings:
            best_match_name = None
            best_distance = float('inf')
            
            for name, user_data in self.trusted_users.items():
                # Compare against all encodings for this user
                all_user_encodings = user_data['encodings']
                
                # Calculate distance to each encoding and take the minimum
                distances = face_recognition.face_distance(all_user_encodings, unknown_encoding)
                min_distance = np.min(distances)
                
                if min_distance < self.tolerance and min_distance < best_distance:
                    best_distance = min_distance
                    best_match_name = name
            
            if best_match_name:
                # Improved confidence calculation
                confidence = max(0, min(100, (1 - best_distance / self.tolerance) * 100))
                return True, best_match_name, confidence
        
        return False, "Unknown user", 0.0
    
    def verify_webcam(self):
        """
        Verify faces from webcam feed - captures one verification result
        """
        print("Opening webcam for verification...")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not access webcam")
            return
        
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        import time
        verification_done = False
        result = None
        
        print("Webcam opened, look at the camera...")
        
        try:
            start_time = time.time()
            
            while not verification_done and (time.time() - start_time) < 10:  # Max 10 seconds
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to read frame")
                    break
                
                # Try to detect and verify face
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.5, fy=0.5)
                
                is_trusted, user_name, confidence = self._verify_face_from_array(small_frame)
                
                # Show live feedback on frame
                if is_trusted and user_name != "No face detected":
                    color = (0, 255, 0)
                    text = f"USER: {user_name} ({confidence:.1f}%)"
                    bg_color = (0, 100, 0)
                    result = (True, user_name, confidence)
                    verification_done = True
                elif user_name == "No face detected":
                    color = (255, 255, 0)
                    text = "Position your face in frame"
                    bg_color = (100, 100, 0)
                else:
                    color = (0, 0, 255)
                    text = "INTRUDER: Unknown user"
                    bg_color = (0, 0, 100)
                    result = (False, user_name, confidence)
                    verification_done = True
                
                # Draw on frame
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                overlay = frame.copy()
                cv2.rectangle(overlay, (5, 5), (text_width + 20, text_height + 20), bg_color, -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                cv2.putText(frame, text, (10, text_height + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                cv2.putText(frame, text, (10, text_height + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                cv2.imshow('Face Verification', frame)
                cv2.waitKey(1)
                
                # Small delay between checks
                time.sleep(0.3)
                
        except Exception as e:
            print(f"Error during webcam verification: {str(e)}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Print final result
            print("\n" + "="*50)
            if result:
                is_trusted, user_name, confidence = result
                if is_trusted:
                    print("✅ ACCESS GRANTED")
                    print(f"   USER: {user_name}")
                    print(f"   Confidence: {confidence:.1f}%")
                else:
                    print("❌ ACCESS DENIED")
                    print(f"   INTRUDER: {user_name}")
            else:
                print("⚠️  No face detected within time limit")
            print("="*50)
    
    def list_trusted_users(self):
        """Display information about enrolled trusted users"""
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
    
    def remove_user(self, name):
        """Remove a trusted user from the database"""
        if name in self.trusted_users:
            del self.trusted_users[name]
            print(f"✅ Removed '{name}'")
            return True
        else:
            print(f"❌ User '{name}' not found")
            return False
    
    def get_embedding_info(self):
        """Display information about face embeddings"""
        print("\n🧠 Face Embedding Information:")
        print("-" * 40)
        print("• 128-dimensional vectors representing facial features")
        print("• Compared using Euclidean distance")
        print(f"• Tolerance threshold: {self.tolerance}")
        print(f"• Detection model: {self.face_detection_model}")
        print("• Multiple photos per user for robustness")
        
        if self.trusted_users:
            total_encodings = sum(user['num_photos'] for user in self.trusted_users.values())
            print(f"• Total encodings stored: {total_encodings}")
            
            sample_encoding = next(iter(self.trusted_users.values()))['encodings'][0]
            print(f"\n• Embedding statistics:")
            print(f"  Min: {np.min(sample_encoding):.4f}")
            print(f"  Max: {np.max(sample_encoding):.4f}")
            print(f"  Mean: {np.mean(sample_encoding):.4f}")


def main():
    """Main demo function"""
    system = TrustedUserSystem()
    
    while True:
        print("\n" + "="*50)
        print("🔐 Face Recognition System")
        print("="*50)
        print("1. Enroll trusted user")
        print("2. Verify face from image")
        print("3. Verify from webcam")
        print("4. List trusted users")
        print("5. Remove user")
        #print("6. Show embedding info")
        #print("7. Save database")
        #print("8. Exit")
        
        choice = input("\nSelect (1-5): ").strip()
        
        if choice == '1':
            name = input("Enter name: ").strip()
            
            if name:
                print(f"\nEnrolling '{name}' (2 photos required)")
                print("Tip: Use photos with different lighting/angles/hairstyles")
                
                photo_paths = []
                for i in range(2):
                    photo_path = input(f"  Photo {i+1} path: ").strip()
                    photo_paths.append(photo_path)
                
                if system.enroll_user(name, photo_paths):
                    save = input("Save now? (y/n): ").lower().startswith('y')
                    if save:
                        system.save_database()
        
        elif choice == '2':
            image_path = input("Enter image path: ").strip()
            if image_path:
                system.verify_face_from_image(image_path)
        
        elif choice == '3':
            
            system.verify_webcam()
        
        elif choice == '4':
            system.list_trusted_users()
        
        elif choice == '5':
            system.list_trusted_users()
            name = input("Enter name to remove: ").strip()
            if name:
                if system.remove_user(name):
                    save = input("Save now? (y/n): ").lower().startswith('y')
                    if save:
                        system.save_database()
        
        #elif choice == '6':
        #    system.get_embedding_info()
        
        #elif choice == '7':
        #    system.save_database()
        
        #elif choice == '8':
        #    print("Goodbye!")
        #    break
        
        else:
            print("Invalid option.")


if __name__ == "__main__":
    print("🔐 Face Recognition - Milestone 2")
    print("=" * 50)
    print("\n⚠️  Ensure you have permission to use photos")
    print("📦 Required: pip install face_recognition opencv-python\n")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")
    except Exception as e:
        print(f"\nError: {str(e)}")