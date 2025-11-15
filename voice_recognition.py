import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from scipy.signal import butter, lfilter
import librosa
import json
import base64
import hashlib
from datetime import datetime, timezone
from decimal import Decimal
import io


class VoiceBiometricSystem:
    """
    Voice recognition system for student verification
    Uses voice features (MFCC) for identification
    """

    def __init__(self, sample_rate=16000, duration=3):
        """
        Initialize voice biometric system

        Args:
            sample_rate: Audio sample rate (16000 Hz is standard for speech)
            duration: Recording duration in seconds
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.is_connected = True  # Always available (uses microphone)
        self.scanner_type = "Voice Biometric Authentication"
        self.model_number = "VBA-v1.0"

        # Check microphone availability
        self._check_microphone()

    def _check_microphone(self):
        """Check if microphone is available"""
        try:
            devices = sd.query_devices()
            input_devices = [d for d in devices if d['max_input_channels'] > 0]

            if not input_devices:
                print("âš ï¸  No microphone detected!")
                self.is_connected = False
                return

            # Use default input device
            default_input = sd.query_devices(kind='input')
            print(f"âœ“ Microphone detected: {default_input['name']}")
            self.is_connected = True

        except Exception as e:
            print(f"âŒ Microphone check failed: {e}")
            self.is_connected = False

    def initialize_scanner(self):
        """Initialize voice capture system (compatibility with fingerprint API)"""
        print(f"Initializing {self.scanner_type}...")
        self._check_microphone()

        if self.is_connected:
            print(f"âœ“ Voice biometric system ready")
            print(f"  Sample rate: {self.sample_rate} Hz")
            print(f"  Recording duration: {self.duration} seconds")

        return self.is_connected

    def _record_audio(self, prompt_message=None):
        """Record audio from microphone"""

        if prompt_message:
            print(f"\n{prompt_message}")
        else:
            print("\nðŸŽ¤ Recording your voice...")

        print(f"   Please speak clearly for {self.duration} seconds")
        print(f"   Say: 'My name is [Your Name] and my registration number is [Number]'")
        print("\n   Recording starts in:")

        for i in range(3, 0, -1):
            print(f"   {i}...")
            import time
            time.sleep(1)

        print("   ðŸ”´ RECORDING NOW - Speak clearly!\n")

        try:
            # Record audio
            recording = sd.rec(
                int(self.duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32'
            )
            sd.wait()  # Wait for recording to finish

            print("   âœ“ Recording complete!\n")

            # Convert to numpy array
            audio_data = recording.flatten()

            # Check if audio was captured
            if np.max(np.abs(audio_data)) < 0.01:
                raise Exception("No audio detected - please check microphone")

            return audio_data

        except Exception as e:
            raise Exception(f"Audio recording failed: {e}")

    def _preprocess_audio(self, audio_data):
        """Preprocess audio for feature extraction"""

        # Remove DC offset
        audio_data = audio_data - np.mean(audio_data)

        # Normalize
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val

        # Apply bandpass filter (80 Hz - 8000 Hz for speech)
        try:
            from scipy.signal import butter, lfilter

            nyquist = self.sample_rate / 2.0
            low_freq = 80.0 / nyquist
            high_freq = min(8000.0, self.sample_rate * 0.45) / nyquist  # Cap at 45% of sample rate

            # Ensure frequencies are in valid range (0 < Wn < 1)
            low_freq = max(0.01, min(0.99, low_freq))
            high_freq = max(0.01, min(0.99, high_freq))

            # Ensure low < high
            if low_freq >= high_freq:
                high_freq = low_freq + 0.1

            b, a = butter(4, [low_freq, high_freq], btype='band')
            filtered_audio = lfilter(b, a, audio_data)

            return filtered_audio

        except Exception as e:
            print(f"Filter warning: {e}. Returning unfiltered audio.")
            # If filtering fails, return normalized audio
            return audio_data

    def _extract_voice_features(self, audio_data):
        """Extract voice biometric features (MFCC)"""

        # Preprocess
        processed_audio = self._preprocess_audio(audio_data)

        # Extract MFCCs (Mel-Frequency Cepstral Coefficients)
        # These capture the unique characteristics of a person's voice
        mfccs = librosa.feature.mfcc(
            y=processed_audio,
            sr=self.sample_rate,
            n_mfcc=20,  # 20 coefficients
            n_fft=512,
            hop_length=256
        )

        # Statistical features from MFCCs
        mfcc_mean = np.mean(mfccs, axis=1).tolist()
        mfcc_std = np.std(mfccs, axis=1).tolist()
        mfcc_min = np.min(mfccs, axis=1).tolist()
        mfcc_max = np.max(mfccs, axis=1).tolist()

        # Extract pitch (fundamental frequency)
        pitches, magnitudes = librosa.piptrack(
            y=processed_audio,
            sr=self.sample_rate,
            n_fft=512,
            hop_length=256
        )

        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)

        pitch_mean = float(np.mean(pitch_values)) if pitch_values else 0.0
        pitch_std = float(np.std(pitch_values)) if pitch_values else 0.0

        # Extract spectral features
        spectral_centroids = librosa.feature.spectral_centroid(
            y=processed_audio, sr=self.sample_rate
        )[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=processed_audio, sr=self.sample_rate
        )[0]

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(processed_audio)[0]

        # Energy
        energy = librosa.feature.rms(y=processed_audio)[0]

        features = {
            'mfcc_mean': mfcc_mean,
            'mfcc_std': mfcc_std,
            'mfcc_min': mfcc_min,
            'mfcc_max': mfcc_max,
            'pitch_mean': pitch_mean,
            'pitch_std': pitch_std,
            'spectral_centroid_mean': float(np.mean(spectral_centroids)),
            'spectral_centroid_std': float(np.std(spectral_centroids)),
            'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
            'spectral_rolloff_std': float(np.std(spectral_rolloff)),
            'zcr_mean': float(np.mean(zcr)),
            'zcr_std': float(np.std(zcr)),
            'energy_mean': float(np.mean(energy)),
            'energy_std': float(np.std(energy))
        }

        return features

    def _calculate_quality_score(self, audio_data, features):
        """Calculate voice sample quality (0-100)"""

        quality_score = 100.0

        # Check 1: Signal strength
        rms = np.sqrt(np.mean(audio_data ** 2))
        if rms < 0.01:
            quality_score -= 30
        elif rms < 0.05:
            quality_score -= 15

        # Check 2: Speech detection (energy variance)
        energy = librosa.feature.rms(y=audio_data)[0]
        energy_var = np.var(energy)
        if energy_var < 0.001:
            quality_score -= 20  # Too uniform, might be silence

        # Check 3: Frequency content (speech should have rich spectrum)
        spectral_centroid = features['spectral_centroid_mean']
        if spectral_centroid < 500 or spectral_centroid > 8000:
            quality_score -= 15

        # Check 4: Pitch detection (should have some pitch for speech)
        if features['pitch_mean'] < 50:
            quality_score -= 10

        return max(50.0, min(100.0, quality_score))

    def capture_fingerprint(self):
        """
        Capture voice sample (compatible with fingerprint API)
        Returns: dict with voice biometric data
        """

        if not self.is_connected:
            raise Exception("Microphone not available")

        try:
            print("=" * 60)
            print("VOICE BIOMETRIC ENROLLMENT")
            print("=" * 60)

            # Record audio
            audio_data = self._record_audio(
                "ðŸŽ¤ ENROLLMENT: Please speak clearly"
            )

            # Extract features
            print("Processing voice sample...")
            features = self._extract_voice_features(audio_data)

            # Calculate quality
            quality_score = self._calculate_quality_score(audio_data, features)

            if quality_score < 60:
                print("âš ï¸  Warning: Low quality recording")
                print("   Try again in a quieter environment")
                print("   Speak closer to the microphone")

            # Save audio as base64 (for reference/debugging)
            audio_int16 = (audio_data * 32767).astype(np.int16)
            buffer = io.BytesIO()
            wav.write(buffer, self.sample_rate, audio_int16)
            audio_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # Create template
            template_data = {
                'format': 'VOICE_BIOMETRIC_MFCC',
                'quality_score': quality_score,
                'sample_rate': self.sample_rate,
                'duration': self.duration,
                'capture_timestamp': datetime.now(timezone.utc).isoformat(),
                'features': features,
                'feature_hash': hashlib.sha256(
                    json.dumps(features, sort_keys=True).encode()
                ).hexdigest()
            }

            print(f"\nâœ“ Voice sample captured successfully")
            print(f"  Quality: {quality_score:.1f}/100")
            print(f"  Pitch: {features['pitch_mean']:.1f} Hz")

            return {
                'image_data': audio_base64,  # Audio instead of image
                'quality_score': quality_score,
                'template': json.dumps(template_data),
                'template_data': template_data
            }

        except Exception as e:
            raise Exception(f"Voice capture failed: {e}")

    def compare_fingerprints(self, template1, template2):
        """
        Compare two voice templates (compatible with fingerprint API)
        Returns: (confidence_score, is_match)
        """

        try:
            # Parse templates
            if isinstance(template1, str):
                data1 = json.loads(template1)
            else:
                data1 = template1

            if isinstance(template2, bytes):
                data2 = json.loads(template2.decode('utf-8'))
            elif isinstance(template2, str):
                data2 = json.loads(template2)
            else:
                data2 = template2

            features1 = data1.get('features', {})
            features2 = data2.get('features', {})

            if not features1 or not features2:
                return 0.0, False

            # Calculate similarity between voice features

            # 1. MFCC similarity (most important)
            mfcc_mean1 = np.array(features1['mfcc_mean'])
            mfcc_mean2 = np.array(features2['mfcc_mean'])
            mfcc_distance = np.linalg.norm(mfcc_mean1 - mfcc_mean2)
            mfcc_similarity = max(0, 100 - (mfcc_distance * 10))  # Scale to 0-100

            # 2. Pitch similarity
            pitch_diff = abs(features1['pitch_mean'] - features2['pitch_mean'])
            pitch_similarity = max(0, 100 - (pitch_diff / 2))

            # 3. Spectral similarity
            spectral_diff = abs(
                features1['spectral_centroid_mean'] -
                features2['spectral_centroid_mean']
            )
            spectral_similarity = max(0, 100 - (spectral_diff / 50))

            # 4. Energy similarity
            energy_diff = abs(features1['energy_mean'] - features2['energy_mean'])
            energy_similarity = max(0, 100 - (energy_diff * 100))

            # Combined confidence score (weighted average)
            confidence = (
                    mfcc_similarity * 0.6 +  # MFCCs are most important
                    pitch_similarity * 0.2 +  # Pitch is characteristic
                    spectral_similarity * 0.15 +  # Spectral features
                    energy_similarity * 0.05  # Energy patterns
            )

            # Match threshold
            is_match = confidence >= 75.0  # 75% similarity for match

            return confidence, is_match

        except Exception as e:
            print(f"Voice comparison error: {e}")
            return 0.0, False

    def get_device_info(self):
        """Get system information (compatible with fingerprint API)"""

        device_info = {
            'model': self.scanner_type,
            'model_number': self.model_number,
            'is_connected': self.is_connected,
            'sample_rate': f"{self.sample_rate} Hz",
            'duration': f"{self.duration} seconds",
            'implementation': 'Voice Biometric (MFCC + Pitch)'
        }

        if self.is_connected:
            try:
                default_device = sd.query_devices(kind='input')
                device_info['microphone'] = default_device['name']
            except:
                device_info['microphone'] = 'Unknown'

        return device_info

    def test_connection(self):
        """Test microphone (compatible with fingerprint API)"""

        if not self.is_connected:
            return False, "Microphone not available"

        try:
            # Quick test recording
            print("Testing microphone (1 second)...")
            test_audio = sd.rec(
                int(1 * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32'
            )
            sd.wait()

            # Check if audio was captured
            if np.max(np.abs(test_audio)) > 0.01:
                return True, "Microphone working - ready for voice capture"
            else:
                return False, "Microphone detected but no audio signal"

        except Exception as e:
            return False, f"Microphone test failed: {e}"

    def close(self):
        """Release resources"""
        print("Voice biometric system closed")


# Test code
if __name__ == "__main__":
    print("=" * 60)
    print("Voice Biometric System Test")
    print("=" * 60 + "\n")

    try:
        # Initialize system
        voice_system = VoiceBiometricSystem(duration=3)

        if not voice_system.is_connected:
            print("\nâŒ Microphone not available")
            exit(1)

        # Test microphone
        success, message = voice_system.test_connection()
        print(f"Microphone test: {message}\n")

        if success:
            # Enrollment
            print("ENROLLMENT TEST")
            print("-" * 60)
            input("Press Enter to record enrollment sample...")

            enrollment_result = voice_system.capture_fingerprint()

            print(f"\nâœ“ Enrollment complete")
            print(f"  Quality: {enrollment_result['quality_score']:.1f}/100")

            # Save enrollment template
            enrollment_template = enrollment_result['template']

            # Verification
            print("\n\nVERIFICATION TEST")
            print("-" * 60)
            print("Now speak again to verify your identity")
            input("Press Enter to record verification sample...")

            verification_result = voice_system.capture_fingerprint()
            verification_template = verification_result['template']

            # Compare
            print("\n\nCOMPARISON")
            print("-" * 60)
            confidence, is_match = voice_system.compare_fingerprints(
                enrollment_template,
                verification_template
            )

            print(f"Confidence: {confidence:.1f}%")
            print(f"Match: {'âœ“ YES' if is_match else 'âœ— NO'}")

            if is_match:
                print("\nðŸŽ‰ Voice verification successful!")
            else:
                print("\nâš ï¸  Voice verification failed")
                print("   (This is normal for a test - voice changes slightly each time)")

        voice_system.close()

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()