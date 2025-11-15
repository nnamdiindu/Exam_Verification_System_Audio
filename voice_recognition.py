import sounddevice as sd
import numpy as np
import librosa
from scipy.signal import butter, lfilter
import json
import base64
import hashlib
from datetime import datetime


class VoiceBiometricSystem:
    """Voice biometric system using MFCC features"""

    def __init__(self, sample_rate=16000, duration=5):
        self.sample_rate = sample_rate
        self.duration = duration  # Changed from 3 to 5 seconds
        self.scanner_type = "Voice Biometric"
        self.is_connected = self._check_microphone()

        # Audio processing parameters
        self.n_mfcc = 13
        self.n_fft = 2048
        self.hop_length = 512

        # Matching thresholds
        self.match_threshold = 0.65  # 65% match required
        self.min_quality_score = 50  # 50/100 minimum

        # Silence detection (very lenient)
        self.silence_threshold = 0.003
        self.min_speech_ratio = 0.10

        print(f"✓ Voice Biometric System initialized")
        print(f"  Sample rate: {self.sample_rate} Hz")
        print(f"  Duration: {self.duration} seconds")
        print(f"  Microphone: {'Connected' if self.is_connected else 'Not detected'}")

    def _check_microphone(self):
        """Check if microphone is available"""
        try:
            devices = sd.query_devices()
            input_devices = [d for d in devices if d['max_input_channels'] > 0]

            if input_devices:
                default_input = sd.query_devices(kind='input')
                print(f"✓ Microphone found: {default_input['name']}")
                return True
            else:
                print("✗ No microphone detected")
                return False
        except Exception as e:
            print(f"✗ Microphone check failed: {e}")
            return False

    def _bandpass_filter(self, data, lowcut=80, highcut=3000):
        """Apply bandpass filter to focus on speech frequencies"""
        nyquist = 0.5 * self.sample_rate
        low = lowcut / nyquist
        high = highcut / nyquist

        b, a = butter(4, [low, high], btype='band')
        filtered_data = lfilter(b, a, data)
        return filtered_data

    def _record_audio(self, prompt_message="🎤 Please speak now"):
        """Record audio from microphone"""
        try:
            print("\n" + "=" * 60)
            print("VOICE BIOMETRIC CAPTURE")
            print("=" * 60)
            print(f"{prompt_message}")
            print(f"   Recording for {self.duration} seconds...")
            print("   Say: 'My name is [Your Name] and my registration number is [Number]'")
            print("   Recording starts in:")

            # Countdown
            for i in range(3, 0, -1):
                print(f"   {i}...")
                sd.sleep(1000)

            print("   🔴 RECORDING NOW - Speak clearly!")

            # Record audio
            audio_data = sd.rec(
                int(self.duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32'
            )
            sd.wait()

            print("   ✓ Recording complete!")

            # Flatten to 1D array
            audio_data = audio_data.flatten()

            # Better audio validation
            max_amplitude = np.max(np.abs(audio_data))
            mean_amplitude = np.mean(np.abs(audio_data))
            rms = np.sqrt(np.mean(audio_data ** 2))

            print(f"\n📊 Audio Analysis:")
            print(f"   Max amplitude: {max_amplitude:.6f}")
            print(f"   Mean amplitude: {mean_amplitude:.6f}")
            print(f"   RMS level: {rms:.6f}")

            # Very lenient checks
            if max_amplitude < self.silence_threshold:
                print(f"   ⚠️  Audio too quiet (threshold: {self.silence_threshold})")
                print(f"   💡 TIP: Increase microphone volume in system settings")
                raise Exception(f"Audio too quiet ({max_amplitude:.6f})")

            # Check for speech activity
            speech_frames = np.sum(np.abs(audio_data) > self.silence_threshold)
            speech_ratio = speech_frames / len(audio_data)
            print(f"   Speech ratio: {speech_ratio:.2%} (need {self.min_speech_ratio:.0%})")

            if speech_ratio < self.min_speech_ratio:
                print(f"   ⚠️  Insufficient speech detected")
                raise Exception(f"Insufficient speech ({speech_ratio:.1%})")

            # Apply preprocessing
            audio_data = self._bandpass_filter(audio_data)

            # Normalize
            if max_amplitude > 0:
                audio_data = audio_data / max_amplitude

            print("   ✓ Audio validation passed")
            print("=" * 60 + "\n")

            return audio_data

        except sd.PortAudioError as e:
            raise Exception(f"Microphone error: {e}")
        except Exception as e:
            raise Exception(f"Audio recording failed: {e}")

    def _extract_mfcc_features(self, audio_data):
        """Extract MFCC features from audio"""
        try:
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(
                y=audio_data,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )

            # Calculate statistics for each MFCC coefficient
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            mfcc_delta = librosa.feature.delta(mfccs)
            mfcc_delta_mean = np.mean(mfcc_delta, axis=1)

            # Combine features (39 dimensions total)
            features = np.concatenate([mfcc_mean, mfcc_std, mfcc_delta_mean])

            return features.tolist(), mfccs.tolist()

        except Exception as e:
            raise Exception(f"Feature extraction failed: {e}")

    def _extract_pitch(self, audio_data):
        """Extract pitch (fundamental frequency) from audio"""
        try:
            pitches, magnitudes = librosa.piptrack(
                y=audio_data,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )

            # Get pitch values where magnitude is highest
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:  # Valid pitch
                    pitch_values.append(pitch)

            if pitch_values:
                pitch_mean = np.mean(pitch_values)
                pitch_std = np.std(pitch_values)
                return float(pitch_mean), float(pitch_std)
            else:
                return 0.0, 0.0

        except Exception as e:
            print(f"Pitch extraction warning: {e}")
            return 0.0, 0.0

    def _calculate_quality_score(self, audio_data, mfcc_features):
        """
        FIXED QUALITY SCORING - Based on actual audio characteristics
        """
        try:
            print(f"\n🎯 Quality Score Calculation:")

            # Component 1: Amplitude level (40 points)
            max_amplitude = np.max(np.abs(audio_data))
            rms = np.sqrt(np.mean(audio_data ** 2))

            # Score based on RMS (more stable than max)
            if rms > 0.025:  # Excellent (your value: 0.033)
                amplitude_score = 40
            elif rms > 0.015:  # Good
                amplitude_score = 35
            elif rms > 0.008:  # Acceptable
                amplitude_score = 30
            elif rms > 0.003:  # Low but usable
                amplitude_score = 25
            else:  # Very low
                amplitude_score = 20

            print(f"   Amplitude (RMS {rms:.6f}): {amplitude_score}/40 points")

            # Component 2: Speech activity (30 points)
            speech_frames = np.sum(np.abs(audio_data) > self.silence_threshold)
            speech_ratio = speech_frames / len(audio_data)

            # Score based on speech percentage
            if speech_ratio > 0.30:  # Excellent (your value: 0.37)
                speech_score = 30
            elif speech_ratio > 0.20:  # Good
                speech_score = 27
            elif speech_ratio > 0.15:  # Acceptable
                speech_score = 23
            elif speech_ratio > 0.10:  # Minimum
                speech_score = 20
            else:  # Too little
                speech_score = 15

            print(f"   Speech activity ({speech_ratio:.2%}): {speech_score}/30 points")

            # Component 3: Feature quality (30 points)
            # Check if MFCCs are well-formed
            mfcc_mean = np.mean(mfcc_features)
            mfcc_std = np.std(mfcc_features)
            mfcc_range = np.max(mfcc_features) - np.min(mfcc_features)

            # Good MFCCs should have reasonable variation
            if mfcc_std > 0.5 and mfcc_range > 1.0:  # Well-formed features
                feature_score = 30
            elif mfcc_std > 0.3:  # Decent features
                feature_score = 25
            elif mfcc_std > 0.1:  # Weak features
                feature_score = 20
            else:  # Very weak
                feature_score = 15

            print(f"   Feature quality (std {mfcc_std:.4f}): {feature_score}/30 points")

            # Total score
            total_score = amplitude_score + speech_score + feature_score

            print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"   TOTAL QUALITY: {total_score}/100")

            if total_score >= 85:
                print(f"   ✅ EXCELLENT - High quality recording")
            elif total_score >= 70:
                print(f"   ✅ GOOD - Should work well")
            elif total_score >= 60:
                print(f"   ✅ ACCEPTABLE - Should work")
            elif total_score >= 50:
                print(f"   ⚠️  MINIMUM - May work but not ideal")
            else:
                print(f"   ❌ LOW - May cause verification issues")

            return float(total_score)

        except Exception as e:
            print(f"Quality calculation error: {e}")
            return 75.0

    def capture_fingerprint(self):
        """Capture voice biometric sample"""
        try:
            if not self.is_connected:
                raise Exception("Microphone not connected")

            # Record audio
            audio_data = self._record_audio(
                "🎤 ENROLLMENT: Please speak clearly"
            )

            # Extract features
            mfcc_features, mfcc_raw = self._extract_mfcc_features(audio_data)
            pitch_mean, pitch_std = self._extract_pitch(audio_data)

            # Calculate quality
            quality_score = self._calculate_quality_score(audio_data, mfcc_features)

            # Create template (as dictionary)
            template = {
                'features': mfcc_features,
                'mfcc_raw': mfcc_raw,
                'pitch_mean': float(pitch_mean),  # Ensure it's a float
                'pitch_std': float(pitch_std),  # Ensure it's a float
                'duration': self.duration,
                'sample_rate': self.sample_rate,
                'timestamp': datetime.now().isoformat()
            }

            # CRITICAL: Convert to JSON STRING for storage
            template_json = json.dumps(template)

            # CRITICAL: Encode as UTF-8 bytes (not raw audio bytes!)
            template_bytes = template_json.encode('utf-8')

            # Also create base64 for 'image_data' field (backward compatibility)
            audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

            print(f"\n✅ Voice sample captured successfully!")
            print(f"   Quality: {quality_score:.1f}/100")
            print(f"   Pitch: {pitch_mean:.1f} Hz")
            print(f"   Features: {len(mfcc_features)} dimensions")
            print(f"   Template size: {len(template_bytes)} bytes (UTF-8 JSON)\n")

            return {
                'template': template_json,  # JSON string for comparison
                'template_bytes': template_bytes,  # UTF-8 encoded for DB storage
                'quality_score': quality_score,
                'image_data': audio_b64,  # For backward compatibility
                'template_data': template,  # Dictionary for reference
                'pitch_mean': pitch_mean
            }

        except Exception as e:
            print(f"\n✗ Voice capture error: {e}")
            import traceback
            traceback.print_exc()
            raise Exception(f"Voice capture failed: {e}")

    def compare_fingerprints(self, template1, template2):
        """Compare two voice templates using cosine similarity"""
        try:
            # Parse templates
            if isinstance(template1, str):
                t1 = json.loads(template1)
            else:
                t1 = template1

            if isinstance(template2, str):
                t2 = json.loads(template2)
            else:
                t2 = template2

            # Extract feature vectors
            features1 = np.array(t1['features'])
            features2 = np.array(t2['features'])

            # Cosine similarity
            dot_product = np.dot(features1, features2)
            norm1 = np.linalg.norm(features1)
            norm2 = np.linalg.norm(features2)

            if norm1 == 0 or norm2 == 0:
                return 0.0, False

            similarity = dot_product / (norm1 * norm2)

            # Pitch comparison (additional check)
            pitch1 = t1.get('pitch_mean', 0)
            pitch2 = t2.get('pitch_mean', 0)

            if pitch1 > 0 and pitch2 > 0:
                pitch_diff = abs(pitch1 - pitch2) / max(pitch1, pitch2)
                pitch_similarity = max(0, 1 - pitch_diff)
            else:
                pitch_similarity = 0.5

            # Combined score (70% MFCC, 30% pitch)
            combined_score = (similarity * 0.7 + pitch_similarity * 0.3) * 100

            # Match decision
            is_match = combined_score >= (self.match_threshold * 100)

            return combined_score, is_match

        except Exception as e:
            print(f"Comparison error: {e}")
            return 0.0, False

    def get_scanner_info(self):
        """Get scanner information"""
        return {
            'name': self.scanner_type,
            'version': '2.1.5',
            'is_connected': self.is_connected,
            'sample_rate': self.sample_rate,
            'duration': self.duration,
            'features': f'{self.n_mfcc}-dimensional MFCC'
        }