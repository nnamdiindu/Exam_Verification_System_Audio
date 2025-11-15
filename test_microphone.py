"""
Audio Diagnostic Tool - Run this to test your microphone
Save as: test_microphone.py
"""

import sounddevice as sd
import numpy as np
import librosa
import matplotlib.pyplot as plt
from datetime import datetime


def test_microphone():
    """Comprehensive microphone diagnostic"""

    print("=" * 70)
    print("VOICE BIOMETRIC - MICROPHONE DIAGNOSTIC TOOL")
    print("=" * 70)

    # 1. List all audio devices
    print("\n1️⃣  AVAILABLE AUDIO DEVICES:")
    print("-" * 70)
    devices = sd.query_devices()

    input_devices = []
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            input_devices.append(i)
            is_default = "← DEFAULT" if i == sd.default.device[0] else ""
            print(f"   [{i}] {device['name']} {is_default}")
            print(f"       Channels: {device['max_input_channels']}, Sample Rate: {device['default_samplerate']} Hz")

    if not input_devices:
        print("   ❌ ERROR: No input devices found!")
        return

    print(f"\n   ℹ️  Using device [{sd.default.device[0]}] for testing")

    # 2. Test recording
    print("\n2️⃣  RECORDING TEST (5 seconds - LONGER FOR COMFORT)")
    print("-" * 70)
    print("   🎤 Recording in:")
    for i in range(3, 0, -1):
        print(f"      {i}...")
        sd.sleep(1000)

    print("   🔴 RECORDING NOW - Speak comfortably!")
    print("   💡 Say: 'My name is [Your Name] and my registration number is [Your Number]'")

    sample_rate = 16000
    duration = 5  # CHANGED FROM 3 TO 5 SECONDS

    try:
        audio_data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        audio_data = audio_data.flatten()

        print("   ✅ Recording complete!")

    except Exception as e:
        print(f"   ❌ Recording failed: {e}")
        return

    # 3. Analyze audio
    print("\n3️⃣  AUDIO ANALYSIS")
    print("-" * 70)

    # Basic statistics
    max_amplitude = np.max(np.abs(audio_data))
    mean_amplitude = np.mean(np.abs(audio_data))
    rms = np.sqrt(np.mean(audio_data ** 2))

    print(f"   📊 Raw Audio Statistics:")
    print(f"      Max amplitude: {max_amplitude:.6f}")
    print(f"      Mean amplitude: {mean_amplitude:.6f}")
    print(f"      RMS level: {rms:.6f}")
    print(f"      Dynamic range: {20 * np.log10(max_amplitude + 1e-10):.2f} dB")

    # Check for issues
    print(f"\n   🔍 Issue Detection:")

    if max_amplitude < 0.001:
        print(f"      ❌ CRITICAL: Audio is extremely quiet (max: {max_amplitude:.6f})")
        print(f"         → Microphone may be muted or not selected")
        print(f"         → Check system sound settings")
    elif max_amplitude < 0.01:
        print(f"      ⚠️  WARNING: Audio is very quiet (max: {max_amplitude:.6f})")
        print(f"         → Increase microphone volume/boost")
        print(f"         → Speak closer to microphone")
    elif max_amplitude < 0.05:
        print(f"      ⚠️  Audio is somewhat quiet (max: {max_amplitude:.6f})")
        print(f"         → Consider increasing microphone gain")
    else:
        print(f"      ✅ Audio level is good (max: {max_amplitude:.6f})")

    # Speech activity detection
    silence_threshold = 0.005
    speech_frames = np.sum(np.abs(audio_data) > silence_threshold)
    speech_ratio = speech_frames / len(audio_data)

    print(f"\n   🗣️  Speech Detection:")
    print(f"      Speech ratio: {speech_ratio:.2%}")
    print(f"      Silence threshold: {silence_threshold}")

    if speech_ratio < 0.10:
        print(f"      ❌ CRITICAL: Very little speech detected ({speech_ratio:.2%})")
        print(f"         → Speak louder and for the full 3 seconds")
    elif speech_ratio < 0.20:
        print(f"      ⚠️  WARNING: Low speech activity ({speech_ratio:.2%})")
        print(f"         → Speak more continuously")
    else:
        print(f"      ✅ Good speech activity ({speech_ratio:.2%})")

    # 4. Calculate quality score (FIXED ALGORITHM)
    print("\n4️⃣  QUALITY SCORE CALCULATION (FIXED)")
    print("-" * 70)

    # Component 1: Amplitude level (RMS-based) - 40 points
    rms = np.sqrt(np.mean(audio_data ** 2))

    if rms > 0.025:  # Excellent (your value: 0.044)
        amplitude_score = 40
    elif rms > 0.015:  # Good
        amplitude_score = 35
    elif rms > 0.008:  # Acceptable
        amplitude_score = 30
    elif rms > 0.003:  # Low but usable
        amplitude_score = 25
    else:  # Very low
        amplitude_score = 20

    print(f"   📈 Amplitude Score:")
    print(f"      RMS level: {rms:.6f}")
    print(f"      Score: {amplitude_score}/40 points")

    # Component 2: Speech activity - 30 points
    silence_threshold = 0.003
    speech_frames = np.sum(np.abs(audio_data) > silence_threshold)
    speech_ratio = speech_frames / len(audio_data)

    if speech_ratio > 0.30:  # Excellent
        speech_score = 30
    elif speech_ratio > 0.20:  # Good (your value: 0.28)
        speech_score = 27
    elif speech_ratio > 0.15:  # Acceptable
        speech_score = 23
    elif speech_ratio > 0.10:  # Minimum
        speech_score = 20
    else:  # Too little
        speech_score = 15

    print(f"\n   🗣️  Speech Activity:")
    print(f"      Speech ratio: {speech_ratio:.2%}")
    print(f"      Score: {speech_score}/30 points")

    # Component 3: Feature consistency - 30 points
    try:
        mfccs = librosa.feature.mfcc(
            y=audio_data,
            sr=sample_rate,
            n_mfcc=13
        )
        mfcc_features = np.mean(mfccs, axis=1)
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

        print(f"\n   📊 Feature Quality:")
        print(f"      MFCC std: {mfcc_std:.4f}")
        print(f"      MFCC range: {mfcc_range:.4f}")
        print(f"      Score: {feature_score}/30 points")
    except Exception as e:
        print(f"\n   ⚠️  Could not calculate MFCC features: {e}")
        feature_score = 20

    # Total quality (FIXED)
    total_quality = amplitude_score + speech_score + feature_score
    print(f"\n   🎯 TOTAL QUALITY SCORE: {total_quality:.2f}/100")

    if total_quality >= 85:
        print(f"      ✅ EXCELLENT - High quality recording")
    elif total_quality >= 70:
        print(f"      ✅ GOOD - Should work well")
    elif total_quality >= 60:
        print(f"      ✅ ACCEPTABLE - Should work")
    elif total_quality >= 50:
        print(f"      ⚠️  MINIMUM - May work but not ideal")
    else:
        print(f"      ❌ LOW - May cause verification issues")

    # 5. Pitch analysis
    print("\n5️⃣  PITCH ANALYSIS")
    print("-" * 70)

    try:
        pitches, magnitudes = librosa.piptrack(
            y=audio_data,
            sr=sample_rate
        )

        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)

        if pitch_values:
            pitch_mean = np.mean(pitch_values)
            pitch_std = np.std(pitch_values)
            print(f"   📈 Pitch detected:")
            print(f"      Mean pitch: {pitch_mean:.1f} Hz")
            print(f"      Pitch variation: {pitch_std:.1f} Hz")
            print(f"      Pitch samples: {len(pitch_values)}")

            # Typical ranges
            if pitch_mean < 80:
                print(f"      ℹ️  Very low pitch (possible bass/male voice)")
            elif pitch_mean < 180:
                print(f"      ℹ️  Male voice range")
            elif pitch_mean < 250:
                print(f"      ℹ️  Female voice range")
            else:
                print(f"      ℹ️  High pitch")
        else:
            print(f"   ⚠️  No pitch detected - audio may be too quiet or noisy")

    except Exception as e:
        print(f"   ⚠️  Pitch analysis failed: {e}")

    # 6. Recommendations
    print("\n6️⃣  RECOMMENDATIONS")
    print("-" * 70)

    recommendations = []

    if rms < 0.008:
        recommendations.append("🔧 CRITICAL: Increase microphone volume in Windows Sound Settings")
        recommendations.append("   → Right-click speaker icon → Sounds → Recording")
        recommendations.append("   → Select your microphone → Properties → Levels")
        recommendations.append("   → Set to 80-100% and enable Microphone Boost (+20dB)")

    if speech_ratio < 0.20:
        recommendations.append("🗣️  Speak continuously for the full 5 seconds")
        recommendations.append("   → Don't pause between words")
        recommendations.append("   → Speak at normal conversation volume")

    if rms < 0.015:
        recommendations.append("📍 Position microphone 5-10cm from mouth")
        recommendations.append("   → Not too close (distortion) or too far (quiet)")

    if total_quality < 60:
        recommendations.append("🔄 Try a different microphone/headset if available")
        recommendations.append("   → Built-in laptop mics often produce low quality")
    elif total_quality >= 85:
        recommendations.append("✅ Excellent audio quality! Your microphone setup is perfect.")
        recommendations.append("   → Your system should work flawlessly")
    elif total_quality >= 70:
        recommendations.append("✅ Good audio quality! Should work well for voice biometrics.")

    if len(recommendations) == 0:
        recommendations.append("✅ Audio quality is good! Your system should work fine.")

    for rec in recommendations:
        print(f"   {rec}")

    # 7. Save audio for inspection
    print("\n7️⃣  SAVING DIAGNOSTIC FILES")
    print("-" * 70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save audio as WAV
    try:
        import scipy.io.wavfile as wavfile
        filename = f"audio_test_{timestamp}.wav"
        audio_int16 = (audio_data * 32767).astype(np.int16)
        wavfile.write(filename, sample_rate, audio_int16)
        print(f"   💾 Audio saved: {filename}")
        print(f"      → Listen to this file to hear what was recorded")
    except Exception as e:
        print(f"   ⚠️  Could not save audio: {e}")

    # Save plot
    try:
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))

        # Waveform
        time = np.arange(len(audio_data)) / sample_rate
        axes[0].plot(time, audio_data)
        axes[0].set_title('Waveform')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True)
        axes[0].axhline(y=0.01, color='r', linestyle='--', label='Minimum threshold')
        axes[0].axhline(y=-0.01, color='r', linestyle='--')
        axes[0].legend()

        # Spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
        img = axes[1].imshow(D, aspect='auto', origin='lower',
                             extent=[0, duration, 0, sample_rate / 2])
        axes[1].set_title('Spectrogram')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Frequency (Hz)')
        plt.colorbar(img, ax=axes[1], format='%+2.0f dB')

        # Energy over time
        frame_length = 2048
        hop_length = 512
        energy = np.array([
            np.sum(audio_data[i:i + frame_length] ** 2)
            for i in range(0, len(audio_data) - frame_length, hop_length)
        ])
        time_frames = np.arange(len(energy)) * hop_length / sample_rate
        axes[2].plot(time_frames, energy)
        axes[2].set_title('Energy over Time')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Energy')
        axes[2].grid(True)

        plt.tight_layout()
        plot_filename = f"audio_analysis_{timestamp}.png"
        plt.savefig(plot_filename, dpi=100)
        print(f"   📊 Analysis plot saved: {plot_filename}")
        plt.close()
    except Exception as e:
        print(f"   ⚠️  Could not save plot: {e}")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Review the recommendations above")
    print("2. Listen to the saved audio file to verify quality")
    print("3. Check the analysis plot for visual inspection")
    print("4. Re-run this test after making adjustments")
    print("\n")


if __name__ == "__main__":
    try:
        test_microphone()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()