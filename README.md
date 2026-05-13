To comprehensively evaluate the capabilities of an Image-to-Video (Ref2Video) model, we need to **decouple the generation tasks across multiple dimensions**. With only a single static image of a woman as input, the model must not only maintain Identity (ID) consistency but also handle temporal coherence, physical laws, spatial geometry, and complex semantic adherence.

Here are the 20 test prompts translated into English, divided into 7 core evaluation dimensions. They range from basic to advanced, designed to probe the model's performance boundaries, expose potential "capability mismatches," and test its generalization to unseen scenarios.

### 1. Fine-Grained Facial Control & Micro-Expressions (Micro-Consistency & Emotion)

These tests observe whether the model can generate natural muscle movements and expression changes while maintaining facial proportions.

* **Prompt 1 (Natural Transition):** The woman looks at the camera with a slight, gentle smile, blinking naturally. Her eyes appear soft, focused, and lively.
* **Prompt 2 (Lip Sync & Speech):** The woman looks directly into the camera, her lips opening and closing naturally as if delivering a serious, fluent speech. Her facial muscles show subtle movements synchronized with her articulation.
* **Prompt 3 (Drastic Emotion Shift):** The woman's expression gradually transitions from calm to extreme shock. She slightly opens her mouth, her eyes widen instantly, and her brow furrows tightly.

### 2. Head & Upper Body 3D Coherence (Geometry & Viewpoint Understanding)

Testing the model's understanding of 3D human structure, particularly whether facial distortion or texture degradation occurs during perspective shifts.

* **Prompt 4 (Large Angle Head Turn):** The woman slowly turns her head to the right, looking off-screen into the distance. She pauses for a second, then smoothly turns her head back to look directly at the camera.
* **Prompt 5 (Hair Physics & Occlusion):** A strong gust of wind blows from the side, causing the woman's hair to flutter wildly backwards. She reaches up with her hand to tuck a stray strand of hair behind her ear.
* **Prompt 6 (Hand-Object Interaction):** The woman picks up a steaming cup of coffee, brings it to her lips to take a sip, then lowers the cup and lets out a long sigh. *(Note: Focuses on the stability of finger generation and relative object positioning.)*

### 3. Full-Body Skeletal & Large-Amplitude Motion (Temporal Dynamics & Structural Maintenance)

Extrapolating from a static portrait/half-body to full-body dynamics, testing structural integrity during significant pixel displacement.

* **Prompt 7 (Deep Spatial Movement):** The woman turns around, with her back to the camera, and walks forward along a path covered in fallen leaves. Her silhouette gradually shrinks as she moves further away in perspective.
* **Prompt 8 (Complex Posture Generation):** Following the upbeat rhythm of the music, the woman starts dancing a highly energetic jazz routine. Her limb movements are smooth and natural, with no extra or anatomically incorrect limbs appearing.
* **Prompt 9 (Intense Athletic Motion):** The woman suddenly breaks into a sprint, leaps high into the air to jump over a puddle on the ground, with her clothes billowing dramatically from the motion.

### 4. Physical Laws & Environment Interaction (World Model Capabilities)

Testing whether the model understands basic physics (e.g., fluid dynamics, light reflection, material properties).

* **Prompt 10 (Fluid Dynamics & Material Change):** A heavy downpour suddenly begins, with dense raindrops hitting the woman's face and clothes. The fabric's material gradually darkens as it gets soaked, and water droplets glide smoothly down her cheeks.
* **Prompt 11 (Dynamic Point Source Lighting):** In a pitch-dark environment, the woman strikes a match. The warm orange flame instantly illuminates her face. The flame flickers in the breeze, causing dynamic shifts in the lighting and shadows on her skin.
* **Prompt 12 (Fine Object Manipulation):** The woman opens a thick book and looks down to read it carefully. Her fingers gently swipe across the pages, causing natural bending and shifting shadows as the pages turn.

### 5. Camera Control & Spatial Perception (Cinematography)

Testing whether the model can follow instructions for smooth camera trajectories while the subject is moving or stationary.

* **Prompt 13 (Zoom-in):** The camera slowly and smoothly zooms in, transitioning from a medium shot of the woman's upper body to an extreme close-up focusing entirely on her eyes.
* **Prompt 14 (Tracking/Arc Shot):** The camera performs a 360-degree tracking arc shot around the woman. She remains standing perfectly still, while the background undergoes correct perspective shifts as the camera orbits her.
* **Prompt 15 (Tilt Up):** The camera starts with a tight shot on the woman's hands, then slowly tilts up, moving across her torso, and finally rests on her confident face.

### 6. Background Dynamics & Foreground Decoupling (Foreground-Background Separation)

Testing the model's ability to distinguish the foreground subject from the background and control their dynamics independently.

* **Prompt 16 (Time-lapse Background):** The woman stands perfectly still as if time is frozen, but the background behind her rapidly cycles through day and night, with clouds streaking across the sky in a time-lapse effect.
* **Prompt 17 (Scene Transition/Cut):** The woman takes a step forward. Synchronized with her movement, the background instantly and seamlessly transitions from a bustling modern city street to a tranquil beach.
* **Prompt 18 (Motion Contrast):** The woman stands perfectly still in the center of the frame. Around her, blurred silhouettes of people are walking back and forth rapidly (motion blur), creating a strong contrast between stillness and fast motion.

### 7. Extreme Style & Lighting Render Testing (Render Fidelity)

Testing the model's simulation capabilities for extreme visual styles and complex light path tracing.

* **Prompt 19 (Cyberpunk Dual-Tone Lighting):** Cyberpunk-style neon lights (highly saturated reds and blues) alternately cast on the woman's face. Her skin and hair reflect the colors with precise, highly detailed glossy highlights.
* **Prompt 20 (Specular Reflection):** The woman puts on a pair of highly reflective sunglasses. The lenses clearly reflect a spectacular fireworks display exploding right in front of her, with accurate spherical perspective distortion in the reflection.

---

Among these specific dimensions, which one are you noticing the most significant capability drop-off with your current model architecture?
