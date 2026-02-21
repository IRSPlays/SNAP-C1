# Project: SNAP-C1 Mentor Studio (Windows Desktop App)
## The Universal Human-AI Augmentation Platform

**Vision:** A native Windows application entirely powered by the offline Fractal Recurrent Core (FRC). Unlike cloud models that *replace* human effort by doing the work for them, this application acts as a private, real-time **Tutor, Sparring Partner, and Technical Mentor**. 

It runs transparently in the background, analyzing user workflows (Coding, Art, Filmmaking) and offering constructive guidance, theory, and optimization without stealing the user's agency.

---

## 🏗️ 1. Technical Stack Architecture

To make the app fast, beautiful, and native to Windows, but deeply connected to our PyTorch AI engine:

*   **Frontend UI (The "Glass" Layer): `Tauri` + `React`**
    *   Tauri is a modern alternative to Electron. It uses Rust under the hood to build the window natively, resulting in an app that uses **< 50MB of RAM** (crucial since our AI needs the VRAM).
    *   React (with Tailwind CSS and Framer Motion) will build a beautiful, fluid, dark-mode interface that hovers transparently over the user's active software (VSCode, Photoshop, Premiere Pro).
*   **Backend AI Engine (The Logic Layer): `PyTorch` + `FastAPI`**
    *   Our PyTorch script (`frv2_inference.py`) runs in a hidden background Python process.
    *   A local FastAPI server acts as the bridge. The Tauri React app sends the user's screen context to `localhost:8000`, and the FRC streams back the mentored advice.
*   **Hardware Execution Target:**
    *   **Primary:** Local AMD/Nvidia GPU (via PyTorch DirectML/CUDA).
    *   **Fallback:** Cloud Serverless API (Cerebrium/RunPod) for users lacking an 8GB GPU.
    *   **Future Edge:** Local NPU computing (Hailo-8L) for ultra-low watt visual inference.

---

## 🧠 2. The Multi-Modal Vision System (Future Phase)

Currently, the FRC handles logic vectors. To make it a true mentor, it needs eyes. We will not use massive VRAM-heavy Vision-Language Models (VLMs) like LLaVA. Instead, we use the **Vision-Expert Router Mechanism**:

1.  **Tiny Vision Encoders:** We run ultra-fast, tiny vision models (like SigLIP or MobileViT) that constantly capture screenshots of the user's active window twice a second.
2.  **Vector Translation:** The Vision model translates the screenshot (e.g., a messy Python script in VSCode, or a poorly lit frame in Premiere) into a dense 1024-dimensional math coordinate.
3.  **The Hand-off:** This coordinate is fed directly into the Fractal Recurrent Loop. The AI doesn't "see" pixels; it mathematically loops over the structural relationships of the image data until it understands the context.

---

## 🎯 3. Core "Augmentation" Modes

The app UI will have explicit modes ensuring it never "takes over" the user's work.

### Mode A: "Sparring Partner" (For Ideation & Art)
*   *Scenario:* A user is writing a script in Final Draft or tweaking a melody in Ableton.
*   *Action:* The AI observes the structure. It references an SSD Micro-Expert (`screenwriting.safetensors`).
*   *Output:* A small floating widget pops up: *"Your second act is dragging. The protagonist hasn't made an active choice in 15 pages. Consider introducing a complication related to [Previous Scene Context] here."* It gives actionable theory, not the written text.

### Mode B: "Over-the-Shoulder Tutor" (For Coding & Logic)
*   *Scenario:* The user is typing a complex WebGL shader and makes a syntax error resulting in a black screen.
*   *Action:* The App captures the VSCode window and the Chrome window.
*   *Output:* Instead of pasting the correct code, the AI says: *"You passed a 'float' into a uniform expecting a 'vec2'. GLSL requires both X and Y dimensions. Where do you think the Y dimension should come from in this function?"* It forces the human to solve the problem by guiding their logic.

### Mode C: "Vibe / Bulldozer Mode" (For Tedious Tasks)
*   *Scenario:* The user understands the concept but doesn't want to do the boring manual labor (e.g., writing 50 unit tests, masking a subject in a video).
*   *Action:* The UI shifts entirely. The user commands: *"Automate this."*
*   *Output:* The FRC shifts into pure execution mode, writing the code or triggering the macros instantly. 

---

## 💾 4. The SSD "Skill Marketplace"

To keep the application under 1.5GB of VRAM permanently, we utilize the **MoE Router** we built in `ssd_streamer.py`.

*   When the user downloads the App, the core AI has **amnesia**. It only knows fundamental reasoning (How to loop math).
*   The App features a "Skill Store." 
*   If the user is a Video Editor, they click download on `PremierePro_Expert.safetensors` (50MB) and `ColorGrading_Theory_Expert.safetensors` (50MB).
*   These files sit on their NVMe SSD. The FRC seamlessly PCIe-streams them into VRAM in 50 milliseconds only when the user opens Premiere Pro. 
*   **Result:** A completely personalized, infinitely scalable AI that doesn't bloat the user's hard drive with knowledge they will never use.

---

## 🚀 5. The Phased Launch Plan

*   **Phase 1 (Current):** Bake the `frc_pretrained_core.pt` logic weights on Google Colab (In Progress).
*   **Phase 2:** Train the logic weights on Python execution errors using the `RLFSSandbox` (RunPod A6000).
*   **Phase 3:** Build the Tauri + React Desktop Window UI (The "Glass Layer").
*   **Phase 4:** Write the local Python FastAPI server to bridge the Tauri App to the PyTorch Core.
*   **Phase 5:** Integrate the low-vram Screen-Capture Vision Encoder loop.

*This application is the antithesis of the multi-billion dollar API monopoly. It is a local, private, dynamic extension of the human mind.*
