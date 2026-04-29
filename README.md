## 🚗 Controller-Free Driving Simulator

This project is a **controller-free driving simulator system** that leverages **computer vision** to emulate real driving inputs without the need for physical controllers.

### ✨ Features

- 🖐️ **Steering control via hand tracking**
- 🦶 **Acceleration and braking using printed paper pedals**
- 📷 Vision-based input detection (no hardware controllers required)
- 📡 Real-time communication via sockets
- 🎮 Integration with a custom driving simulator

### 🧠 How It Works

A computer vision pipeline tracks:

- Hand position and movement → steering input  
- Interaction with printed pedal templates → gas and brake  

These inputs are processed and converted into commands, which are then sent via sockets to a driving simulator built using the Godot Game Engine

### 🕹️ Simulator

The driving simulator is being developed from scratch using the **Godot Engine**, allowing full control over:

- Physics simulation  
- Input handling  
- Environment design  

### 🚧 Project Status

> ⚠️ This project is currently a **work in progress**.  
> Features may change as development progresses.


## 👨‍🎓 Academic Context

This project was developed as part of  **HUMAN COMPUTER INTERACTION ** course, academic year 2025-2026.
